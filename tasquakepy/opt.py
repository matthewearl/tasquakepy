import argparse
import concurrent.futures
import dataclasses
import json
import logging
import multiprocessing
import os
import pathlib
import pickle
import threading
from typing import Optional

import matplotlib.pyplot as plt
import nevergrad as ng
import numpy as np
import pyade.ilshade
import pyade.jade
import pyade.jso
import scipy.optimize
import wandb
import yaml

from . import qlib


# Optional imports
try:
    import pyswarms
except ImportError:
    pyswarms = None


logger = logging.getLogger(__name__)
THREAD_LOCAL_DATA = threading.local()
YAW_CVAR = "tas_strafe_yaw"
MAXFPS_CVAR = "cl_maxfps"


def _get_thread_data():
    # Reset thread local data over forks.
    pid = os.getpid()
    if getattr(THREAD_LOCAL_DATA, 'pid', None) != pid:
        THREAD_LOCAL_DATA.pid = pid
        if hasattr(THREAD_LOCAL_DATA, 'data'):
            del THREAD_LOCAL_DATA.data

    if not hasattr(THREAD_LOCAL_DATA, 'data'):
        THREAD_LOCAL_DATA.data = {}

    return THREAD_LOCAL_DATA.data
        

def _get_qlib(library_path, base_dir):
    thread_data = _get_thread_data()
    if 'quake' not in thread_data:
        logger.info(f"Loading {library_path} {THREAD_LOCAL_DATA} {threading.get_native_id()}")
        thread_data['quake'] = qlib.Quake(threading.get_native_id(), library_path, base_dir)
    return thread_data['quake']


@dataclasses.dataclass
class TasOpt:
    library_path: pathlib.Path
    base_dir: pathlib.Path
    cfg: dict

    _block_frames: Optional[np.ndarray] = dataclasses.field(init=False, default=None)
    _yaw_block_nums: Optional[np.ndarray] = dataclasses.field(init=False, default=None)
    _iter_num: Optional[int] = dataclasses.field(init=False, default=None)
    _all_time_min_energy: Optional[float] = dataclasses.field(init=False, default=None)
    _log_dir: Optional[pathlib.Path] = dataclasses.field(init=False, default=None)

    def _callback_jade(self, *, population, fitness, u_cr, u_f, **kwargs):
        return self._callback(population, fitness, {'u_cr': u_cr, 'u_f': u_f})

    def _callback_ilshade(self, *, population, fitness, m_cr, m_f, current_size, num_evals,
                          indexes, **kwargs):
        return self._callback(population, fitness,
                              {'current_size': current_size,
                               'num_evals': num_evals,
                               'num_success': len(indexes),
                               **{f'm_f[{i}]': v for i, v in enumerate(m_f)},
                               **{f'm_cr[{i}]': v for i, v in enumerate(m_cr)},
                               })

    def _callback_nelder_mead(self, *args, **kwargs):
        print(f'callback: {args=} {kwargs=}')
        print(f'fun: {self._objective(args[0])}')

    def _callback(self, population, fitness, extras):
        iter_num = self._iter_num

        best_idx = np.argmin(fitness)
        feasible = ~np.isinf(fitness)
        frac_feasible = np.mean(feasible)
        mean_energy = np.mean(fitness[feasible])
        std_energy = np.std(fitness[feasible])
        mean_std_params = np.mean(np.std(population, axis=0))
        min_energy = fitness[best_idx]
        best_params = population[best_idx]
        logger.info(f'{iter_num=} {min_energy=} {frac_feasible=} {mean_energy=} {std_energy=} '
                    f'{mean_std_params=} {extras=}\n{best_params=}')
        if iter_num % 10 == 0:
            log_dict = {'min_energy': min_energy,
                        'iter_num': iter_num,
                        'frac_feasible': frac_feasible,
                        'mean_energy': mean_energy,
                        'std_energy': std_energy,
                        'mean_std_params': mean_std_params,
                        **extras}
            if iter_num % 500 == 0:
                # It takes a long time to plot, so only do it periodically.
                plt.imshow(np.corrcoef(population.T), cmap='seismic', vmin=-1, vmax=1)
                log_dict['corr'] = plt
            wandb.log(log_dict)

        # If this is a new all time best, dump it.
        atb_dir = self._log_dir / 'all_time_bests'
        atb_dir.mkdir(exist_ok=True)
        if min_energy < self._all_time_min_energy:
            with (atb_dir / f'{iter_num}_{min_energy}.pkl').open('wb') as f:
                pickle.dump(best_params, f)
            self._all_time_min_energy = min_energy

        # Periodically dump the entire population.
        pops_dir = self._log_dir / 'pops'
        pops_dir.mkdir(parents=True, exist_ok=True)
        if iter_num % 100 == 0:
            with (pops_dir / f'{iter_num}.pkl').open('wb') as f:
                pickle.dump((fitness, population), f)

        self._iter_num = iter_num + 1

    def _setup_script(self, q, params):
        cfg = self.cfg
        frame_params, yaw_params = (params[:cfg['num_frame_params']],
                                    params[cfg['num_frame_params']:])
        assert len(yaw_params) == cfg['num_yaw_params']

        # Only load the script once --- reloading erases save states.
        thread_data = _get_thread_data()
        if thread_data.get('loaded_script') != cfg['tas_script']:
            print(f'loading script {threading.get_native_id()}')
            q.load_tas_script(cfg['tas_script'])

            # Remove the FPS trick for a fair comparison
            for bl in reversed(q.blocks):
                if MAXFPS_CVAR in bl and bl[MAXFPS_CVAR] == 10:
                    logger.info(f'{threading.get_native_id()}: Removed FPS trick from block {bl.block_num}')
                    bl[MAXFPS_CVAR] = 72
                    break
            else:
                raise Exception("FPS trick not found")

            thread_data['loaded_script'] = cfg['tas_script']

        q.set_cvar_vals(YAW_CVAR, self._yaw_block_nums, yaw_params)

        new_block_frames = self._block_frames.copy()
        if cfg['delta_frames']:
            frame_params = np.cumsum(frame_params)
        frame_params = np.maximum.accumulate(frame_params) # make monotonic
        new_block_frames[-cfg['num_frame_params']:] = frame_params
        q.set_block_frames(new_block_frames)

    def _objective(self, params):
        cfg = self.cfg

        q = _get_qlib(self.library_path, self.base_dir)
        self._setup_script(q, params)

        if cfg.get('skip', False):
            # Work out where to skip to, and skip to it.
            skip_frame = min(
                self._block_frames[self._yaw_block_nums[0]],
                self._block_frames[-cfg['num_frame_params'] - 1]
            )
            q.play_tas_script(skip_frame, save_state=True)
        else:
            # Otherwise just play from the start
            skip_frame = 0
            q.play_tas_script()

        for _ in range(cfg['num_frames'] - skip_frame):
            _ = q.step_no_cmd()
            if q.exact_completed_time is not None:
                break

        q.stop_tas_script()

        exact_completed_time = q.exact_completed_time
        if exact_completed_time is None:
            exact_completed_time = np.inf

        if cfg['objective'] == 'exact_finish_time':
            return exact_completed_time
        elif cfg['objective'] == 'feasible':
            return 1 if np.isinf(exact_completed_time) else 0
        else:
            raise Exception(f"invalid objective {cfg['objective']}")

    def _run_pyade(self, num_dims, pyade_module, pyade_param_updates, cb):
        with multiprocessing.Pool(self.cfg['num_workers']) as pool:
            pyade_params = pyade_module.get_default_params(num_dims)
            pyade_params.update(pyade_param_updates)
            best_params, _ = pyade_module.apply(**{**pyade_params, 'callback': cb, 'opts': {'pool': pool}})
        return best_params

    @pyade.vectorized
    def _vector_objective(self, population, opts):
        return np.array(opts['pool'].map(self._objective, population))

    def _ng_objective(self, normed_params):
        params = normed_params * self._std + self._mean
        value = self._objective(params)
        if np.isinf(value):
            value = 25
        return value

    def _load_script(self):
        """Load the configured tas script, and return the parameters within it"""
        cfg = self.cfg
        q = _get_qlib(self.library_path, self.base_dir)
        q.load_tas_script(cfg['tas_script'])
        self._yaw_block_nums, base_yaw_vals = q.get_cvar_vals(YAW_CVAR)
        self._yaw_block_nums = self._yaw_block_nums[-cfg['num_yaw_params']:]
        base_yaw_vals = base_yaw_vals[-cfg['num_yaw_params']:]
        self._block_frames = q.get_block_frames()
        base_frames = self._block_frames[-cfg['num_frame_params']:]
        if cfg['delta_frames']:
            base_frames = np.diff(base_frames, prepend=0)
        base_params = np.concatenate([base_frames, base_yaw_vals])
        return base_params

    def save_params_to_script(self, params, script_name):
        _ = self._load_script()
        q = _get_qlib(self.library_path, self.base_dir)
        self._setup_script(q, params)
        q.save_tas_script(script_name)

    def solve(self, init, base_params, use_base_params):
        cfg = self.cfg

        script_base_params = self._load_script()
        if base_params is None:
            base_params = script_base_params
        else:
            assert len(base_params) == cfg['num_frame_params'] + cfg['num_yaw_params']

        # Obtain base parameters --- parameters derived from the tas script
        base_frames = base_params[:cfg['num_frame_params']]
        base_yaw_vals = base_params[cfg['num_frame_params']:]
        logger.info('base time: %s', self._objective(base_params))

        # Setup pyade parameters.
        if init is None:
            init = np.concatenate([
                base_frames[None, :]
                    + np.random.normal(scale=cfg['init_frame_scale'],
                                       size=(cfg['pop_size'] * len(base_params),
                                             len(base_frames))),
                base_yaw_vals[None, :]
                    + np.random.normal(scale=cfg['init_yaw_scale'],
                                       size=(cfg['pop_size'] * len(base_params),
                                             len(base_yaw_vals))),
            ], axis=1)
        else:
            init = init.copy()
        if use_base_params:
            init[0] = base_params
        bounds = (
            [(x - cfg['frame_bounds'], x + cfg['frame_bounds']) for x in base_frames]
            + [(x - cfg['yaw_bounds'], x + cfg['yaw_bounds']) for x in base_yaw_vals]
        )
        pyade_param_updates = {
            'population_size': cfg['pop_size'] * len(base_params),
            'init': init,
            'bounds': np.array(bounds),
            'func': self._vector_objective,
            'max_evals': cfg['max_evals'],
            **cfg.get('pyade_params', {})
        }

        # Initialize wandb and save the config locally
        wandb.init(config=cfg)
        self._log_dir = pathlib.Path('runs') / str(wandb.run.id)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        with (self._log_dir / 'config.json').open('w') as f:
            json.dump(cfg, f)
        logger.info("Writing logs to %s", self._log_dir)

        # Do the optimization.
        try:
            self._all_time_min_energy = np.inf
            self._iter_num = 0
            if cfg['algorithm'] == 'jade':
                best_params = self._run_pyade(len(base_params), pyade.jade, pyade_param_updates, self._callback_jade)
            elif cfg['algorithm'] == 'ilshade':
                best_params = self._run_pyade(len(base_params), pyade.ilshade, pyade_param_updates,
                                              self._callback_ilshade)
            elif cfg['algorithm'] == 'jso':
                best_params = self._run_pyade(len(base_params), pyade.jso, pyade_param_updates, self._callback_jso)
            elif cfg['algorithm'] == 'pyswarms':
                options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
                optimizer = pyswarms.single.GlobalBestPSO(n_particles=init.shape[0], dimensions=init.shape[1],
                                                          options=options, init_pos=init)
                with multiprocessing.Pool(cfg['num_workers']) as pool:
                    cost, pos = optimizer.optimize(self._vector_objective, iters=1000, opts={'pool': pool})
            elif cfg['algorithm'] == 'nelder-mead':
                with multiprocessing.Pool(cfg['num_workers']) as pool:
                    vals = self._vector_objective(init, {'pool': pool})
                init = init[np.argsort(vals)]
                res = scipy.optimize.minimize(
                    self._objective,
                    np.zeros(len(base_params)),
                    method='Nelder-Mead',
                    callback=self._callback_nelder_mead,
                    options={
                        'initial_simplex': init[:len(base_params) + 1],
                        'maxiter': int(1e10),
                        'maxfev': int(1e10),
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True,
                    }
                )
                best_params = res.x
                print(res)
            elif cfg['algorithm'] == 'nevergrad':
                vals = np.array([self._objective(params) for params in init])
                logger.info('best initial population member: %s', np.min(vals))

                num_evals = 0
                best_value = np.inf
                best_params = None
                rolling_mean = 0.
                def print_candidate_and_value(optimizer, candidate, value):
                    nonlocal num_evals, best_value, best_params, rolling_mean

                    gamma = 1e-3
                    rolling_mean += (value - rolling_mean) * gamma

                    if value < best_value:
                        best_value = value
                        best_params = candidate * self._std + self._mean

                    if num_evals % 1_000 == 0:
                        log_dict = {'min_energy': best_value, 'evals': num_evals, 'rolling_mean': rolling_mean}
                        print(f'{best_params=}')
                        print(log_dict)
                        wandb.log(log_dict)

                    num_evals += 1

                optimizer = ng.optimizers.CMA(
                    parametrization=len(base_params),
                    budget=int(1e6),
                    num_workers=16,
                )
                optimizer.register_callback("tell", print_candidate_and_value)

                self._mean = np.mean(init, axis=0)
                self._std = np.std(init, axis=0)

                for params in init:
                    optimizer.suggest(((params - self._mean) / self._std))

                with concurrent.futures.ProcessPoolExecutor(cfg['num_workers']) as pool:
                    optimizer.minimize(self._ng_objective, executor=pool, verbosity=0)
            else:
                raise Exception(f"Unknown algorithm {cfg['algorithm']}")
        except KeyboardInterrupt:
            logger.info("Caught keyboard interrupt.  Cleaning up.")
            pass
        logger.info(f'{base_params=}')
        logger.info(f'{best_params=}')
        logger.info(f'{best_params - base_params=}')

        return best_params


def optimize_script(library_path, base_dir, init, base_params, use_base_params, cfg):
    # It seems pointless to create a class just to call a method and then dispose of it, however in this case the
    # callback / objective methods would have to be nested functions, which cannot be pickled.  Perhaps there is an
    # easier way?
    return TasOpt(library_path, base_dir, cfg).solve(init, base_params, use_base_params)


def _setup_logging():
    logging.basicConfig(level=logging.INFO)

def solve_entrypoint():
    _setup_logging()

    parser = argparse.ArgumentParser(description='Optimize TAS quake scripts')
    parser.add_argument('--library-path', '-l', type=str, help='Path to libtasquake.so.')
    parser.add_argument('--base-dir', '-b', type=str, help='Quake base dir.')
    parser.add_argument('--override-base-params', '-p', type=str, nargs='?',
                        help='Pickle file containing params used to generate initial population.')
    parser.add_argument('--initial-population', '-P', type=str, nargs='?',
                        help='Pickle file containing initial popualtion. Overrides --base-params.')
    parser.add_argument('--config', '-c', type=str, nargs='?',
                        default='config.yaml',
                        help='Config file')
    parser.add_argument('--use-base-params', '-u', action='store_true',
                        help='Include base params in initial population')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    init = None
    if args.initial_population is not None:
        with open(args.initial_population, 'rb') as f:
            _, init = pickle.load(f)
    base_params = None
    if args.override_base_params:
        with open(args.override_base_params, 'rb') as f:
            _, base_params = pickle.load(f)

    optimize_script(args.library_path, args.base_dir, init, base_params, args.use_base_params, cfg)


def save_script_entrypoint():
    _setup_logging()
    parser = argparse.ArgumentParser(description='Create script from all-time-best pickle')
    parser.add_argument('--library-path', '-l', type=str, help='Path to libtasquake.so.')
    parser.add_argument('--base-dir', '-b', type=str, help='Quake base dir.')
    parser.add_argument('--params', '-p', type=str,
                        help='Pickle file containing params to use to create script.')
    parser.add_argument('--output-name', '-o', type=str,
                        help='Name of the .qtas script to be created')
    parser.add_argument('--config', '-c', type=str, nargs='?',
                        default='config.yaml',
                        help='Config file')
    args = parser.parse_args()

    with open(args.config, 'rb') as f:
        cfg = yaml.safe_load(f)

    with open(args.params, 'rb') as f:
        print(args.params)
        params = pickle.load(f)

    return TasOpt(args.library_path, args.base_dir, cfg).save_params_to_script(
        params, args.output_name
    )
