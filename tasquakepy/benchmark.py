import argparse
import itertools
import logging
import multiprocessing
import threading
import time

from . import qlib


logger = logging.getLogger(__name__)
THREAD_LOCAL_DATA = threading.local()


def _get_qlib(library_path, base_dir):
    if not hasattr(THREAD_LOCAL_DATA, 'quake'):
        logger.info(f"Loading {library_path} {THREAD_LOCAL_DATA} {threading.get_native_id()}")
        THREAD_LOCAL_DATA.quake = qlib.Quake(threading.get_native_id(), library_path, base_dir)
    return THREAD_LOCAL_DATA.quake


def run_tas_script(args):
    q = _get_qlib(args.library_path, args.base_dir)

    # Only load the script once --- reloading erases save states.
    if getattr(THREAD_LOCAL_DATA, 'loaded_script', None) != args.script:
        q.load_tas_script(args.script)
        THREAD_LOCAL_DATA.loaded_script = args.script

    if args.frame is not None:
        q.play_tas_script(frame_num=args.frame, save_state=True)
    else:
        q.play_tas_script()

    num_frames_local = 0
    while q.exact_completed_time is None:
        _ = q.step_no_cmd()
        num_frames_local += 1

    q.stop_tas_script()

    with num_games.get_lock():
        num_games.value += 1
        num_games_val = num_games.value

    with num_frames.get_lock():
        num_frames.value += num_frames_local
        num_frames_val = num_frames.value

    if num_games_val % 100 == 0:
        time_elapsed = time.perf_counter() - start_time
        gps = num_games_val / time_elapsed
        fps = num_frames_val / time_elapsed
        fpg = num_frames_val / num_games_val
        print(f'#games: {num_games_val}, games per second: {gps:.3f}, '
              f'frames per game: {fpg:.3f} frames per second: {fps:.3f} '
              f'speedup: {fps / 72:.2f}x')


def benchmark_entrypoint():
    global start_time, num_games, num_frames

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Benchmark libtasquake')
    parser.add_argument('--library-path', '-l', type=str, help='Path to libtasquake.so.')
    parser.add_argument('--base-dir', '-b', type=str, help='Quake base dir.')
    parser.add_argument('--script', '-s', type=str, help='TAS script.')
    parser.add_argument('--num-games', '-n', type=int, default=10_000,
                        help='Number of games to simulate.')
    parser.add_argument('--frame', '-f', type=int, help='Skip to the given frame.')
    parser.add_argument('--jobs', '-j', type=int, help='Process pool size')
    args = parser.parse_args()

    num_games = multiprocessing.Value('i')
    num_frames = multiprocessing.Value('i')
    start_time = time.perf_counter()
    with multiprocessing.Pool(args.jobs) as pool:
        pool.map(run_tas_script, itertools.repeat(args, args.num_games))
