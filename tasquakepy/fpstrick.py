import argparse
import logging

import numpy as np
import tqdm

from . import qlib


logger = logging.getLogger(__name__)


MAXFPS_CVAR = 'cl_maxfps'


INVALID_SCRIPT_MSG = """
Invalid input qtas file: {}

The script must have 3 sequential blocks that contain a "cl_maxfps" command,
with values 72, 10, 72 respectively.  The blocks should be one frame apart, and
the sequence should occur near the finish time.
"""


class _InvalidScript(Exception):
    pass


def _apply_trick(q, offset, fps):
    block_frames = q.get_block_frames()

    # validate the script is as expected
    block_num = None
    for bl in reversed(q.blocks):
        if MAXFPS_CVAR in bl:
            break
    else:
        raise _InvalidScript("trick not found")
    if bl.block_num < 2:
        raise _InvalidScript("not enough blocks")
    block_nums = np.arange(3) + bl.block_num - 2
    for block_num in block_nums:
        if MAXFPS_CVAR not in q.blocks[block_num]:
            raise _InvalidScript("cl_maxfps cvar not present on preceding "
                                "blocks")
    if ([q.blocks[block_num][MAXFPS_CVAR] for block_num in block_nums]
            != [72, 10, 72]):
        raise _InvalidScript("unexpected cl_maxfps values")

    if not np.all(np.diff(block_frames[block_nums]) == 1):
        raise _InvalidScript("blocks not on sequential frames")

    # Modify the blocks
    q.blocks[block_nums[0]][MAXFPS_CVAR] = fps
    block_frames[block_nums] += offset
    q.set_block_frames(block_frames)


def save_trick(q, tas_script_in, tas_script_out, offset, fps):
    q.load_tas_script(tas_script_in)
    _apply_trick(q, offset, fps)
    q.save_tas_script(tas_script_out)


def _count_frames(q, tas_script):
    """Run the script once to count number of frames"""

    logger.info("Counting frames in script")
    q.load_tas_script(tas_script)
    q.play_tas_script()

    num_frames = 0
    while True:
        _ = q.step_no_cmd()
        if q.exact_completed_time is not None:
            break
        num_frames += 1

    q._quake_cy.add_command('tas_script_stop')
    q._quake_cy.do_frame()

    logger.info("%s frames in input script", num_frames)
    logger.info("input exact completed time: %s", q.exact_completed_time)
    logger.info("input completed time: %s", q.completed_time)
    return num_frames


def _run_game(q, tas_script, offset, fps, num_frames):
    q.load_tas_script(tas_script)

    _apply_trick(q, offset, fps)

    # Play the modified script and return the completed time
    q.play_tas_script()

    for frame_num in range(num_frames):
        _ = q.step_no_cmd()
        if q.exact_completed_time is not None:
            break

    q.stop_tas_script()

    assert (q.exact_completed_time is not None) == (q.completed_time is not None)

    intermission = q.exact_completed_time is not None
    return intermission, q.completed_time, q.exact_completed_time


def apply_fps_trick_entry_point():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Create script from all-time-best pickle')
    parser.add_argument('--library-path', '-l', type=str, help='Path to libtasquake.so.')
    parser.add_argument('--base-dir', '-b', type=str, help='Quake base dir.')
    parser.add_argument('--input-name', '-i', type=str, help='Input script.')
    parser.add_argument('--output-name', '-o', type=str, help='Output script.')
    args = parser.parse_args()

    q = qlib.Quake(args.library_path, args.base_dir)

    offsets = np.arange(-32, 32)
    fpss = np.arange(10, 73)
    search_values = [(offset, fps) for offset in offsets for fps in fpss]

    num_frames = _count_frames(q, args.input_name) + 100
    try:
        results = [
            _run_game(q, args.input_name, offset, fps, num_frames)
            for offset, fps in tqdm.tqdm(search_values)
        ]
    except _InvalidScript as e:
        print(INVALID_SCRIPT_MSG.format(e))
        raise SystemExit(1)

    intermissions = np.array([result[0] for result in results])
    completed_times = np.array([result[1] for result in results])
    exact_completed_times = np.array([result[2] for result in results])

    completed_times[intermissions == 0] = np.inf

    min_time = np.min(completed_times)
    min_idxs = np.where(completed_times == min_time)[0]

    saved = False
    for min_idx in min_idxs:
        completed_time = completed_times[min_idx]
        exact_completed_time = exact_completed_times[min_idx]
        min_offset, min_fps = search_values[min_idx]

        logger.info("output exact completed time: %s", exact_completed_time)
        logger.info("output completed time: %s", completed_time)
        logger.info("min_offset: %s min_fps: %s", min_offset, min_fps)
        if not saved:
            save_trick(q, args.input_name, args.output_name, min_offset,
                       min_fps)
            logger.info('saved')
            saved = True
