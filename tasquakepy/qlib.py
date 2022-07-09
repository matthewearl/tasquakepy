# Copyright (c) 2022 Matthew Earl
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.


import collections.abc
import dataclasses
import enum
import os
from typing import Dict

import numpy as np

from .qlib_cy import QuakeCy


class Key(enum.IntEnum):
    """Enumeration of input keys."""
    STRAFE_LEFT = 0
    STRAFE_RIGHT = enum.auto()
    FORWARD = enum.auto()
    JUMP = enum.auto()


_KEY_MAP = {
    Key.FORWARD: 'w',
    Key.STRAFE_LEFT: 'a',
    Key.STRAFE_RIGHT: 'd',
    Key.JUMP: ' ',
}


@dataclasses.dataclass
class Block:
    block_num: int
    q: QuakeCy

    def __setitem__(self, key, value):
        return self.q.block_set_cvar(self.block_num, key, value)

    def __getitem__(self, key):
        return self.q.block_get_cvar(self.block_num, key)

    def __contains__(self, key):
        return self.q.block_has_cvar(self.block_num, key)


@dataclasses.dataclass
class BlockSeq(collections.abc.Sequence):
    q: QuakeCy

    def __getitem__(self, block_num):
        if block_num >= len(self):
            raise IndexError
        return Block(block_num, self.q)

    def __len__(self):
        return self.q.get_num_blocks()


class Quake:
    def __init__(self, library_path, base_dir=None):
        self._quake_cy = QuakeCy(library_path)

        if base_dir is None:
            base_dir = os.path.expanduser('~/.quakespasm')
        self._quake_cy.start_host(['-noudp', '-protocol', '15', '-basedir', base_dir])
        self._key_state = None
        self._current_map_name = None
        self.completed_time = None

    def setup_cvars(self, side_speed):
        self._quake_cy.add_command(f"cl_sidespeed {side_speed}")
        self._quake_cy.add_command("cl_forwardspeed 800")
        self._quake_cy.add_command("cl_rollangle 0")
        self._quake_cy.add_command("cl_bob 0")
        self._quake_cy.add_command("skill 0")
        self._quake_cy.add_command("host_jq 1")
        self._quake_cy.do_frame()

    def load_map(self, map_name):
        if map_name != self._current_map_name:
            self._quake_cy.add_command(f"map {map_name}")
            num_to_skip = 5
        else:
            self._send_key_events(())  # Release any keys that are currently being held
            self._quake_cy.add_command("restart")
            num_to_skip = 7
        self._current_map_name = map_name
        self._key_state = {k: False for k in Key}

        self.completed_time = None
        self.exact_completed_time = None

        while (info := self._quake_cy.read_player_info()) is None:
            self._quake_cy.do_frame()

        # It takes a few more frames before the player actually gains control, and is in the right position.
        for i in range(num_to_skip):
            self._quake_cy.do_frame()

        return self._quake_cy.read_player_info()

    def _send_key_events(self, keys):
        for k in Key:
            is_down = self._key_state[k]
            want_down = (k in keys)
            if is_down != want_down:
                self._quake_cy.add_key_event(ord(_KEY_MAP[k]), want_down)
                self._key_state[k] = want_down

    def step_no_cmd(self):
        self._quake_cy.do_frame()

        intermission, completed_time, exact_completed_time = self._quake_cy.check_intermission()
        if intermission != 0:
            self.completed_time = completed_time
            self.exact_completed_time = exact_completed_time

        return self._quake_cy.read_player_info()

    def step(self, keys, yaw, pitch):
        self._send_key_events(keys)
        self._quake_cy.set_angle(yaw, pitch)

        return self.step_no_cmd()

    def step_usercmd(self, yaw, pitch, roll, fmove, smove, jump):
        self._send_key_events((Key.JUMP,) if jump else ())
        self._quake_cy.set_usercmd(yaw, pitch, roll, fmove, smove, 0)

        return self.step_no_cmd()

    def load_tas_script(self, script_name):
        self._quake_cy.add_command(f'tas_script_load {script_name}')
        self._quake_cy.do_frame()

    def save_tas_script(self, script_name):
        self._quake_cy.add_command(f'tas_edit_save {script_name}')
        self._quake_cy.do_frame()

    @property
    def blocks(self):
        return BlockSeq(self._quake_cy)

    def get_cvar_vals(self, cvar: str):
        block_nums = []
        vals = []
        for bl in self.blocks:
            if cvar in bl:
                block_nums.append(bl.block_num)
                vals.append(bl[cvar])

        return np.array(block_nums), np.array(vals)

    def set_cvar_vals(self, cvar: str, block_nums: np.ndarray, vals: np.ndarray):
        assert block_nums.shape == vals.shape
        for block_num, val in zip(block_nums, vals):
            self.blocks[block_num][cvar] = val

    def get_block_frames(self) -> np.ndarray:
        num_blocks = self._quake_cy.get_num_blocks()
        frames = np.empty(num_blocks, dtype=np.int32)
        self._quake_cy.get_block_frames(frames)
        return frames

    def set_block_frames(self, frames: np.ndarray):
        self._quake_cy.set_block_frames(frames)

    def play_tas_script(self):
        self._quake_cy.add_command('tas_script_play')
        for _ in range(3):
            self._quake_cy.do_frame()

        self._key_state = {k: False for k in Key}
        self.completed_time = None
        self.exact_completed_time = None

    def stop_tas_script(self):
        self._quake_cy.add_command('tas_script_stop')
        self._quake_cy.do_frame()

    def run_tas_script(self, max_frames: int):
        self._quake_cy.add_command('tas_script_play')
        for _ in range(3):
            self._quake_cy.do_frame()

        for frame_num in range(max_frames):
            self._quake_cy.do_frame()
            intermission, completed_time, exact_completed_time = self._quake_cy.check_intermission()
            if completed_time:
                break
        return frame_num, intermission, completed_time, exact_completed_time

