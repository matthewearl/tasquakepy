#!/usr/bin/env python

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


from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
    name="tasquakepy",
    version="0.1",
    entry_points={
        "console_scripts": [
            "opttasquake_solve = tasquakepy.opt:solve_entrypoint",
            "opttasquake_save_script = tasquakepy.opt:save_script_entrypoint",
            "opttasquake_fps_trick = tasquakepy.fpstrick:apply_fps_trick_entry_point",
            "opttasquake_benchmark = tasquakepy.benchmark:benchmark_entrypoint",
        ]
    },
    description="Python wrapper around libtasquake",
    install_requires=["numpy", "matplotlib", "scipy", "wandb", "PyYAML", "tqdm"],
    author="Matt Earl",
    packages=["tasquakepy"],
    ext_modules=cythonize([Extension("tasquakepy.qlib_cy", ["tasquakepy/qlib_cy.pyx"])]),
)
