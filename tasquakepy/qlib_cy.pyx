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


from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdlib cimport malloc, free


cdef extern from "<dlfcn.h>" nogil:
    ctypedef long int Lmid_t;

    void *dlopen(const char *, int)
    void *dlsym(void *, const char *)
    int dlclose(void *)
    char *dlerror()

    void *dlmopen (Lmid_t, const char *, int)

    enum:
        LM_ID_NEWLM
        RTLD_NOW
        RTLD_LAZY


ctypedef void (*start_host_fn_t)(unsigned int instance_id, int argc, char *argv[])
ctypedef void (*add_command_fn_t)(const char *command)
ctypedef void (*add_key_event_fn_t)(int key, int down)
ctypedef void (*add_mouse_motion_fn_t)(int dx, int dy)
ctypedef void (*set_angle_fn_t)(float yaw, float pitch)
ctypedef void (*do_frame_fn_t)()
ctypedef double (*get_time_fn_t)()
ctypedef int (*read_player_info_fn_t)(float pos[3], float vel[3], float view_angle[3],
                                      int *jump_released, int *on_ground, float cl_pos[3], float cl_vel[3])
ctypedef void (*check_intermission_fn_t)(int *intermission, double *completed_time, double *exact_completed_time);
ctypedef void (*set_usercmd_fn_t)(float view_angle[3], float fmove, float smove, float upmove);
ctypedef int (*get_num_blocks_fn_t)();
ctypedef void (*get_block_frames_fn_t)(int *frames);
ctypedef void (*set_block_frames_fn_t)(int *frames);
ctypedef int (*block_has_cvar_fn_t)(int block_num, char *name);
ctypedef float (*block_get_cvar_fn_t)(int block_num, char *name);
ctypedef void (*block_set_cvar_fn_t)(int block_num, char *name, float value);
ctypedef void (*get_playback_state_fn_t)(int *block_num, int *frame_num);

cdef _vec3_to_tuple(float v[3]):
    return tuple(v[i] for i in range(3))


cdef class QuakeCy:
    """A headless quake client.

    Example:

    >>> q = QuakeCy('libquakespasm.so')
    >>> q.start_host(['-basedir', os.expanduser('~/.quakespasm')])  # Launch headless quake
    >>> q.add_command('map e1m1')   # Push a command, to be executed on the next frame
    >>> while (info := q.read_player_info()) is None:   # Loop until the player object has been created
    ...     q.do_frame()
    >>> # Press the forward key for one second
    >>> q.add_key_event(ord('w'), True)
    >>> for i in range(72):
    ...     q.do_frame()
    >>> q.add_key_event(ord('w'), False)

    """

    cdef void *ref
    cdef start_host_fn_t start_host_fn

    cdef add_key_event_fn_t add_key_event_fn
    cdef add_mouse_motion_fn_t add_mouse_motion_fn
    cdef set_angle_fn_t set_angle_fn
    cdef do_frame_fn_t do_frame_fn
    cdef get_time_fn_t get_time_fn
    cdef read_player_info_fn_t read_player_info_fn
    cdef check_intermission_fn_t check_intermission_fn
    cdef set_usercmd_fn_t set_usercmd_fn
    cdef get_num_blocks_fn_t get_num_blocks_fn
    cdef get_block_frames_fn_t get_block_frames_fn
    cdef set_block_frames_fn_t set_block_frames_fn
    cdef block_has_cvar_fn_t block_has_cvar_fn
    cdef block_get_cvar_fn_t block_get_cvar_fn
    cdef block_set_cvar_fn_t block_set_cvar_fn
    cdef get_playback_state_fn_t get_playback_state_fn
    cdef add_command_fn_t add_command_fn

    cdef list args_bytes
    cdef char **argv

    def __cinit__(self, str library_path):
        cdef bytes library_path_bytes
        library_path_bytes = library_path.encode('ascii')
        self.ref = dlmopen(LM_ID_NEWLM, library_path_bytes, RTLD_LAZY);
        #self.ref = dlopen(library_path.encode('ascii'), RTLD_LAZY);

        if self.ref is NULL:
            raise Exception("dlmopen failed: " + str(dlerror()))

        self.start_host_fn = <start_host_fn_t> dlsym(self.ref, "start_host")

        self.add_command_fn = <add_command_fn_t> dlsym(self.ref, "add_command")
        self.add_key_event_fn = <add_key_event_fn_t> dlsym(self.ref, "add_key_event")
        self.add_mouse_motion_fn = <add_mouse_motion_fn_t> dlsym(self.ref, "add_mouse_motion")
        self.set_angle_fn = <set_angle_fn_t> dlsym(self.ref, "set_angle")
        self.do_frame_fn = <do_frame_fn_t> dlsym(self.ref, "do_frame")
        self.get_time_fn = <get_time_fn_t> dlsym(self.ref, "get_time")
        self.read_player_info_fn = <read_player_info_fn_t> dlsym(self.ref, "read_player_info")
        self.check_intermission_fn = <check_intermission_fn_t> dlsym(self.ref, "check_intermission")
        self.set_usercmd_fn = <set_usercmd_fn_t> dlsym(self.ref, "set_usercmd")
        self.get_num_blocks_fn = <get_num_blocks_fn_t> dlsym(self.ref, "get_num_blocks")
        self.get_block_frames_fn = <get_block_frames_fn_t> dlsym(self.ref, "get_block_frames")
        self.set_block_frames_fn = <set_block_frames_fn_t> dlsym(self.ref, "set_block_frames")
        self.block_has_cvar_fn = <block_has_cvar_fn_t> dlsym(self.ref, "block_has_cvar")
        self.block_get_cvar_fn = <block_get_cvar_fn_t> dlsym(self.ref, "block_get_cvar")
        self.block_set_cvar_fn = <block_set_cvar_fn_t> dlsym(self.ref, "block_set_cvar")
        self.get_playback_state_fn = <get_playback_state_fn_t> dlsym(self.ref, "get_playback_state")

        self.argv = NULL

    def __dealloc__(self):
        if self.ref is not NULL:
            dlclose(self.ref)
            self.ref = NULL
            PyMem_Free(self.argv)

    def start_host(self, unsigned int instance_id, list args):
        args = ['quake-binary-placeholder'] + args
        cdef int argc = len(args)
        assert self.argv == NULL
        self.argv = <char **> PyMem_Malloc(argc * sizeof(char*));

        self.args_bytes = [a.encode('ascii') for a in args]

        for i in range(argc):
            self.argv[i] = self.args_bytes[i]

        self.start_host_fn(instance_id, argc, self.argv)

    def add_command(self, str command):
        command = command + "\n"
        command_bytes = command.encode('ascii')
        self.add_command_fn(command_bytes)

    def add_key_event(self, key, down):
        self.add_key_event_fn(key, down)

    def set_angle(self, yaw, pitch):
        self.set_angle_fn(yaw, pitch)

    def do_frame(self):
        self.do_frame_fn()

    def get_time(self):
        return self.get_time_fn()

    def read_player_info(self):
        cdef float pos[3], vel[3], view_angle[3], cl_pos[3], cl_vel[3]
        cdef int rc, jump_released, on_ground

        rc = self.read_player_info_fn(pos, vel, view_angle, &jump_released, &on_ground, cl_pos, cl_vel)
        if rc == 0:
            return None
        else:
            return {'pos': _vec3_to_tuple(pos),
                    'vel': _vec3_to_tuple(vel),
                    'view_angle': _vec3_to_tuple(view_angle),
                    'jump_released': bool(jump_released),
                    'on_ground': bool(on_ground),
                    'client_pos': _vec3_to_tuple(cl_pos),
                    'client_vel': _vec3_to_tuple(cl_vel)}

    def check_intermission(self):
        cdef int intermission
        cdef double completed_time
        cdef double exact_completed_time

        self.check_intermission_fn(&intermission, &completed_time, &exact_completed_time)

        return intermission, completed_time, exact_completed_time

    def set_usercmd(self, yaw, pitch, roll, fmove, smove, upmove):
        cdef float angles[3];

        angles[0] = pitch;
        angles[1] = yaw;
        angles[2] = roll;

        self.set_usercmd_fn(angles, fmove, smove, upmove)

    def get_num_blocks(self):
        return self.get_num_blocks_fn()

    def get_block_frames(self, int[::1] frames):
        return self.get_block_frames_fn(&frames[0])

    def set_block_frames(self, int[::1] frames):
        return self.set_block_frames_fn(&frames[0])

    def block_has_cvar(self, int block_num, str name):
        return <bint>self.block_has_cvar_fn(block_num, name.encode('utf-8'))

    def block_get_cvar(self, int block_num, str name):
        return self.block_get_cvar_fn(block_num, name.encode('utf-8'))

    def block_set_cvar(self, int block_num, str name, float value):
        return self.block_set_cvar_fn(block_num, name.encode('utf-8'), value)

    def get_playback_state(self):
        cdef:
            int block_num
            int frame_num
        self.get_playback_state_fn(&block_num, &frame_num)
        return (block_num, frame_num)

    def get_block_command(self, int block_num, int command_num):
        return self.get_block_command_fn(block_num, command_num)
