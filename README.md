# Tasquakepy

This is the source code behind my improving Quake TAS video:

[![Beating Quake faster with evolutionary algorithms](https://img.youtube.com/vi/H8sDdEKizkk/0.jpg)](https://www.youtube.com/watch?v=H8sDdEKizkk)

Once setup, you should be able to reproduce the run from the video.

## Installation

1. Install [libtasquake](https://github.com/matthewearl/TASQuake/blob/me/lib/LIB.md).
The rest of this readme assumes libtasquake is installed in `~/TASQuake/`.

2. Setup a virtualenv, and then run to install tasquakepy (this repo):
```bash
git clone git@github.com:matthewearl/tasquakepy.git
pip install Cython
pip install -e tasquakepy
```

3. Install my fork of pyade:

```bash
git clone git@github.com:matthewearl/pyade.git
pip install -e pyade
```

This contains some fixes to a few of the DE algorithms, as well as an option for
providing a vectorized objective function.

4. Install [quake-light from SDA](https://speeddemosarchive.com/quake/downloads/quake-light.zip).
you may wish to install into a RAM disk (eg. `/dev/shm`) to avoid excessive disk
reads.  The only files you actually need are:

```
bcbf5ad90f8664dcef0fd7f168a2a613e4cbedf50c4fb2ff34612acd1145facd  /dev/shm/quake/joequake/pak0.pak
e1b21d8273721ab3fcbeadabbebb6dcd5c42e707205cf27d49f1668ef717249f  /dev/shm/quake/id1/pak0.pak
```

The above are sha256 sums, which you might also want to check.  The input script
you want to modify should also be in `/dev/quake/joequake/tas/`.


## Optimizing a run

This section should (with some luck) reproduce the parameters seen in the video.

1. Optionally [configure wandb](https://docs.wandb.ai/ref/cli/wandb-init).


2. Modify `configs/{feasible,finish}.yaml` with whatever settings you want.  To
   reproduce the parameters from the video leave it as is, but you might want to
   change the `num_workers` option to match your CPU.

3. Find a good initial population for the main solve:

```bash
opttasquake_solve -l ~/TASQuake/Source/release_lib/libtasquake.so -b /dev/shm/quake/ -c configs/feasible.yaml
```

Run this until `mean_std_params` plateaus and `mean_energy` is zero.
Populations and parameters will periodically be written into the `runs/`
directory.

4. Run the main optimization using the population found above:

```bash
opttasquake_solve -l ~/TASQuake/Source/release_lib/libtasquake.so -b /dev/shm/quake/ -c configs/finish.yaml -P [population file from the previous step]
```

This will take a while to complete.

## Saving the script

To convert the parameters found above into a script suitable for playing in
TASQuake, run:

```bash
opttasquake_save_script -l ~/TASQuake/Source/release_lib/libtasquake.so -b /dev/shm/quake/ -c configs/finish.yaml -o opt_notrick -p [param file from previous step]
```

## Applying the FPS trick

The exact finish time reported above is the interpolated time where the player
would first hit the exit trigger.  I added it to TASQuake to give a smoother
objective function.

As such, the exact finish time as reported above is *not* the usual finish time
used to rank Quake speedruns.  To get as good a real finish time as possible,
the FPS trick must be used.  This is a changing of framerates near the end of
the run to trigger the exit sooner.  With an optimal application of the FPS
trick it should be possible to get a real finish time around 0.1 seconds faster
than the exact finish time.

This repo includes a script to apply the FPS trick in an optimal way.  Before
running the untricked file (eg. `opt_notrick`) must be manually edited so that
the end looks like this:

```
+150:
        cl_maxfps 72
+1:
        cl_maxfps 10
+1:
        cl_maxfps 72
+63:
        echo e1m1 done
```

The important thing is that there are `cl_maxfps` commands on successive frames,
and that they take values 72, 10, and 72 respectively.  These values will be
modified by the script to optimize the finish time.

Once the script has been modified, run the following to do a brute force search
for the optimal FPS trick parameters:

```bash
opttasquake_fps_trick -l ~/TASQuake/Source/release_lib/libtasquake.so -b /dev/shm/quake/ -i opt_notrick -o opt_trick
```

Your final qtas script will be in `/dev/shm/quake/joequake/tas/opt_trick.qtas`.
To produce a demo from the script modify the `map` line in the file to be a
`record` line.
