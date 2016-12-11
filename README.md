# Legion-SNAP

Implementation of the SNAP Mini-App in Legion

Some important points regarding this implementation:

* Note the minimalism of Legion. There are only 75 runtime 
  calls to handle all the important control logic of SNAP, 
  inlcluding partitioning the data, creating all the fields,
  and predicating the various tasks across iterations. Much of the 
  verbosity of this implementation of SNAP is in the actual 
  kernels themselves (more on this below), and is not a problem 
  that Legion is designed to solve. In the future we plan to 
  replace all the different task variants with task generators 
  that can JIT specialize task implementations on the fly.
* The only way to really test the overhead of a runtime is with 
  optimized tasks. It's very easy to hide runtime overhead if 
  the tasks are unoptimized. Therefore all the kernels 
  in this implementation are tuned well beyond what a normal user
  would likely ever be capable of doing to ensure that we can 
  accurately gauge the runtime overhead. These kernels are highly
  optimized to block for caches, use vector intrinstics, and use
  inline assembly code. This simply ensures we make an accurate
  assessment of the Legion runtime's overhead and should not be
  used in determining the verbosity of the Legion interface.
* There is one optimization that Legion supports for execution
  with GPUs that is not available in other programming models.
  Legion's ability to find significant parallelism from sweeps
  from different corners and energy groups allows us to launch
  kernels with significantly less parallelism, but also much
  better locality. All GPU sweeps in this version of SNAP
  launch a single CTA per energy group and corner. We can do
  this because we run ahead and get enough of these "kernels"
  in flight to fill the GPU's SMs. Try doing that in another
  programming system. :)

