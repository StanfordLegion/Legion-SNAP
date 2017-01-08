# Legion-SNAP

This is an implementation of the [SNAP](https://github.com/losalamos/SNAP)
mini-application in Legion. This work was done by NVIDIA as part of the
Fast Forward 2 project and was funded by the U.S. Department of Energy
under subcontract B609478 with Lawrence Livermore National Security, LLC.
The code is released under the Apache License, Version 2.0. A copy of the
license can be found in this repository.

Several notes on the code for this implementation of SNAP. 

* This version of SNAP only implements the mini-KBA sweep algorithm
  for performing the computation. It also only supports 3D computations.
  The Legion version issues index space launches for each stage of a
  sweep for each energy group and direction. This allows Legion to 
  extract task parallelism from the different sweeps. This proves
  especially useful for way Legion performs the GPU computation. 
  This version launches a single CTA per sweep and energy group and
  relies on task parallelism to launch multiple kernels onto the GPU
  to keep all the SMs on the GPU busy. This is unorthodox, but allows
  for a more effecient implementation that can store per-angle fluxes
  in the register file as the CTA sweeps through cells.

* The Legion style of this implementation is designed to illustrate
  how code should be generated from a higher-level compiler, with
  good application-specific abstractions and multiple different task
  variants for each logical task. This allows the application to
  specialize itself for different target architectures. The downside
  is that the code can appear to be verbose. This is not a function
  of Legion, but of what needs to be done to make a code portable
  across many different architectures. In general you will notice
  that there are very few places where Legion shows up in this code.
  There are fewer than 100 Legion runtime calls which represents 
  less than 2% of all the code in the application. All of the task
  variants are highly tuned so it is possible to accurately gauge
  the runtime overhead that is incurred.

* This version of SNAP is the first real Legion application that
  relies heavily upon using predication to handle the dynamic
  convergence tests needed in SNAP. This implementation shows 
  how to chain together predicates to perform the convergence
  tests. It also demonstrates how to use Legion futures and 
  tasks to construct a monad for performing accurate timing of
  tasks in a deferred execution environment (similar to how
  monads in Haskell are needed to handle the laziness of the
  execution model). Users should see this as a template for
  handling other kinds of predicated computations that needed 
  to be done in a deferred execution model. 

* Included in this version of SNAP is a custom mapper that 
  demonstrates how a mapper can be specialized to a particular
  application by overriding specific calls from the default
  mapper implementation. An interesting observation is that
  the implementation of the mapper calls for a custom mapper
  are considerably simpler than the default mapper 
  implementations. The reason for this is that with an 
  application specific mapper the implementation can be 
  tailored directly to the application and we know exactly
  what layouts and locations to use for physical instances.

