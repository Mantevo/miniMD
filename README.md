   miniMD is a simple, parallel molecular dynamics (MD) code.   miniMD is
   an MD microapplication in the Mantevo project at Sandia National 
   Laboratories ( http://www.mantevo.org ). The primary 
   authors of miniMD are Steve Plimpton, Paul Crozier (pscrozi@sandia.gov)
   and Christian Trott (crtrott@sandia.gov).

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This library is free software; you 
   can redistribute it and/or modify it under the terms of the GNU Lesser 
   General Public License as published by the Free Software Foundation; 
   either version 3 of the License, or (at your option) any later 
   version.
  
   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
   Lesser General Public License for more details.
    
   You should have received a copy of the GNU Lesser General Public 
   License along with this software; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
   USA.  See also: http://www.gnu.org/licenses/lgpl.txt .

   For questions, contact Christian Trott (crtrott@sandia.gov). 

   Please read the accompanying README and LICENSE files.



------------------------------------------------
Description:
------------------------------------------------
miniMD is a parallel molecular dynamics (MD) simulation package written
in C++ and intended for use on parallel supercomputers and new 
architechtures for testing purposes. The software package is meant to  
be simple, lightweight, and easily adapted to new hardware. It is 
designed following many of the same algorithm concepts as our LAMMPS 
(http://lammps.sandia.gov) parallel MD code, but is much simpler.

Authors: Steve Plimpton, Paul Crozier (pscrozi@sandia.gov)
   and Christian Trott (crtrott@sandia.gov)        

This simple code is a self-contained piece of C++ software 
that performs parallel molecular dynamics simulation of a Lennard-Jones
or a EAM system and gives timing information.

It is implemented to be very scalable (in a weak sense).  Any 
reasonable parallel computer should be able to achieve excellent 
scaled speedup (weak scaling).  miniMD uses a spatial decomposition
parallelism and has many other similarities to the much more 
complicated LAMMPS MD code: http://lammps.sandia.gov

The sub-directories contain different variants of miniMD:

miniMD_ref:          supports MPI+OpenMP hybrid mode.
miniMD_OpenCL:       an OpenCL version of miniMD, uses MPI to parallelize over 
                     multiple devices. Limited Features. There is an issue with 
                     running larger system then -s 39. Other problems exist with
                     double precision support.
miniMD_Kokkos:       supports MPI and uses Kokkos on top of it, compiles
                     with pThreads, OpenMP or CUDA backend
miniMD_KokkosLambda: another Kokkos variant making extensive use of C++11
miniMD_Intel:        supports MPI+OpenMP hybrid mode. Optimized by Intel.
                     Comes with an intrinsic version of the LJ-force kernel 
                     and the neighborlist construction for Xeon Phi.
miniMD_OpenACC:      supports MPI+OpenACC hybrid mode. 


Each variant is self contained and does not reference any source files of 
the other variants.

 
------------------------------------------------
Strengths and Weaknesses:
------------------------------------------------

miniMD consists of less than 5,000 lines of C++ code. Like LAMMPS, miniMD uses
spatial decomposition MD, where individual processors in a cluster own subsets
of the simulation box. And like LAMMPS, miniMD enables users to specify a problem
size, atom density, temperature, timestep size, number of timesteps to perform,
and particle interaction cutoff distance. But compared to LAMMPS, MiniMD's feature 
set is extremely limited, and only two types interactions (Lennard-Jones/ EAM) are
available. No long-range electrostatics or molecular force field features are 
available. Inclusion of such features is unnecessary for testing basic MD and 
would have made miniMD much bigger, more complicated, and harder to port to novel
hardware. The current version of LAMMPS includes over 200,000 lines of code in 
hundreds of files, nineteen optional packages, over one hundred different commands, 
and over five hundred pages of documentation. Such a large and complicated code 
is not ideally suited for answering certain performance questions or for tinkering 
by non-MD-experts. The biggest difference to LAMMPS in terms of performance is
caused by using only a single atom-type. Thus all force parameter lookups are 
simple variable references, while in LAMMPS they are gather operations. On 
architectures with slow vector-gather operations, this can cause signifcant 
performance differences between miniMD and LAMMPS.  

MiniMD uses neighborlists for the force calculation, as opposed to cell lists 
which are employed by for example COMD. The neighborlist approach (or variants 
of it) are used by most commonly used MD applications, such as LAMMPS, Amber 
and NAMD. Cell lists are employed by some specialised codes, in particular for 
very large scale simulations which might be memory capacity limited. 
With neighborlists the memory footprint of a simulation is significantly larger,
though with about 500,000 atoms per GB it is still small compared to many other 
applications. On the other hand the number of distance checks in the neighborlist 
approach is much smaller than with cell lists. For neighborlists the distances 
to all atoms in a volume of 4/3*PI*r_cut^3, r_cut being the neighbor cutoff 
distance, have to be checked. With celllists that volume is 27*r_cut^3. While 
the latter approach makes the data access for positions coalesced reads, as 
opposed to random reads with neighborlists, on most architectures this is not 
enough of an advantage to compensate for the ~6x difference in distance checks.

In versions >=2.x miniMD now simulates the behaviour of having multiple atom
types. Mostly this means that certain variable accesses such as force parameters
and cutoffs are now replaced by table lookups. This will reduce performance
compared to 1.x variants of miniMD. It will also hinder vectorization more. 
On the upside this change closes the biggest gap between the miniApp and
what happens in real apps. In fact the performance differenc with LAMMPS
was significantly reduced.

------------------------------------------------
Compiling the code:
------------------------------------------------

There is a simple Makefile that should be easily modified for most 
Unix-like environments.  There are also one or more Makefiles with 
extensions that indicate the target machine and compilers. Read the 
Makefile for further instructions.  If you generate a Makefile for 
your platform and care to share it, please send it to Paul Crozier:
pscrozi@sandia.gov . By default the code compiles with MPI support 
and can be run on one or more processors. There is also a 
Makefile.default which should NOT require a GNU Make compatible 
make. 

==Compiling:

  make
  
  Get info on all options, and targets
  
  make -f Makefile.default
  
  Build with simplified Makefile, using defaults for a CPU system
  
  make <platform>

  Note, when building the KokkosArray variant
  directly out of the svn repository you need to do 

  make <platform> SVN=yes   

  for building miniMD_KokkosArray. 

  Furthermore miniMD_ref and miniMD_KokkosArray support both single
  and double precision builds. Single precision can be triggered by using
  SP=yes/no in the make command-line (e.g. make openmpi SP=yes).
  
  Other options are:
  DEBUG=yes -- enable debugmode
  AVX=yes -- enable compilation for avx [DEFAULT] 
  KNC=yes -- enable compilation for Xeon Phi
  SIMD=yes -- use #pragma simd for some kernels [DEFAULT]
  PAD=[3/4] -- pad arrays to 3 or 4 elements
  RED_PREC=yes -- enable fast_math and similar (reduced precision divide)
  GSUNROLL=yes -- unroll gather and scatter (for Xeon Phi only) [DEFAULT]
  SP=yes -- use single precision  
  LIBRT=yes -- use librt timers (more precise)
  
  For KokkosArray Variant only:
  KOKKOSPATH=path -- path to the Kokkos core source directory (kokkos/core/src)
  OMP=yes -- use OpenMP (if not use PThread) [DEFAULT]
  HWLOC=yes -- use HWLOC for thread pinning
  HWLOCPATH=path -- path to HWLOC library when building with HWLOC support   
  CUDA=yes -- build with cuda support (works only with the cuda target)
  CUDAARCH=sm_xx -- set GPU architecture target (default sm_35)

  Typical choices:
  
  CPUS:
  make openmpi -j 16
  
  Xeon Phi
  make intel KNC=yes -j 16
  
  Build with pthreads [KokkosArray Variant only:
  make openmpi OMP=no HWLOC=yes KOKKOSPATH=/usr/local/kokkos/core/src -j 16
  
==To remove all output files, type:

  make clean_<platform>

  or 

  make clean

==Testing:

  make test

  The test will run a simulation and compare it against reference output. 
  Where are different test modes, which change the amount of tests run.
  Running 'make test' will give instructions how to run more complex tests.
  Note the test does not currently run with multiple GPUs since it does not 
  provide the necessary environment variables.

------------------------------------------------
Running the code and sample I/O:
------------------------------------------------

Usage:

miniMD (serial mode)

mpirun -np numproc miniMD (MPI mode)

Example:

mpirun -np 16 ./miniMD 

MiniMD understands a number of command-line options. To get the options 
for each particular variant of miniMD please use "-h" as an argument.

You will also need to provide a simple input script, which you can model
after the ones included in this directory (e.g. in.lj.miniMD). The format and
parameter description is as follows:

Sample input file contents found in "lj.in":
------------------------------------------------

Lennard-Jones input file for MD benchmark

lj             units (lj or metal)
none           data file (none or filename)       
lj             force style (lj or eam)
1.0 1.0        force parameters for LJ (epsilon, sigma)
32 32 32       size of problem
100            timesteps
0.005          timestep size 
1.44           initial temperature 
0.8442         density 
20             reneighboring every this many steps
2.5 0.30       force cutoff and neighbor skin 
100            thermo calculation every this many steps (0 = start,end)


------------------------------------------------

Sample output file contents found in "out.lj.miniMD":
------------------------------------------------

# Create System:
# Done .... 
# miniMD-Reference 1.2 (MPI+OpenMP) output ...
# Run Settings: 
	# MPI processes: 2
	# OpenMP threads: 16
	# Inputfile: in.lj.miniMD
	# Datafile: None
# Physics Settings: 
	# ForceStyle: LJ
	# Force Parameters: 1.00 1.00
	# Units: LJ
	# Atoms: 864000
	# System size: 100.78 100.78 100.78 (unit cells: 60 60 60)
	# Density: 0.844200
	# Force cutoff: 2.500000
	# Timestep size: 0.005000
# Technical Settings: 
	# Neigh cutoff: 2.800000
	# Half neighborlists: 0
	# Neighbor bins: 50 50 50
	# Neighbor frequency: 20
	# Sorting frequency: 20
	# Thermo frequency: 100
	# Ghost Newton: 1
	# Use intrinsics: 0
	# Do safe exchange: 0
	# Size of float: 8

# Starting dynamics ...
# Timestep T U P Time
0 1.440000e+00 -6.773368e+00 -5.019671e+00  0.000
100 7.310629e-01 -5.712170e+00 1.204577e+00  3.650


# Performance Summary:
# MPI_proc OMP_threads nsteps natoms t_total t_force t_neigh t_comm t_other performance perf/thread grep_string t_extra
2 16 100 864000 3.649762 2.584821 0.735003 0.145945 0.183993 23672777.021430 739774.281920 PERF_SUMMARY 0.035863

------------------------------------------------
Running on GPUs with Kokkos
------------------------------------------------

The Kokkos variant needs a CUDA aware MPI for running on GPUs (though it might work on a single GPU with any MPI).
Currently known MPI implementations with CUDA support are:
mvapich2 1.8 or higher
openmpi 1.7 or higher
cray mpi on XK7 and higher

Note those typically require some environment variables to be set. For example mvapich2 1.9 can be used like this:
mpiexec -np 2 -env MV2_USE_CUDA=1 ./miniMD_mvapichcuda --half_neigh 0 -s 60

When compiling for GPU Architectures prior to Kepler (sm_21 or lower) you need to put -DUSE_TEXTURE_REFERENCES in 
the compiler flags to use Texture Memory during the force calculations. If not you loose about 70% of your performance.

------------------------------------------------

Known Issues:
------------------------------------------------


The OpenCL variant does not currently support all features of the Reference and Kokkos variant. In particular
it does not support EAM simulations. Also due to limitations in OpenCL (and the author not having the time to work
around them) the simulations are limited to about 240k atoms in the standard LJ settings. This corresponds to -s 39.

Running the in.*-data.miniMD inputs on the GPU with the Kokkos variant defaults to too many neighbor bins. This
causes significantly increased memory consumption and longer runtimes. Use -b 30 as a command line option, to override
the default neighbor bin size.

The option --safe_exchange is currently not active in publicly available builds. 
