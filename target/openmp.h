/* ----------------------------------------------------------------------
   miniMD is a simple, parallel molecular dynamics (MD) code.   miniMD is
   an MD microapplication in the Mantevo project at Sandia National
   Laboratories ( http://www.mantevo.org ). The primary
   authors of miniMD are Steve Plimpton (sjplimp@sandia.gov), Paul Crozier
   (pscrozi@sandia.gov) and Christian Trott (crtrott@sandia.gov).

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

   For questions, contact Paul S. Crozier (pscrozi@sandia.gov) or
   Christian Trott (crtrott@sandia.gov).

   Please read the accompanying README and LICENSE files.
   ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
OpenMP wrapper for headerfile
-defines fake OpenMP functions for compilation without OpenMP support
-"#pragma omp" statements will be ignored by compiler automatically but
 could be ifdef'd in order to get rid of Compiler warnings
---------------------------------------------------------------------- */

#ifdef _OPENMP
#include <omp.h>
#else
inline int  omp_get_thread_num() { return 0; }
inline int  omp_get_max_threads() { return 1; }
inline void omp_set_num_threads(int num_threads) { return 1; }
inline int  __sync_fetch_and_add(int *ptr, int value)
{
  int tmp = *ptr;
  ptr[0] += value;
  return tmp;
}
#endif
