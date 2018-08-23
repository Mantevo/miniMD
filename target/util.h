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

#ifndef UTIL_H
#define UTIL_H

#include <cstdint>

template <int N>
struct CASmap;

template <>
struct CASmap<4>
{
  typedef uint32_t CAS_type;
};

template <>
struct CASmap<8>
{
  typedef uint64_t CAS_type;
};

template <typename T>
void atomicAdd_CAS(volatile T *fptr, T x)
{
  typedef typename CASmap<sizeof(T)>::CAS_type CAS_t;
  volatile CAS_t *                             ptr = ( CAS_t * )fptr;
  CAS_t                                        expected, desired;
  do
  {
    expected         = *ptr;
    T expected_float = *fptr;
    T desired_float  = expected_float + x;
    desired          = *(( CAS_t * )&desired_float);
  } while(not __sync_bool_compare_and_swap(ptr, expected, desired));
};

#ifdef USE_OFFLOAD
template <typename T>
void atomic_add(volatile T *val, T update)
{
  atomicAdd_CAS(val, update);
}
#else
template <typename T>
void atomic_add(volatile T *val, T update)
{
  #pragma omp atomic
  *val += update;
}
#endif

#endif
