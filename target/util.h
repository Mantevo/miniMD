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

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#if PRECISION == 1
typedef uint32_t CAS_t;
#else
typedef uint64_t CAS_t;
#endif

#ifdef USE_OFFLOAD
#define atomic_add(fptr, x) \
{ \
  volatile CAS_t *ptr = (CAS_t*) fptr; \
  CAS_t expected, desired; \
  do \
  { \
    expected = *ptr; \
    MMD_float expected_float = *fptr; \
    MMD_float desired_float = expected_float + x; \
    desired = *((CAS_t*) &desired_float); \
  } while(not __sync_bool_compare_and_swap(ptr, expected, desired)); \
}
#else
#define atomic_add(fptr, x) \
{ \
  _Pragma("omp atomic") \
  *fptr += x; \
}
#endif

inline uint32_t log2_ceil(uint32_t N)
{
  if(N == 0)
  {
    return 0;
  }
  const uint32_t allbut1 = __builtin_clzl(1);
  const uint32_t lz      = __builtin_clzl(N);
  const uint32_t log2    = allbut1 - lz;
  uint32_t       cand    = 1 << log2;
  if(N > cand)
  {
    return log2 + 1;
  }
  return log2;
}

template <typename T>
void blelloch_excl_scan_target_inplace(T *data, int N)
{
  const int levels = log2_ceil(N);
#ifndef USE_OFFLOAD
  #pragma omp parallel
#endif
  {
#ifdef USE_OFFLOAD
    #pragma omp target teams distribute parallel for
#else
    #pragma omp for
#endif
    for(int i = 0; i < N - 1; i += 2)
    {
      data[i + 1] = data[i] + data[i + 1];
    }
    for(int l = 1; l < levels; ++l)
    {
      const int stride = 1 << (l + 1);
#ifdef USE_OFFLOAD
      #pragma omp target teams distribute parallel for
#else
      #pragma omp for
#endif
      for(int i = stride - 1; i < N; i += stride)
      {
        data[i] = data[i - stride / 2] + data[i];
      }
    }
    for(int l = levels - 2; l >= 0; --l)
    {
      const int stride = 1 << (l + 1);
#ifdef USE_OFFLOAD
      #pragma omp target teams distribute parallel for
#else
      #pragma omp for
#endif
      for(int i = stride + stride / 2 - 1; i < N; i += stride)
      {
        data[i] = data[i - stride / 2] + data[i];
      }
    }
  }
}

static void die(const char *format, ...)
{
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
  exit(EXIT_FAILURE);
}

#endif
