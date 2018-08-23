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
#ifndef OFFLOAD_H
#define OFFLOAD_H

#include <cassert>
#include <cstring>
#include <omp.h>

#ifdef USE_OFFLOAD
#define MAX_TEAM_SIZE 128
#endif

static void *mmd_alloc(size_t bytes)
{
#ifdef ALIGNMALLOC
  char *h_ptr = ( char * )_mm_malloc(bytes + ALIGNMALLOC, ALIGNMALLOC);
#ifdef USE_OFFLOAD
  char *d_ptr = ( char * )omp_target_alloc(bytes + ALIGNMALLOC, 0);
#endif
#else
  char *h_ptr = ( char * )malloc(bytes);
#ifdef USE_OFFLOAD
  char *d_ptr = ( char * )omp_target_alloc(bytes, 0);
#endif
#endif

#ifdef USE_OFFLOAD
  omp_target_associate_ptr(h_ptr, d_ptr, bytes, 0, 0);
#endif

  return ( void * )h_ptr;
}

static void mmd_free(void *ptr)
{
  if(ptr)
  {
#ifdef USE_OFFLOAD
    omp_target_disassociate_ptr(ptr, 0);
#endif

#ifdef ALIGNMALLOC
    _mm_free(ptr);
#else
    free(ptr);
#endif
  }
}

static void *mmd_replace_alloc(void *ptr, size_t bytes)
{
  mmd_free(ptr);
  return mmd_alloc(bytes);
}

static void *mmd_grow_alloc(void *ptr, size_t old_bytes, size_t new_bytes)
{
  char *old_ptr = ( char * )ptr;
  char *new_ptr = ( char * )mmd_alloc(new_bytes);
  if(old_ptr)
  {
    assert(old_bytes > 0);
    std::memcpy(new_ptr, old_ptr, old_bytes);
    mmd_free(old_ptr);
  }
  return new_ptr;
}

#endif
