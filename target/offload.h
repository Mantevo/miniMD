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
#include <cstdio>
#include <cstring>
#include <omp.h>
#include "util.h"

inline const char *get_variant_string()
{
#ifdef USE_OFFLOAD
#ifdef OFFLOAD_X86
  return "OFFLOAD_X86";
#else // OFLOAD_X86
#ifdef OFFLOAD_NVPTX
  return "OFFLOAD_NVPTX";
#else // OFFLOAD_NVPTX
#error "Unknown variant"
  return "";
#endif // OFFLOAD_NVPTX
#endif // OFFLOAD_X86
#else  // USE_OFFLOAD
  return "HOST";
#endif // USE_OFFLOAD
}

inline int get_user_max_team_size()
{
#ifdef USE_OFFLOAD
  return MAX_TEAM_SIZE;
#else
  return 0;
#endif
}

inline int heuristic_nteam(int nlocal)
{
#ifdef USE_OFFLOAD
  return 1 << log2_ceil(nlocal / get_user_max_team_size());
#else
  return 0;
#endif
}

inline int conforming_team_size(int size)
{
  int observed_size;
  #pragma omp target teams map(tofrom:observed_size) thread_limit(size)
  {
    if(omp_get_team_num() == 0)
    {
      observed_size = omp_get_max_threads();
    }
  }
  return observed_size;
}

inline int team_num_test(int nteams, int thread_lim)
{
  int observed_num;
  #pragma omp target teams distribute parallel for num_teams(nteams) map(tofrom:observed_num) thread_limit(thread_lim)
  for(int i = 0; i < nteams * thread_lim; ++i)
  {
    if(i == 0)
    {
      observed_num = omp_get_num_teams();
    }
  }
  return observed_num;
}

inline bool check_offload_device(int nteams, int thread_lim)
{
  int is_initial = 1;
  #pragma omp target teams distribute parallel for num_teams(nteams) thread_limit(thread_lim) map(tofrom:is_initial)
  for(int i = 0; i < nteams * thread_lim; ++i)
  {
    if(i == 0)
    {
      is_initial = omp_is_initial_device();
    }
  }
  return is_initial == 0;
}


inline bool check_offload()
{
#ifdef USE_OFFLOAD
  const int  team_num_check = 2048;
  const bool is_offloading  = check_offload_device(team_num_check, MAX_TEAM_SIZE);
  if(is_offloading)
  {
    printf("Offload appears to be running somewhere other than the host (that's good.)\n");
  }
  else
  {
    fprintf(stderr, "'Offload' appears to be running on the host (that's bad.)\n");
  }

  const int  teamsize    = conforming_team_size(MAX_TEAM_SIZE);
  const bool teamsize_ok = teamsize <= MAX_TEAM_SIZE;
  if(teamsize_ok)
  {
    printf("Team size limit %d respected by offload (got %d.)\n", MAX_TEAM_SIZE, teamsize);
  }
  else
  {
    fprintf(stderr, "Team size limit %d NOT respected by offload! (got %d.)\n", MAX_TEAM_SIZE, teamsize);
  }
  // this is to get the runtime to say that it is running on the device more than anything
  // we don't expect it to be honored, so we won't bail if it isn't.
  const int team_num = team_num_test(team_num_check, MAX_TEAM_SIZE);
  if(team_num == team_num_check)
  {
    printf("Team num %d respected by offload.\n", team_num_check);
  }
  else
  {
    printf("Team num %d NOT respected by offload! (got %d.)\n", team_num_check, team_num);
  }
  return teamsize_ok;
#else
  return false;
#endif
}

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
