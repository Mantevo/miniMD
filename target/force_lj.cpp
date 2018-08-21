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

#include "force.h"
#include "miniMD_math.h"
#include "offload.h"
#include "openmp.h"
#include "stdio.h"
#include "util.h"

#ifndef VECTORLENGTH
#define VECTORLENGTH 4
#endif

Force::Force(int ntypes_)
{
  cutforce       = 0.0;
  use_oldcompute = 0;

  reneigh = 1;
  style   = FORCELJ;
  ntypes  = ntypes_;

  cutforcesq = ( MMD_float * )mmd_alloc(sizeof(MMD_float) * ntypes * ntypes);
  epsilon    = ( MMD_float * )mmd_alloc(sizeof(MMD_float) * ntypes * ntypes);
  sigma6     = ( MMD_float * )mmd_alloc(sizeof(MMD_float) * ntypes * ntypes);
  sigma      = ( MMD_float * )mmd_alloc(sizeof(MMD_float) * ntypes * ntypes);

  for(int i = 0; i < ntypes * ntypes; i++)
  {
    cutforcesq[i] = 0.0;
    epsilon[i]    = 1.0;
    sigma6[i]     = 1.0;
    sigma[i]      = 1.0;
  }
}

Force::~Force()
{
  mmd_free(cutforcesq);
  mmd_free(epsilon);
  mmd_free(sigma6);
  mmd_free(sigma);
}

void Force::setup()
{
  for(int i = 0; i < ntypes * ntypes; i++)
  {
    cutforcesq[i] = cutforce * cutforce;
  }
#ifdef USE_OFFLOAD
  #pragma omp target update to(cutforcesq[0:ntypes * ntypes])
  #pragma omp target update to(epsilon[0:ntypes * ntypes])
  #pragma omp target update to(sigma6[0:ntypes * ntypes])
  #pragma omp target update to(sigma[0:ntypes * ntypes])
#endif
}

void Force::compute(Atom &atom, Neighbor &neighbor, Comm &comm, int me)
{
  eng_vdwl = 0;
  virial   = 0;

  if(evflag)
  {
    if(use_oldcompute)
    {
      return compute_original<1>(atom, neighbor, me);
    }

    if(neighbor.halfneigh)
    {
      if(neighbor.ghost_newton)
      {
        if(threads->omp_num_threads > 1)
        {
          if(atom.privatize)
          {
            return compute_halfneigh_threaded_private<1, 1>(atom, neighbor, me);
          }
          else
          {
            return compute_halfneigh_threaded<1, 1>(atom, neighbor, me);
          }
        }
        else
        {
          return compute_halfneigh<1, 1>(atom, neighbor, me);
        }
      }
      else
      {
        if(threads->omp_num_threads > 1)
        {
          if(atom.privatize)
          {
            return compute_halfneigh_threaded_private<1, 0>(atom, neighbor, me);
          }
          else
          {
            return compute_halfneigh_threaded<1, 0>(atom, neighbor, me);
          }
        }
        else
        {
          return compute_halfneigh<1, 0>(atom, neighbor, me);
        }
      }
    }
    else
    {
      return compute_fullneigh<1>(atom, neighbor, me);
    }
  }
  else
  {
    if(use_oldcompute)
    {
      return compute_original<0>(atom, neighbor, me);
    }


    if(neighbor.halfneigh)
    {
      if(neighbor.ghost_newton)
      {
        if(threads->omp_num_threads > 1)
        {
          if(atom.privatize)
          {
            return compute_halfneigh_threaded_private<0, 1>(atom, neighbor, me);
          }
          else
          {
            return compute_halfneigh_threaded<0, 1>(atom, neighbor, me);
          }
        }
        else
        {
          return compute_halfneigh<0, 1>(atom, neighbor, me);
        }
      }
      else
      {
        if(threads->omp_num_threads > 1)
        {
          if(atom.privatize)
          {
            return compute_halfneigh_threaded_private<0, 0>(atom, neighbor, me);
          }
          else
          {
            return compute_halfneigh_threaded<0, 0>(atom, neighbor, me);
          }
        }
        else
        {
          return compute_halfneigh<0, 0>(atom, neighbor, me);
        }
      }
    }
    else
    {
      return compute_fullneigh<0>(atom, neighbor, me);
    }
  }
}

// original version of force compute in miniMD
//  -MPI only
//  -not vectorizable
template <int EVFLAG>
void Force::compute_original(Atom &atom, Neighbor &neighbor, int me)
{
  int        nlocal = atom.nlocal;
  int        nall   = atom.nlocal + atom.nghost;
  MMD_float *x      = atom.x;
  MMD_float *f      = atom.f;
  int *      type   = atom.type;

  eng_vdwl = 0;
  virial   = 0;
  // clear force on own and ghost atoms

  for(int i = 0; i < nall; i++)
  {
    f[i * PAD + 0] = MMD_float(0.0);
    f[i * PAD + 1] = MMD_float(0.0);
    f[i * PAD + 2] = MMD_float(0.0);
  }

  // loop over all neighbors of my atoms
  // store force on both atoms i and j

  for(int i = 0; i < nlocal; i++)
  {
    const int *const neighs   = &neighbor.neighbors[i * neighbor.maxneighs];
    const int        numneigh = neighbor.numneigh[i];
    const MMD_float  xtmp     = x[i * PAD + 0];
    const MMD_float  ytmp     = x[i * PAD + 1];
    const MMD_float  ztmp     = x[i * PAD + 2];
    const int        type_i   = type[i];

    for(int k = 0; k < numneigh; k++)
    {
      const int       j      = neighs[k];
      const MMD_float delx   = xtmp - x[j * PAD + 0];
      const MMD_float dely   = ytmp - x[j * PAD + 1];
      const MMD_float delz   = ztmp - x[j * PAD + 2];
      int             type_j = type[j];
      const MMD_float rsq    = delx * delx + dely * dely + delz * delz;

      const int type_ij = type_i * ntypes + type_j;

      if(rsq < cutforcesq[type_ij])
      {
        const MMD_float sr2   = MMD_float(1.0) / rsq;
        const MMD_float sr6   = sr2 * sr2 * sr2 * sigma6[type_ij];
        const MMD_float force = MMD_float(48.0) * sr6 * (sr6 - MMD_float(0.5)) * sr2 * epsilon[type_ij];
        f[i * PAD + 0] += delx * force;
        f[i * PAD + 1] += dely * force;
        f[i * PAD + 2] += delz * force;
        f[j * PAD + 0] -= delx * force;
        f[j * PAD + 1] -= dely * force;
        f[j * PAD + 2] -= delz * force;

        if(EVFLAG)
        {
          eng_vdwl += (MMD_float(4.0) * sr6 * (sr6 - MMD_float(1.0))) * epsilon[type_ij];
          virial += (delx * delx + dely * dely + delz * delz) * force;
        }
      }
    }
  }
}


// optimised version of compute
//  -MPI only
//  -use temporary variable for summing up fi
//  -enables vectorization by:
//     -getting rid of 2d pointers
//     -use pragma omp simd to force vectorization of inner loop
template <int EVFLAG, int GHOST_NEWTON>
void Force::compute_halfneigh(Atom &atom, Neighbor &neighbor, int me)
{
  const int              nlocal = atom.nlocal;
  const int              nall   = atom.nlocal + atom.nghost;
  const MMD_float *const x      = atom.x;
  MMD_float *const       f      = atom.f;
  const int *const       type   = atom.type;

  // clear force on own and ghost atoms
  for(int i = 0; i < nall; i++)
  {
    f[i * PAD + 0] = MMD_float(0.0);
    f[i * PAD + 1] = MMD_float(0.0);
    f[i * PAD + 2] = MMD_float(0.0);
  }

  // loop over all neighbors of my atoms
  // store force on both atoms i and j
  MMD_float t_energy = 0;
  MMD_float t_virial = 0;

  for(int i = 0; i < nlocal; i++)
  {
    const int *const neighs    = &neighbor.neighbors[i * neighbor.maxneighs];
    const int        numneighs = neighbor.numneigh[i];
    const MMD_float  xtmp      = x[i * PAD + 0];
    const MMD_float  ytmp      = x[i * PAD + 1];
    const MMD_float  ztmp      = x[i * PAD + 2];
    const int        type_i    = type[i];

    MMD_float fix = MMD_float(0.0);
    MMD_float fiy = MMD_float(0.0);
    MMD_float fiz = MMD_float(0.0);

#ifdef USE_SIMD
#pragma vector unaligned
#pragma omp simd reduction(+ : fix, fiy, fiz)
#endif
    for(int k = 0; k < numneighs; k++)
    {
      const int       j       = neighs[k];
      const MMD_float delx    = xtmp - x[j * PAD + 0];
      const MMD_float dely    = ytmp - x[j * PAD + 1];
      const MMD_float delz    = ztmp - x[j * PAD + 2];
      const int       type_j  = type[j];
      const MMD_float rsq     = delx * delx + dely * dely + delz * delz;
      const int       type_ij = type_i * ntypes + type_j;

      if(rsq < cutforcesq[type_ij])
      {
        const MMD_float sr2   = MMD_float(1.0) / rsq;
        const MMD_float sr6   = sr2 * sr2 * sr2 * sigma6[type_ij];
        const MMD_float force = MMD_float(48.0) * sr6 * (sr6 - MMD_float(0.5)) * sr2 * epsilon[type_ij];

        fix += delx * force;
        fiy += dely * force;
        fiz += delz * force;

        if(GHOST_NEWTON || j < nlocal)
        {
          f[j * PAD + 0] -= delx * force;
          f[j * PAD + 1] -= dely * force;
          f[j * PAD + 2] -= delz * force;
        }

        if(EVFLAG)
        {
          const MMD_float scale = (GHOST_NEWTON || j < nlocal) ? MMD_float(1.0) : MMD_float(0.5);
          t_energy += scale * (MMD_float(4.0) * sr6 * (sr6 - MMD_float(1.0))) * epsilon[type_ij];
          t_virial += scale * (delx * delx + dely * dely + delz * delz) * force;
        }
      }
    }

    f[i * PAD + 0] += fix;
    f[i * PAD + 1] += fiy;
    f[i * PAD + 2] += fiz;
  }

  eng_vdwl += t_energy;
  virial += t_virial;
}

// optimised version of compute
//  -MPI + OpenMP (atomics for fj update)
//  -use temporary variable for summing up fi
//  -enables vectorization by:
//    -getting rid of 2d pointers
//    -use pragma omp simd to force vectorization of inner loop (not currently supported due to OpenMP atomics
template <int EVFLAG, int GHOST_NEWTON>
void Force::compute_halfneigh_threaded(Atom &atom, Neighbor &neighbor, int me)
{
  MMD_float t_eng_vdwl = 0;
  MMD_float t_virial   = 0;

  const int              nlocal = atom.nlocal;
  const int              nall   = atom.nlocal + atom.nghost;
  const MMD_float *const x      = atom.x;
  MMD_float *const       f      = atom.f;
  const int *const       type   = atom.type;

  const int  maxneighs = neighbor.maxneighs;
  const int *neighbors = neighbor.neighbors;
  const int *numneigh  = neighbor.numneigh;
#ifdef USE_OFFLOAD
  const MMD_float *cutforcesq = this->cutforcesq;
  const MMD_float *epsilon    = this->epsilon;
  const MMD_float *sigma6     = this->sigma6;
  const int        ntypes     = this->ntypes;
  #pragma omp target enter data map(alloc:x[0:nall * PAD])
  #pragma omp target enter data map(alloc:type[0:nall])
  #pragma omp target enter data map(alloc:f[0:nall * PAD])
  #pragma omp target update to(x[:nall * PAD])
  #pragma omp target update to(type[:nall])  // TODO: only copy type after sorting also: when copy x?
#endif

// clear force on own and ghost atoms
#ifdef USE_OFFLOAD
  #pragma omp target teams distribute parallel for
#else
  #pragma omp parallel for
#endif
  for(int i = 0; i < nall; i++)
  {
    f[i * PAD + 0] = MMD_float(0.0);
    f[i * PAD + 1] = MMD_float(0.0);
    f[i * PAD + 2] = MMD_float(0.0);
  }

  // loop over all neighbors of my atoms
  // store force on both atoms i and j
#ifdef USE_OFFLOAD
  #pragma omp target teams distribute map(tofrom:t_eng_vdwl,t_virial)
#else
  #pragma omp parallel for reduction(+ : t_eng_vdwl, t_virial)
#endif
  for(int i = 0; i < nlocal; i++)
  {
    const int *const neighs    = neighbors + i * maxneighs;
    const int        numneighs = numneigh[i];
    const MMD_float  xtmp      = x[i * PAD + 0];
    const MMD_float  ytmp      = x[i * PAD + 1];
    const MMD_float  ztmp      = x[i * PAD + 2];
    const int        type_i    = type[i];
    MMD_float        fix       = MMD_float(0.0);
    MMD_float        fiy       = MMD_float(0.0);
    MMD_float        fiz       = MMD_float(0.0);

    MMD_float w_virial   = MMD_float(0.0);
    MMD_float w_eng_vdwl = MMD_float(0.0);

// clang-format off
#ifdef USE_SIMD
#ifndef __clang__ /* clang errors on atomic inside simd */
    #pragma vector unaligned
    #pragma omp simd reduction(+:fix, fiy, fiz, t_eng_vdwl, t_virial)
#endif
#endif
#ifdef USE_OFFLOAD
     #pragma omp parallel for
#endif // clang-format on
    for(int k = 0; k < numneighs; k++)
    {
      const int       j       = neighs[k];
      const MMD_float delx    = xtmp - x[j * PAD + 0];
      const MMD_float dely    = ytmp - x[j * PAD + 1];
      const MMD_float delz    = ztmp - x[j * PAD + 2];
      const int       type_j  = type[j];
      const MMD_float rsq     = delx * delx + dely * dely + delz * delz;
      const int       type_ij = type_i * ntypes + type_j;

      if(rsq < cutforcesq[type_ij])
      {
        const MMD_float sr2   = MMD_float(1.0) / rsq;
        const MMD_float sr6   = sr2 * sr2 * sr2 * sigma6[type_ij];
        const MMD_float force = MMD_float(48.0) * sr6 * (sr6 - MMD_float(0.5)) * sr2 * epsilon[type_ij];

        atomic_add(&fix, delx * force);
        atomic_add(&fiy, dely * force);
        atomic_add(&fiz, delz * force);

        if(GHOST_NEWTON || j < nlocal)
        {
          atomic_add(&f[j * PAD + 0], -delx * force);
          atomic_add(&f[j * PAD + 1], -dely * force);
          atomic_add(&f[j * PAD + 2], -delz * force);
        }

        if(EVFLAG)
        {
          const MMD_float scale      = (GHOST_NEWTON || j < nlocal) ? MMD_float(1.0) : MMD_float(0.5);
          const MMD_float l_eng_vdwl = scale * (MMD_float(4.0) * sr6 * (sr6 - MMD_float(1.0))) * epsilon[type_ij];
          atomic_add(&w_eng_vdwl, l_eng_vdwl);
          const MMD_float l_virial = scale * (delx * delx + dely * dely + delz * delz) * force;
          atomic_add(&w_virial, l_virial);
        }
      }
    }

    if(EVFLAG)
    {
      atomic_add(&t_eng_vdwl, w_eng_vdwl);
      atomic_add(&t_virial, w_virial);
    }

    atomic_add(&f[i * PAD + 0], fix);
    atomic_add(&f[i * PAD + 1], fiy);
    atomic_add(&f[i * PAD + 2], fiz);
  }

#ifdef USE_OFFLOAD
  #pragma omp target update from(f[0:nall * PAD])
  #pragma omp target exit data map(delete:x[0:nall * PAD])
  #pragma omp target exit data map(delete:type[0:nall])
  #pragma omp target exit data map(delete:f[0:nall * PAD])
#endif

  eng_vdwl += t_eng_vdwl;
  virial += t_virial;
}

//#define USE_SCATTER_VARIANT
// clang-format off
__attribute__((noinline)) // clang-format on
void private_force_update(MMD_float* f, const int32_t j, const MMD_float x, const MMD_float y, const MMD_float z)
{
  f[j * PAD + 0] -= x;
  f[j * PAD + 1] -= y;
  f[j * PAD + 2] -= z;
}

#ifdef __INTEL_COMPILER
#if(PAD == 4)
#include <immintrin.h>
// clang-format off
__declspec(vector_variant(implements(private_force_update(MMD_float* f, const int32_t j, const MMD_float x, const MMD_float y, const MMD_float z)),
                          uniform(f),
                          vectorlength(16),
                          mask,
                          processor(knl))) // clang-format on
    void _mm512_private_force_update_ps(MMD_float *f, const __m512i j, const __m512 x, const __m512 y, const __m512 z, __mmask16 mask)
{
  // respect the mask
  __m512 mx = _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), x);
  __m512 my = _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), y);
  __m512 mz = _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), z);

  // transpose 3x16 => 16x4 (w/ undefined padding)
  __m512 out[4];
#if 0
    __m512 xy1256 = _mm512_shuffle_ps(mx, my,                    _MM_SHUFFLE(1, 0, 1, 0));
    __m512 xy3478 = _mm512_shuffle_ps(mx, my,                    _MM_SHUFFLE(3, 2, 3, 2));
    __m512 z01256 = _mm512_shuffle_ps(mz, _mm512_undefined_ps(), _MM_SHUFFLE(1, 0, 1, 0));
    __m512 z03478 = _mm512_shuffle_ps(mz, _mm512_undefined_ps(), _MM_SHUFFLE(3, 2, 3, 2));
    out[0] = _mm512_shuffle_ps(xy1256, z01256, _MM_SHUFFLE(2, 0, 2, 0));
    out[1] = _mm512_shuffle_ps(xy1256, z01256, _MM_SHUFFLE(3, 1, 3, 1));
    out[2] = _mm512_shuffle_ps(xy3478, z03478, _MM_SHUFFLE(2, 0, 2, 0));
    out[3] = _mm512_shuffle_ps(xy3478, z03478, _MM_SHUFFLE(3, 1, 3, 1));
#else
  __m512 xy1256 = _mm512_unpacklo_ps(mx, my);
  __m512 xy3478 = _mm512_unpackhi_ps(mx, my);
  __m512 z01256 = _mm512_unpacklo_ps(mz, _mm512_undefined_ps());
  __m512 z03478 = _mm512_unpackhi_ps(mz, _mm512_undefined_ps());
  out[0]        = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(xy1256), _mm512_castps_pd(z01256)));
  out[1]        = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(xy1256), _mm512_castps_pd(z01256)));
  out[2]        = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(xy3478), _mm512_castps_pd(z03478)));
  out[3]        = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(xy3478), _mm512_castps_pd(z03478)));
#endif

  // gather/add/scatter the atoms in groups of 4
  __declspec(aligned(64)) int32_t js[16];
  _mm512_store_epi32(js, _mm512_mullo_epi32(j, _mm512_set1_epi32(4)));
  for(int g = 0; g < 4; ++g)
  {
    __m512 fj;
    fj = _mm512_castps128_ps512(_mm_load_ps(f + js[g + 0]));
    fj = _mm512_mask_broadcast_f32x4(fj, 0x00F0, _mm_load_ps(f + js[g + 4]));
    fj = _mm512_mask_broadcast_f32x4(fj, 0x0F00, _mm_load_ps(f + js[g + 8]));
    fj = _mm512_mask_broadcast_f32x4(fj, 0xF000, _mm_load_ps(f + js[g + 12]));
    fj = _mm512_sub_ps(fj, out[g]);
    _mm512_mask_compressstoreu_ps(f + js[g + 0], 0x000F, fj);
    _mm512_mask_compressstoreu_ps(f + js[g + 4], 0x00F0, fj);
    _mm512_mask_compressstoreu_ps(f + js[g + 8], 0x0F00, fj);
    _mm512_mask_compressstoreu_ps(f + js[g + 12], 0xF000, fj);
  }
}
#endif
#endif

// optimised version of compute
//  -MPI + OpenMP (privatization for fj update)
//  -use temporary variable for summing up fi
//  -enables vectorization by:
//    -getting rid of 2d pointers
//    -use pragma omp simd to force vectorization of inner loop
//    -use private force arrays for each thread
template <int EVFLAG, int GHOST_NEWTON>
void Force::compute_halfneigh_threaded_private(Atom &atom, Neighbor &neighbor, int me)
{
  MMD_float t_eng_vdwl = 0;
  MMD_float t_virial   = 0;

  const int nthreads = threads->omp_num_threads;

  const int              nlocal = atom.nlocal;
  const int              nall   = atom.nlocal + atom.nghost;
  const MMD_float *const x      = atom.x;
  const int *const       type   = atom.type;

  #pragma omp parallel for reduction(+ : t_eng_vdwl, t_virial)
  for(int i = 0; i < nlocal; i++)
  {
    MMD_float *const f         = &atom.f_private[omp_get_thread_num() * nall * PAD]; // TODO: Hoist this
    const int *const neighs    = &neighbor.neighbors[i * neighbor.maxneighs];
    const int        numneighs = neighbor.numneigh[i];
    const MMD_float  xtmp      = x[i * PAD + 0];
    const MMD_float  ytmp      = x[i * PAD + 1];
    const MMD_float  ztmp      = x[i * PAD + 2];
    const int        type_i    = type[i];
    MMD_float        fix       = MMD_float(0.0);
    MMD_float        fiy       = MMD_float(0.0);
    MMD_float        fiz       = MMD_float(0.0);

#ifdef USE_SIMD
#pragma vector unaligned
#pragma omp simd reduction(+ : fix, fiy, fiz, t_eng_vdwl, t_virial)
#endif
    for(int k = 0; k < numneighs; k++)
    {
      const int       j       = neighs[k];
      const MMD_float delx    = xtmp - x[j * PAD + 0];
      const MMD_float dely    = ytmp - x[j * PAD + 1];
      const MMD_float delz    = ztmp - x[j * PAD + 2];
      const int       type_j  = type[j];
      const MMD_float rsq     = delx * delx + dely * dely + delz * delz;
      const int       type_ij = type_i * ntypes + type_j;

      if(rsq < cutforcesq[type_ij])
      {
        const MMD_float sr2   = MMD_float(1.0) / rsq;
        const MMD_float sr6   = sr2 * sr2 * sr2 * sigma6[type_ij];
        const MMD_float force = MMD_float(48.0) * sr6 * (sr6 - MMD_float(0.5)) * sr2 * epsilon[type_ij];

        fix += delx * force;
        fiy += dely * force;
        fiz += delz * force;

#ifndef USE_SCATTER_VARIANT
        if(GHOST_NEWTON || j < nlocal)
        {
          f[j * PAD + 0] -= delx * force;
          f[j * PAD + 1] -= dely * force;
          f[j * PAD + 2] -= delz * force;
        }
#else
        if(GHOST_NEWTON || j < nlocal)
        {
          private_force_update(f, j, delx * force, dely * force, delz * force);
        }
#endif

        if(EVFLAG)
        {
          const MMD_float scale = (GHOST_NEWTON || j < nlocal) ? MMD_float(1.0) : MMD_float(0.5);
          t_eng_vdwl += scale * (MMD_float(4.0) * sr6 * (sr6 - MMD_float(1.0))) * epsilon[type_ij];
          t_virial += scale * (delx * delx + dely * dely + delz * delz) * force;
        }
      }
    }

    f[i * PAD + 0] += fix;
    f[i * PAD + 1] += fiy;
    f[i * PAD + 2] += fiz;
  }

  // reduce private copies and clear them for the next timestep
  // likely sub-optimal: makes no assumptions about which threads touch which atoms
  // iterates through array(s) in cache-line-sized chunks

  MMD_float *reduction_out = ( MMD_float * )atom.f;
  #pragma omp parallel for
  for(int chunk_offset = 0; chunk_offset < nall * PAD; chunk_offset += CACHELINE_SIZE / sizeof(MMD_float))
  {
    // don't worry about the last chunk; arrays are padded at the end
    #pragma vector aligned
    #pragma omp simd
    for(int c = 0; c < CACHELINE_SIZE / sizeof(MMD_float); ++c)
    {
      MMD_float *reduction_in = ( MMD_float * )&atom.f_private[chunk_offset + c];
      MMD_float  tmp          = MMD_float(0.0);
      #pragma unroll(2)
      for(int t = 0; t < nthreads; ++t)
      {
        tmp += reduction_in[t * nall * PAD];
        reduction_in[t * nall * PAD] = MMD_float(0.0);
      }
      reduction_out[chunk_offset + c] = tmp;
    }
  }

  eng_vdwl += t_eng_vdwl;
  virial += t_virial;
}

// optimised version of compute
//  -MPI + OpenMP (using full neighborlists)
//  -gets rid of fj update (read/write to memory)
//  -use temporary variable for summing up fi
//  -enables vectorization by:
//    -get rid of 2d pointers
//    -use pragma omp simd to force vectorization of inner loop
template <int EVFLAG>
void Force::compute_fullneigh(Atom &atom, Neighbor &neighbor, int me)
{
  MMD_float t_eng_vdwl = 0;
  MMD_float t_virial   = 0;

  const int              nlocal = atom.nlocal;
  const int              nall   = atom.nlocal + atom.nghost;
  const MMD_float *const x      = atom.x;
  MMD_float *const       f      = atom.f;
  const int *const       type   = atom.type;

  const int  maxneighs = neighbor.maxneighs;
  const int *neighbors = neighbor.neighbors;
  const int *numneigh  = neighbor.numneigh;
#ifdef USE_OFFLOAD
  const MMD_float *cutforcesq = this->cutforcesq;
  const MMD_float *epsilon    = this->epsilon;
  const MMD_float *sigma6     = this->sigma6;
  const int        ntypes     = this->ntypes;
  #pragma omp target enter data map(alloc:f[0:nall * PAD])
  #pragma omp target enter data map(alloc:x[0:nall * PAD])
  #pragma omp target enter data map(alloc:type[0:nall])
  #pragma omp target update to(x[0:nall * PAD])
  #pragma omp target update to(type[0:nall])  // TODO: only copy type after sorting also: when copy x?
#endif

// clear force on own and ghost atoms
#ifdef USE_OFFLOAD
  #pragma omp target teams distribute parallel for
#else
  #pragma omp parallel for
#endif
  for(int i = 0; i < nlocal; i++)
  {
    f[i * PAD + 0] = MMD_float(0.0);
    f[i * PAD + 1] = MMD_float(0.0);
    f[i * PAD + 2] = MMD_float(0.0);
  }

  // loop over all neighbors of my atoms
  // store force on atom i
#ifdef USE_OFFLOAD
  #pragma omp target teams map(tofrom:t_eng_vdwl,t_virial) thread_limit(MAX_TEAM_SIZE)
#endif
  {
#ifdef USE_OFFLOAD
    const int nthr = omp_get_max_threads();
    MMD_float w_eng_vdwl[MAX_TEAM_SIZE];
    MMD_float w_virial[MAX_TEAM_SIZE];
    for(int i = 0; i < nthr; i++)
    {
      w_eng_vdwl[i] = MMD_float(0.0);
      w_virial[i]   = MMD_float(0.0);
    }
    #pragma omp distribute parallel for  // reduction(+:t_eng_vdwl, t_virial)
#else
    #pragma omp parallel for reduction(+ : t_eng_vdwl, t_virial)
#endif
    for(int i = 0; i < nlocal; i++)
    {
      const int        tid       = omp_get_thread_num();
      const int *const neighs    = neighbors + i * maxneighs;
      const int        numneighs = numneigh[i];
      const MMD_float  xtmp      = x[i * PAD + 0];
      const MMD_float  ytmp      = x[i * PAD + 1];
      const MMD_float  ztmp      = x[i * PAD + 2];
      const int        type_i    = type[i];
      MMD_float        fix       = MMD_float(0.0);
      MMD_float        fiy       = MMD_float(0.0);
      MMD_float        fiz       = MMD_float(0.0);

      // pragma omp simd forces vectorization (ignoring the performance objections of the compiler)
      // also give hint to use certain vectorlength for MIC, Sandy Bridge and WESTMERE this should be be 8 here
      // give hint to compiler that fix, fiy and fiz are used for reduction only

// clang-format off
#ifndef USE_OFFLOAD
#ifdef USE_SIMD
      #pragma vector unaligned
      #pragma omp simd reduction(+:fix, fiy, fiz, t_eng_vdwl, t_virial)
#endif
#endif // clang-format on
      for(int k = 0; k < numneighs; k++)
      {
        const int       j       = neighs[k];
        const MMD_float delx    = xtmp - x[j * PAD + 0];
        const MMD_float dely    = ytmp - x[j * PAD + 1];
        const MMD_float delz    = ztmp - x[j * PAD + 2];
        const int       type_j  = type[j];
        const MMD_float rsq     = delx * delx + dely * dely + delz * delz;
        const int       type_ij = type_i * ntypes + type_j;

        if(rsq < cutforcesq[type_ij])
        {
          const MMD_float sr2   = MMD_float(1.0) / rsq;
          const MMD_float sr6   = sr2 * sr2 * sr2 * sigma6[type_ij];
          const MMD_float force = MMD_float(48.0) * sr6 * (sr6 - MMD_float(0.5)) * sr2 * epsilon[type_ij];
          fix += delx * force;
          fiy += dely * force;
          fiz += delz * force;

          if(EVFLAG)
          {
#ifdef USE_OFFLOAD
            w_eng_vdwl[tid] += sr6 * (sr6 - MMD_float(1.0)) * epsilon[type_ij];
            w_virial[tid] += (delx * delx + dely * dely + delz * delz) * force;
#else
            t_eng_vdwl += sr6 * (sr6 - MMD_float(1.0)) * epsilon[type_ij];
            t_virial += (delx * delx + dely * dely + delz * delz) * force;
#endif
          }
        }
      }

      f[i * PAD + 0] += fix;
      f[i * PAD + 1] += fiy;
      f[i * PAD + 2] += fiz;
    }
#ifdef USE_OFFLOAD
    if(EVFLAG)
    {
      MMD_float team_eng_vdwl = MMD_float(0.0);
      MMD_float team_virial   = MMD_float(0.0);
      for(int i = 0; i < nthr; ++i)
      {
        team_eng_vdwl += w_eng_vdwl[i];
        team_virial += w_virial[i];
      }
      atomic_add(&t_eng_vdwl, team_eng_vdwl);
      atomic_add(&t_virial, team_virial);
    }
#endif
  }

#ifdef USE_OFFLOAD
  #pragma omp target update from(f[0:nall*PAD])
  #pragma omp target exit data map(delete:f[0:nall * PAD])
  #pragma omp target exit data map(delete:x[0:nall * PAD])
  #pragma omp target exit data map(delete:type[0:nall])
#endif

  if(EVFLAG)
  {
    t_eng_vdwl *= MMD_float(4.0);
    t_virial *= MMD_float(0.5);

    eng_vdwl += t_eng_vdwl;
    virial += t_virial;
  }
}
