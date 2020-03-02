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

#include "atom.h"
#include "mpi.h"
#include "neighbor.h"
#include "offload.h"
#include "util.h"

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#define DELTA 20000

Atom::Atom(int ntypes_)
{
  natoms    = 0;
  nlocal    = 0;
  nghost    = 0;
  nmax      = 0;
  copy_size = 0;

  x = v = f = xold = x_copy = v_copy = NULL;
  type = type_copy = NULL;
  comm_size        = 3;
  reverse_size     = 3;
  border_size      = 4;

  mass = 1;

  ntypes = ntypes_;
}

Atom::~Atom()
{
  if(nmax)
  {
    mmd_free(x);
    mmd_free(v);
    mmd_free(f);
    mmd_free(xold);
    mmd_free(type);
  }
}

void Atom::growarray_sync()
{
#ifdef USE_OFFLOAD
  {
    int        nlocal = this->nlocal;
    int        nghost = this->nghost;
    int        nmax   = this->nmax;
    MMD_float *x      = this->x;
    MMD_float *v      = this->v;
    MMD_float *f      = this->f;
    int *      type   = this->type;
    MMD_float *xold   = this->xold;
    size_t     N      = nmax * PAD;
    #pragma omp target update from(x[0:N])
    #pragma omp target update from(v[0:N])
    #pragma omp target update from(f[0:N])
    #pragma omp target update from(type[0:nmax])
    #pragma omp target update from(xold[0:N])
  }
#endif
  growarray();
#ifdef USE_OFFLOAD
  {
    int        nlocal = this->nlocal;
    int        nghost = this->nghost;
    int        nmax   = this->nmax;
    MMD_float *x      = this->x;
    MMD_float *v      = this->v;
    MMD_float *f      = this->f;
    int *      type   = this->type;
    MMD_float *xold   = this->xold;
    size_t     N      = nmax * PAD;
    #pragma omp target update to(x[0:N])
    #pragma omp target update to(v[0:N])
    #pragma omp target update to(f[0:N])
    #pragma omp target update to(type[0:nmax])
    #pragma omp target update to(xold[0:N])
  }
#endif
}

void Atom::growarray()
{
  int nold = nmax;
  nmax += DELTA;

  x    = ( MMD_float * )mmd_grow_alloc(x, nold * PAD * sizeof(MMD_float), nmax * PAD * sizeof(MMD_float));
  v    = ( MMD_float * )mmd_grow_alloc(v, nold * PAD * sizeof(MMD_float), nmax * PAD * sizeof(MMD_float));
  f    = ( MMD_float * )mmd_grow_alloc(f, nold * PAD * sizeof(MMD_float), nmax * PAD * sizeof(MMD_float));
  type = ( int * )mmd_grow_alloc(type, nold * sizeof(int), nmax * sizeof(int));
  xold = ( MMD_float * )mmd_grow_alloc(xold, nold * PAD * sizeof(MMD_float), nmax * PAD * sizeof(MMD_float));

  if(x == NULL || v == NULL || f == NULL || xold == NULL)
  {
    printf("ERROR: No memory for atoms\n");
  }
}

void Atom::addatom(MMD_float x_in, MMD_float y_in, MMD_float z_in, MMD_float vx_in, MMD_float vy_in, MMD_float vz_in)
{
  if(nlocal == nmax)
  {
    growarray();
  }

  x[nlocal * PAD + 0] = x_in;
  x[nlocal * PAD + 1] = y_in;
  x[nlocal * PAD + 2] = z_in;
  v[nlocal * PAD + 0] = vx_in;
  v[nlocal * PAD + 1] = vy_in;
  v[nlocal * PAD + 2] = vz_in;
  type[nlocal]        = rand() % ntypes;

  nlocal++;
}

/* enforce PBC
   order of 2 tests is important to insure lo-bound <= coord < hi-bound
   even with round-off errors where (coord +/- epsilon) +/- period = bound */

void Atom::pbc()
{
#ifdef USE_OFFLOAD
  // FIXME: Workaround for automatic copying of class members.
  int        nlocal = this->nlocal;
  Box        box    = this->box;
  MMD_float *x      = this->x;
#endif

#ifdef USE_OFFLOAD
  #pragma omp target teams distribute parallel for
#else
  #pragma omp parallel for
#endif
  for(int i = 0; i < nlocal; i++)
  {
    if(x[i * PAD + 0] < 0.0)
    {
      x[i * PAD + 0] += box.xprd;
    }

    if(x[i * PAD + 0] >= box.xprd)
    {
      x[i * PAD + 0] -= box.xprd;
    }

    if(x[i * PAD + 1] < 0.0)
    {
      x[i * PAD + 1] += box.yprd;
    }

    if(x[i * PAD + 1] >= box.yprd)
    {
      x[i * PAD + 1] -= box.yprd;
    }

    if(x[i * PAD + 2] < 0.0)
    {
      x[i * PAD + 2] += box.zprd;
    }

    if(x[i * PAD + 2] >= box.zprd)
    {
      x[i * PAD + 2] -= box.zprd;
    }
  }
}

void Atom::copy(int i, int j)
{
  x[j * PAD + 0] = x[i * PAD + 0];
  x[j * PAD + 1] = x[i * PAD + 1];
  x[j * PAD + 2] = x[i * PAD + 2];
  v[j * PAD + 0] = v[i * PAD + 0];
  v[j * PAD + 1] = v[i * PAD + 1];
  v[j * PAD + 2] = v[i * PAD + 2];
  type[j]        = type[i];
}

void Atom::pack_comm(int n, int *list, MMD_float *buf, const int *pbc_flags)
{
#ifdef USE_OFFLOAD
  // FIXME: Workaround for automatic copying of class members.
  MMD_float *x = this->x;
#endif
  const MMD_float xprd = box.xprd;
  const MMD_float yprd = box.yprd;
  const MMD_float zprd = box.zprd;


  if(pbc_flags[0] == 0)
  {
#ifdef USE_OFFLOAD
    #pragma omp target teams distribute parallel for
#else
    #pragma omp parallel for
#endif
    for(int i = 0; i < n; i++)
    {
      const int j = list[i];

      buf[3 * i]     = x[j * PAD + 0];
      buf[3 * i + 1] = x[j * PAD + 1];
      buf[3 * i + 2] = x[j * PAD + 2];
    }
  }
  else
  {
#ifdef USE_OFFLOAD
    #pragma omp target teams distribute parallel for
#else
    #pragma omp parallel for
#endif
    for(int i = 0; i < n; i++)
    {
      const int j = list[i];

      buf[3 * i]     = x[j * PAD + 0] + pbc_flags[1] * xprd;
      buf[3 * i + 1] = x[j * PAD + 1] + pbc_flags[2] * yprd;
      buf[3 * i + 2] = x[j * PAD + 2] + pbc_flags[3] * zprd;
    }
  }
}

void Atom::unpack_comm(int n, int first, MMD_float *buf)
{
#ifdef USE_OFFLOAD
  // FIXME: Workaround for automatic copying of class members.
  MMD_float *x = this->x;
#endif

#ifdef USE_OFFLOAD
  #pragma omp target teams distribute parallel for
#else
  #pragma omp parallel for
#endif
  for(int i = 0; i < n; i++)
  {
    x[(first + i) * PAD + 0] = buf[3 * i];
    x[(first + i) * PAD + 1] = buf[3 * i + 1];
    x[(first + i) * PAD + 2] = buf[3 * i + 2];
  }
}

void Atom::self_comm(int n, int *list, int first, const int *pbc_flags)
{
#ifdef USE_OFFLOAD
  // FIXME: Workaround for automatic copying of class members.
  MMD_float *x = this->x;
#endif
  const MMD_float xprd = box.xprd;
  const MMD_float yprd = box.yprd;
  const MMD_float zprd = box.zprd;

  if(pbc_flags[0] == 0)
  {
#ifdef USE_OFFLOAD
    #pragma omp target teams distribute parallel for
#else
    #pragma omp parallel for
#endif
    for(int i = 0; i < n; i++)
    {
      const int j = list[i];

      x[(first + i) * PAD + 0] = x[j * PAD + 0];
      x[(first + i) * PAD + 1] = x[j * PAD + 1];
      x[(first + i) * PAD + 2] = x[j * PAD + 2];
    }
  }
  else
  {
#ifdef USE_OFFLOAD
    #pragma omp target teams distribute parallel for
#else
    #pragma omp parallel for
#endif
    for(int i = 0; i < n; i++)
    {
      const int j = list[i];

      x[(first + i) * PAD + 0] = x[j * PAD + 0] + pbc_flags[1] * xprd;
      x[(first + i) * PAD + 1] = x[j * PAD + 1] + pbc_flags[2] * yprd;
      x[(first + i) * PAD + 2] = x[j * PAD + 2] + pbc_flags[3] * zprd;
    }
  }
}

void Atom::pack_reverse(int n, int first, MMD_float *buf)
{
#ifdef USE_OFFLOAD
  // FIXME: Workaround for automatic copying of class members.
  MMD_float *f = this->f;
#endif

#ifdef USE_OFFLOAD
  #pragma omp target teams distribute parallel for
#else
  #pragma omp parallel for
#endif
  for(int i = 0; i < n; i++)
  {
    buf[3 * i]     = f[(first + i) * PAD + 0];
    buf[3 * i + 1] = f[(first + i) * PAD + 1];
    buf[3 * i + 2] = f[(first + i) * PAD + 2];
  }
}

void Atom::unpack_reverse(int n, int *list, MMD_float *buf)
{
#ifdef USE_OFFLOAD
  // FIXME: Workaround for automatic copying of class members.
  MMD_float *f = this->f;
#endif

#ifdef USE_OFFLOAD
  #pragma omp target teams distribute parallel for
#else
  #pragma omp parallel for
#endif
  for(int i = 0; i < n; i++)
  {
    const int j = list[i];

    f[j * PAD + 0] += buf[3 * i];
    f[j * PAD + 1] += buf[3 * i + 1];
    f[j * PAD + 2] += buf[3 * i + 2];
  }
}

int Atom::pack_border(int i, MMD_float *buf, int *pbc_flags)
{
  int m = 0;

  if(pbc_flags[0] == 0)
  {
    buf[m++] = x[i * PAD + 0];
    buf[m++] = x[i * PAD + 1];
    buf[m++] = x[i * PAD + 2];
    buf[m++] = type[i];
  }
  else
  {
    buf[m++] = x[i * PAD + 0] + pbc_flags[1] * box.xprd;
    buf[m++] = x[i * PAD + 1] + pbc_flags[2] * box.yprd;
    buf[m++] = x[i * PAD + 2] + pbc_flags[3] * box.zprd;
    buf[m++] = type[i];
  }

  return m;
}

int Atom::unpack_border(int i, MMD_float *buf)
{
  if(i == nmax)
  {
    growarray_sync();
  }

  int m = 0;

  x[i * PAD + 0] = buf[m++];
  x[i * PAD + 1] = buf[m++];
  x[i * PAD + 2] = buf[m++];
  type[i]        = buf[m++];
  return m;
}

int Atom::pack_exchange(int i, MMD_float *buf)
{
  int m = 0;

  buf[m++] = x[i * PAD + 0];
  buf[m++] = x[i * PAD + 1];
  buf[m++] = x[i * PAD + 2];
  buf[m++] = v[i * PAD + 0];
  buf[m++] = v[i * PAD + 1];
  buf[m++] = v[i * PAD + 2];
  buf[m++] = type[i];
  return m;
}

int Atom::unpack_exchange(int i, MMD_float *buf)
{
  if(i == nmax)
  {
    growarray_sync();
  }

  int m = 0;

  x[i * PAD + 0] = buf[m++];
  x[i * PAD + 1] = buf[m++];
  x[i * PAD + 2] = buf[m++];
  v[i * PAD + 0] = buf[m++];
  v[i * PAD + 1] = buf[m++];
  v[i * PAD + 2] = buf[m++];
  type[i]        = buf[m++];
  return m;
}

int Atom::skip_exchange(MMD_float *buf) { return 7; }

void Atom::sort(Neighbor &neighbor)
{
  neighbor.binatoms(*this, nlocal);

  int *binpos = neighbor.bincount;
  int *bins   = neighbor.bins;

  const int mbins         = neighbor.mbins;
  const int atoms_per_bin = neighbor.atoms_per_bin;

  blelloch_excl_scan_target_inplace(binpos, mbins);

  if(copy_size < nmax)
  {
    x_copy    = ( MMD_float * )mmd_replace_alloc(x_copy, nmax * PAD * sizeof(MMD_float));
    v_copy    = ( MMD_float * )mmd_replace_alloc(v_copy, nmax * PAD * sizeof(MMD_float));
    type_copy = ( int * )mmd_replace_alloc(type_copy, nmax * sizeof(int));
    copy_size = nmax;
  }

  MMD_float *new_x    = x_copy;
  MMD_float *new_v    = v_copy;
  int *      new_type = type_copy;
  MMD_float *old_x    = x;
  MMD_float *old_v    = v;
  int *      old_type = type;

#ifdef USE_OFFLOAD
  #pragma omp target teams distribute parallel for num_teams(heuristic_nteam(nlocal)) thread_limit(MAX_TEAM_SIZE)
  #else
  #pragma omp parallel for
#endif
  for(int mybin = 0; mybin < mbins; mybin++)
  {
    const int start = mybin > 0 ? binpos[mybin - 1] : 0;
    const int count = binpos[mybin] - start;
    for(int k = 0; k < count; k++)
    {
      const int new_i = start + k;
      const int old_i = bins[mybin * atoms_per_bin + k];

      new_x[new_i * PAD + 0] = old_x[old_i * PAD + 0];
      new_x[new_i * PAD + 1] = old_x[old_i * PAD + 1];
      new_x[new_i * PAD + 2] = old_x[old_i * PAD + 2];
      new_v[new_i * PAD + 0] = old_v[old_i * PAD + 0];
      new_v[new_i * PAD + 1] = old_v[old_i * PAD + 1];
      new_v[new_i * PAD + 2] = old_v[old_i * PAD + 2];
      new_type[new_i]        = old_type[old_i];
    }
  }

  // TODO: Check that this sort of pointer swapping doesn't cause problems
  MMD_float *x_tmp    = x;
  MMD_float *v_tmp    = v;
  int *      type_tmp = type;

  x         = x_copy;
  v         = v_copy;
  type      = type_copy;
  x_copy    = x_tmp;
  v_copy    = v_tmp;
  type_copy = type_tmp;
}
