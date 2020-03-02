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

#include "stdio.h"
#include "stdlib.h"

#include "neighbor.h"
#include "offload.h"
#include "openmp.h"

#define FACTOR 0.999
#define SMALL 1.0e-6

Neighbor::Neighbor(int ntypes_)
{
  ncalls         = 0;
  ntypes         = ntypes_;
  max_totalneigh = 0;
  numneigh       = NULL;
  neighbors      = NULL;
  maxneighs      = 100;
  nmax           = 0;
  bincount       = NULL;
  bins           = NULL;
  atoms_per_bin  = 8;
  stencil        = NULL;
  threads        = NULL;
  halfneigh      = 0;
  ghost_newton   = 1;
  cutneighsq     = ( MMD_float * )mmd_alloc(ntypes * ntypes * sizeof(MMD_float));
}

Neighbor::~Neighbor()
{
  mmd_free(cutneighsq);
  mmd_free(numneigh);
  mmd_free(neighbors);
  mmd_free(bincount);
  mmd_free(bins);
}

/* binned neighbor list construction with full Newton's 3rd law
   every pair stored exactly once by some processor
   each owned atom i checks its own bin and other bins in Newton stencil */

void Neighbor::build(Atom &atom)
{
  ncalls++;
  const int nlocal = atom.nlocal;
  const int nall   = atom.nlocal + atom.nghost;

  /* extend atom arrays if necessary */

  if(nall > nmax)
  {
    nmax      = nall;
    numneigh  = ( int * )mmd_replace_alloc(numneigh, nmax * sizeof(int));
    neighbors = ( int * )mmd_replace_alloc(neighbors, nmax * maxneighs * sizeof(int));
  }

  /* bin local & ghost atoms */

  binatoms(atom);
  count = 0;

  /* loop over each atom, storing neighbors */

  const MMD_float *const x      = atom.x;
  const int *const       type   = atom.type;
  int                    ntypes = atom.ntypes;

  int resize = 1;

#ifdef USE_OFFLOAD
  // FIXME: Workaround for automatic copying of class members.
  int        count         = this->count;
  int        maxneighs     = this->maxneighs;
  int *      neighbors     = this->neighbors;
  int *      numneigh      = this->numneigh;
  int *      bins          = this->bins;
  int *      bincount      = this->bincount;
  int *      stencil       = this->stencil;
  MMD_float *cutneighsq    = this->cutneighsq;
  int        nstencil      = this->nstencil;
  int        atoms_per_bin = this->atoms_per_bin;
  int        halfneigh     = this->halfneigh;
  int        ghost_newton  = this->ghost_newton;
  MMD_float  xprd          = this->xprd;
  MMD_float  yprd          = this->yprd;
  MMD_float  zprd          = this->zprd;
  int        mbinx         = this->mbinx;
  int        mbiny         = this->mbiny;
  int        mbinxlo       = this->mbinxlo;
  int        mbinylo       = this->mbinylo;
  int        mbinzlo       = this->mbinzlo;
  MMD_float  bininvx       = this->bininvx;
  MMD_float  bininvy       = this->bininvy;
  MMD_float  bininvz       = this->bininvz;
  int        nbinx         = this->nbinx;
  int        nbiny         = this->nbiny;
  int        nbinz         = this->nbinz;
#endif

  while(resize)
  {
    int new_maxneighs = maxneighs;
    resize            = 0;

#ifdef USE_OFFLOAD
    #pragma omp target teams distribute parallel for map(tofrom:resize, new_maxneighs) num_teams(heuristic_nteam(atom.nlocal)) thread_limit(MAX_TEAM_SIZE)
#else
    #pragma omp parallel for
#endif
    for(int i = 0; i < nlocal; i++)
    {
      numneigh[i]   = 0;
      int *neighptr = &neighbors[i * maxneighs];

      const MMD_float xtmp = x[i * PAD + 0];
      const MMD_float ytmp = x[i * PAD + 1];
      const MMD_float ztmp = x[i * PAD + 2];

      const int type_i = type[i];

      // Manually inlined coord2bin
      // FIXME: Workaround for automatic copying of class members.
      // const int ibin = coord2bin(xtmp, ytmp, ztmp);
      int ix, iy, iz;
      if(xtmp >= xprd)
      {
        ix = ( int )((xtmp - xprd) * bininvx) + nbinx - mbinxlo;
      }
      else if(xtmp >= 0.0)
      {
        ix = ( int )(xtmp * bininvx) - mbinxlo;
      }
      else
      {
        ix = ( int )(xtmp * bininvx) - mbinxlo - 1;
      }
      if(ytmp >= yprd)
      {
        iy = ( int )((ytmp - yprd) * bininvy) + nbiny - mbinylo;
      }
      else if(ytmp >= 0.0)
      {
        iy = ( int )(ytmp * bininvy) - mbinylo;
      }
      else
      {
        iy = ( int )(ytmp * bininvy) - mbinylo - 1;
      }
      if(ztmp >= zprd)
      {
        iz = ( int )((ztmp - zprd) * bininvz) + nbinz - mbinzlo;
      }
      else if(ztmp >= 0.0)
      {
        iz = ( int )(ztmp * bininvz) - mbinzlo;
      }
      else
      {
        iz = ( int )(ztmp * bininvz) - mbinzlo - 1;
      }
      int ibin = (iz * mbiny * mbinx + iy * mbinx + ix + 1);

      for(int k = 0; k < nstencil; k++)
      {
        const int jbin = ibin + stencil[k];

        int *loc_bin = &bins[jbin * atoms_per_bin];

        if(ibin == jbin)
        {
          for(int m = 0; m < bincount[jbin]; m++)
          {
            const int j = loc_bin[m];

            // for same bin as atom i skip j if i==j and skip atoms "below and to the left" if using halfneighborlists
            if(((j == i) || (halfneigh && !ghost_newton && (j < i)) || (halfneigh && ghost_newton && ((j < i) || ((j >= nlocal) && ((x[j * PAD + 2] < ztmp) || (x[j * PAD + 2] == ztmp && x[j * PAD + 1] < ytmp) || (x[j * PAD + 2] == ztmp && x[j * PAD + 1] == ytmp && x[j * PAD + 0] < xtmp)))))))
            {
              continue;
            }

            const MMD_float delx   = xtmp - x[j * PAD + 0];
            const MMD_float dely   = ytmp - x[j * PAD + 1];
            const MMD_float delz   = ztmp - x[j * PAD + 2];
            const int       type_j = type[j];
            const MMD_float rsq    = delx * delx + dely * dely + delz * delz;

            // TODO: We should do this differently for SIMD
            if(rsq <= cutneighsq[type_i * ntypes + type_j])
            {
              int idx       = numneigh[i]++;
              neighptr[idx] = j;
            }
          }
        }
        else
        {
          for(int m = 0; m < bincount[jbin]; m++)
          {
            const int j = loc_bin[m];

            if(halfneigh && !ghost_newton && (j < i))
            {
              continue;
            }

            const MMD_float delx   = xtmp - x[j * PAD + 0];
            const MMD_float dely   = ytmp - x[j * PAD + 1];
            const MMD_float delz   = ztmp - x[j * PAD + 2];
            const int       type_j = type[j];
            const MMD_float rsq    = delx * delx + dely * dely + delz * delz;

            // TODO: We should do this differently for SIMD
            if(rsq <= cutneighsq[type_i * ntypes + type_j])
            {
              int idx       = numneigh[i]++;
              neighptr[idx] = j;
            }
          }
        }
      }

      if(numneigh[i] >= maxneighs)
      {
        resize = 1;

        if(numneigh[i] >= new_maxneighs)
        {
          new_maxneighs = numneigh[i];
        }
      }
    }

    if(resize)
    {
      maxneighs = new_maxneighs * 1.2;
      neighbors = ( int * )mmd_replace_alloc(neighbors, nmax * maxneighs * sizeof(int));
    }
  }

#ifdef USE_OFFLOAD
  // FIXME: Workaround for automatic copying of class members.
  this->maxneighs = maxneighs;
  this->neighbors = neighbors;
#endif
}

void Neighbor::binatoms(Atom &atom, int count)
{
  const int              nall = count < 0 ? atom.nlocal + atom.nghost : count;
  const MMD_float *const x    = atom.x;

  xprd = atom.box.xprd;
  yprd = atom.box.yprd;
  zprd = atom.box.zprd;

  int resize = 1;

#ifdef USE_OFFLOAD
  // FIXME: Workaround for automatic copying of class members.
  const int mbins         = this->mbins;
  int       atoms_per_bin = this->atoms_per_bin;
  int *     bins          = this->bins;
  int *     bincount      = this->bincount;
  MMD_float xprd          = this->xprd;
  MMD_float yprd          = this->yprd;
  MMD_float zprd          = this->zprd;
  int       mbinx         = this->mbinx;
  int       mbiny         = this->mbiny;
  int       mbinxlo       = this->mbinxlo;
  int       mbinylo       = this->mbinylo;
  int       mbinzlo       = this->mbinzlo;
  MMD_float bininvx       = this->bininvx;
  MMD_float bininvy       = this->bininvy;
  MMD_float bininvz       = this->bininvz;
  int       nbinx         = this->nbinx;
  int       nbiny         = this->nbiny;
  int       nbinz         = this->nbinz;
#endif

  while(resize > 0)
  {
    resize = 0;

#ifdef USE_OFFLOAD
    #pragma omp target teams distribute parallel for
#else
    #pragma omp parallel for
#endif
    for(int i = 0; i < mbins; i++)
    {
      bincount[i] = 0;
    }

#ifdef USE_OFFLOAD
    #pragma omp target teams distribute parallel for
#else
    #pragma omp parallel for
#endif
    for(int i = 0; i < mbins * atoms_per_bin; ++i)
    {
      bins[i] = 0;
    }

#ifdef USE_OFFLOAD
    #pragma omp target teams distribute parallel for map(tofrom:resize)
#else
    #pragma omp parallel for
#endif
    for(int i = 0; i < nall; i++)
    {
      // Manually inlined coord2bin
      // FIXME: Workaround for automatic copying of class members.
      // const int ibin = coord2bin(x[i * PAD + 0], x[i * PAD + 1], x[i * PAD + 2]);
      MMD_float px = x[i * PAD + 0];
      MMD_float py = x[i * PAD + 1];
      MMD_float pz = x[i * PAD + 2];
      int       ix, iy, iz;
      if(px >= xprd)
      {
        ix = ( int )((px - xprd) * bininvx) + nbinx - mbinxlo;
      }
      else if(px >= 0.0)
      {
        ix = ( int )(px * bininvx) - mbinxlo;
      }
      else
      {
        ix = ( int )(px * bininvx) - mbinxlo - 1;
      }
      if(py >= yprd)
      {
        iy = ( int )((py - yprd) * bininvy) + nbiny - mbinylo;
      }
      else if(py >= 0.0)
      {
        iy = ( int )(py * bininvy) - mbinylo;
      }
      else
      {
        iy = ( int )(py * bininvy) - mbinylo - 1;
      }
      if(pz >= zprd)
      {
        iz = ( int )((pz - zprd) * bininvz) + nbinz - mbinzlo;
      }
      else if(pz >= 0.0)
      {
        iz = ( int )(pz * bininvz) - mbinzlo;
      }
      else
      {
        iz = ( int )(pz * bininvz) - mbinzlo - 1;
      }
      int ibin = (iz * mbiny * mbinx + iy * mbinx + ix + 1);

      if(bincount[ibin] < atoms_per_bin)
      {
        int ac;
        #pragma omp atomic capture
        ac                              = bincount[ibin]++;
        bins[ibin * atoms_per_bin + ac] = i;
      }
      else
      {
        resize = 1;
      }
    }

    if(resize)
    {
      atoms_per_bin *= 2;
      bins = ( int * )mmd_replace_alloc(bins, mbins * atoms_per_bin * sizeof(int));
    }
  }

#ifdef USE_OFFLOAD
  // FIXME: Workaround for automatic copying of class members.
  this->atoms_per_bin = atoms_per_bin;
  this->bins          = bins;
#endif
}

/* convert xyz atom coords into local bin #
   take special care to insure ghost atoms with
   coord >= prd or coord < 0.0 are put in correct bins */

inline int Neighbor::coord2bin(MMD_float x, MMD_float y, MMD_float z)
{
  int ix, iy, iz;

  if(x >= xprd)
  {
    ix = ( int )((x - xprd) * bininvx) + nbinx - mbinxlo;
  }
  else if(x >= 0.0)
  {
    ix = ( int )(x * bininvx) - mbinxlo;
  }
  else
  {
    ix = ( int )(x * bininvx) - mbinxlo - 1;
  }

  if(y >= yprd)
  {
    iy = ( int )((y - yprd) * bininvy) + nbiny - mbinylo;
  }
  else if(y >= 0.0)
  {
    iy = ( int )(y * bininvy) - mbinylo;
  }
  else
  {
    iy = ( int )(y * bininvy) - mbinylo - 1;
  }

  if(z >= zprd)
  {
    iz = ( int )((z - zprd) * bininvz) + nbinz - mbinzlo;
  }
  else if(z >= 0.0)
  {
    iz = ( int )(z * bininvz) - mbinzlo;
  }
  else
  {
    iz = ( int )(z * bininvz) - mbinzlo - 1;
  }

  return (iz * mbiny * mbinx + iy * mbinx + ix + 1);
}


/*
setup neighbor binning parameters
bin numbering is global: 0 = 0.0 to binsize
                         1 = binsize to 2*binsize
                         nbin-1 = prd-binsize to binsize
                         nbin = prd to prd+binsize
                         -1 = -binsize to 0.0
coord = lowest and highest values of ghost atom coords I will have
        add in "small" for round-off safety
mbinlo = lowest global bin any of my ghost atoms could fall into
mbinhi = highest global bin any of my ghost atoms could fall into
mbin = number of bins I need in a dimension
stencil() = bin offsets in 1-d sense for stencil of surrounding bins
*/

int Neighbor::setup(Atom &atom)
{
  int       i, j, k, nmax;
  MMD_float coord;
  int       mbinxhi, mbinyhi, mbinzhi;
  int       nextx, nexty, nextz;

  for(int i = 0; i < ntypes * ntypes; i++)
  {
    cutneighsq[i] = cutneigh * cutneigh;
  }
#ifdef USE_OFFLOAD
  #pragma omp target update to(cutneighsq[0:ntypes * ntypes])
#endif

  const MMD_float xprd = atom.box.xprd;
  const MMD_float yprd = atom.box.yprd;
  const MMD_float zprd = atom.box.zprd;

  /*
  c bins must evenly divide into box size,
  c   becoming larger than cutneigh if necessary
  c binsize = 1/2 of cutoff is near optimal

  if (flag == 0) {
    nbinx = 2.0 * xprd / cutneigh;
    nbiny = 2.0 * yprd / cutneigh;
    nbinz = 2.0 * zprd / cutneigh;
    if (nbinx == 0) nbinx = 1;
    if (nbiny == 0) nbiny = 1;
    if (nbinz == 0) nbinz = 1;
  }
  */

  binsizex = xprd / nbinx;
  binsizey = yprd / nbiny;
  binsizez = zprd / nbinz;
  bininvx  = 1.0 / binsizex;
  bininvy  = 1.0 / binsizey;
  bininvz  = 1.0 / binsizez;

  coord   = atom.box.xlo - cutneigh - SMALL * xprd;
  mbinxlo = static_cast<int>(coord * bininvx);

  if(coord < 0.0)
  {
    mbinxlo = mbinxlo - 1;
  }

  coord   = atom.box.xhi + cutneigh + SMALL * xprd;
  mbinxhi = static_cast<int>(coord * bininvx);

  coord   = atom.box.ylo - cutneigh - SMALL * yprd;
  mbinylo = static_cast<int>(coord * bininvy);

  if(coord < 0.0)
  {
    mbinylo = mbinylo - 1;
  }

  coord   = atom.box.yhi + cutneigh + SMALL * yprd;
  mbinyhi = static_cast<int>(coord * bininvy);

  coord   = atom.box.zlo - cutneigh - SMALL * zprd;
  mbinzlo = static_cast<int>(coord * bininvz);

  if(coord < 0.0)
  {
    mbinzlo = mbinzlo - 1;
  }

  coord   = atom.box.zhi + cutneigh + SMALL * zprd;
  mbinzhi = static_cast<int>(coord * bininvz);

  /* extend bins by 1 in each direction to insure stencil coverage */

  mbinxlo = mbinxlo - 1;
  mbinxhi = mbinxhi + 1;
  mbinx   = mbinxhi - mbinxlo + 1;

  mbinylo = mbinylo - 1;
  mbinyhi = mbinyhi + 1;
  mbiny   = mbinyhi - mbinylo + 1;

  mbinzlo = mbinzlo - 1;
  mbinzhi = mbinzhi + 1;
  mbinz   = mbinzhi - mbinzlo + 1;

  /*
  compute bin stencil of all bins whose closest corner to central bin
  is within neighbor cutoff
  for partial Newton (newton = 0),
  stencil is all surrounding bins including self
  for full Newton (newton = 1),
  stencil is bins to the "upper right" of central bin, does NOT include self
  next(xyz) = how far the stencil could possibly extend
  factor < 1.0 for special case of LJ benchmark so code will create
  correct-size stencil when there are 3 bins for every 5 lattice spacings
  */

  nextx = static_cast<int>(cutneigh * bininvx);

  if(nextx * binsizex < FACTOR * cutneigh)
  {
    nextx++;
  }

  nexty = static_cast<int>(cutneigh * bininvy);

  if(nexty * binsizey < FACTOR * cutneigh)
  {
    nexty++;
  }

  nextz = static_cast<int>(cutneigh * bininvz);

  if(nextz * binsizez < FACTOR * cutneigh)
  {
    nextz++;
  }

  nmax = (2 * nextz + 1) * (2 * nexty + 1) * (2 * nextx + 1);

  mmd_free(stencil);
  stencil = ( int * )mmd_alloc(nmax * sizeof(int));

  nstencil   = 0;
  int kstart = -nextz;

  if(halfneigh && ghost_newton)
  {
    kstart              = 0;
    stencil[nstencil++] = 0;
  }

  for(k = kstart; k <= nextz; k++)
  {
    for(j = -nexty; j <= nexty; j++)
    {
      for(i = -nextx; i <= nextx; i++)
      {
        if(!ghost_newton || !halfneigh || (k > 0 || j > 0 || (j == 0 && i > 0)))
        {
          if(bindist(i, j, k) < cutneighsq[0])
          {
            stencil[nstencil++] = k * mbiny * mbinx + j * mbinx + i;
          }
        }
      }
    }
  }
#ifdef USE_OFFLOAD
  #pragma omp target update to(stencil[0:nmax])
#endif

  mbins = mbinx * mbiny * mbinz;

  mmd_free(bincount);
  bincount = ( int * )mmd_alloc(mbins * sizeof(int));

  mmd_free(bins);
  bins = ( int * )mmd_alloc(mbins * atoms_per_bin * sizeof(int));

  return 0;
}

/* compute closest distance between central bin (0,0,0) and bin (i,j,k) */

MMD_float Neighbor::bindist(int i, int j, int k)
{
  MMD_float delx, dely, delz;

  if(i > 0)
  {
    delx = (i - 1) * binsizex;
  }
  else if(i == 0)
  {
    delx = 0.0;
  }
  else
  {
    delx = (i + 1) * binsizex;
  }

  if(j > 0)
  {
    dely = (j - 1) * binsizey;
  }
  else if(j == 0)
  {
    dely = 0.0;
  }
  else
  {
    dely = (j + 1) * binsizey;
  }

  if(k > 0)
  {
    delz = (k - 1) * binsizez;
  }
  else if(k == 0)
  {
    delz = 0.0;
  }
  else
  {
    delz = (k + 1) * binsizez;
  }

  return (delx * delx + dely * dely + delz * delz);
}
