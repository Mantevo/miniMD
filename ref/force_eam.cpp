/* ----------------------------------------------------------------------
   miniMD is a simple, parallel molecular dynamics (MD) code.   miniMD is
   an MD microapplication in the Mantevo project at Sandia National
   Laboratories ( http://www.mantevo.org ). The primary
   authors of miniMD are Steve Plimpton (sjplimp@sandia.gov) , Paul Crozier
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

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "force_eam.h"
#include "atom.h"
#include "comm.h"
#include "neighbor.h"
#include "memory.h"

#define MAXLINE 1024

#define MAX(a,b) (a>b?a:b)
#define MIN(a,b) (a<b?a:b)

/* ---------------------------------------------------------------------- */

ForceEAM::ForceEAM(int ntypes_)
{
  ntypes = ntypes_;
  cutforce = 0.0;
  cutforcesq = new MMD_float[ntypes*ntypes];
  for( int i = 0; i<ntypes*ntypes; i++)
    cutforcesq[i] = 0.0;
  use_oldcompute = 0;

  nmax = 0;

  rho = 0;
  fp = 0;
  style = FORCEEAM;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

ForceEAM::~ForceEAM()
{

}

void ForceEAM::setup()
{
  me = threads->mpi_me;
  coeff("Cu_u6.eam");
  init_style();
}


void ForceEAM::compute(Atom &atom, Neighbor &neighbor, Comm &comm, int me)
{
  if(neighbor.halfneigh) {
    if(threads->omp_num_threads > 1)
      return ;
    else
      return compute_halfneigh(atom, neighbor, comm, me);
  } else return compute_fullneigh(atom, neighbor, comm, me);

}
/* ---------------------------------------------------------------------- */

void ForceEAM::compute_halfneigh(Atom &atom, Neighbor &neighbor, Comm &comm, int me)
{

  MMD_float evdwl = 0.0;

  virial = 0;
  // grow energy and fp arrays if necessary
  // need to be atom->nmax in length

  if(atom.nmax > nmax) {
    nmax = atom.nmax;
    delete [] rho;
    delete [] fp;

    rho = new MMD_float[nmax];
    fp = new MMD_float[nmax];
  }

  const MMD_float* const x = atom.x;
  MMD_float* const f = atom.f;
  int* type = atom.type;

  const int nlocal = atom.nlocal;

  // zero out density

  for(int i = 0; i < atom.nlocal + atom.nghost; i++) {
    f[i * PAD + 0] = 0;
    f[i * PAD + 1] = 0;
    f[i * PAD + 2] = 0;
  }

  for(MMD_int i = 0; i < nlocal; i++) rho[i] = 0.0;

  // rho = density at each atom
  // loop over neighbors of my atoms

  for(MMD_int i = 0; i < nlocal; i++) {
    int* neighs = &neighbor.neighbors[i * neighbor.maxneighs];
    const int numneigh = neighbor.numneigh[i];
    const MMD_float xtmp = x[i * PAD + 0];
    const MMD_float ytmp = x[i * PAD + 1];
    const MMD_float ztmp = x[i * PAD + 2];
    const int type_i = type[i];
    MMD_float rhoi = 0.0;

    for(MMD_int jj = 0; jj < numneigh; jj++) {
      const MMD_int j = neighs[jj];

      const MMD_float delx = xtmp - x[j * PAD + 0];
      const MMD_float dely = ytmp - x[j * PAD + 1];
      const MMD_float delz = ztmp - x[j * PAD + 2];
      const int type_j = type[j];
      const MMD_float rsq = delx * delx + dely * dely + delz * delz;

      const int type_ij = type_i*ntypes+type_j;

      if(rsq < cutforcesq[type_ij]) {
        MMD_float p = sqrt(rsq) * rdr + 1.0;
        MMD_int m = static_cast<int>(p);
        m = m < nr - 1 ? m : nr - 1;
        p -= m;
        p = p < 1.0 ? p : 1.0;

        rhoi += ((rhor_spline[type_ij*nr_tot + m * 7 + 3] * p + rhor_spline[type_ij*nr_tot + m * 7 + 4]) * p + rhor_spline[type_ij*nr_tot + m * 7 + 5]) * p + rhor_spline[type_ij*nr_tot + m * 7 + 6];

        if(j < nlocal) {
          rho[j] += ((rhor_spline[type_ij*nr_tot + m * 7 + 3] * p + rhor_spline[type_ij*nr_tot + m * 7 + 4]) * p + rhor_spline[type_ij*nr_tot + m * 7 + 5]) * p + rhor_spline[type_ij*nr_tot + m * 7 + 6];
        }
      }
    }

    rho[i] += rhoi;
  }

  // fp = derivative of embedding energy at each atom
  // phi = embedding energy at each atom

  for(MMD_int i = 0; i < nlocal; i++) {
    MMD_float p = 1.0 * rho[i] * rdrho + 1.0;
    MMD_int m = static_cast<int>(p);
    const int type_ii = type[i] * type[i];
    m = MAX(1, MIN(m, nrho - 1));
    p -= m;
    p = MIN(p, 1.0);
    fp[i] = (frho_spline[type_ii*nrho_tot + m * 7 + 0] * p + frho_spline[type_ii*nrho_tot + m * 7 + 1]) * p + frho_spline[type_ii*nrho_tot + m * 7 + 2];

    // printf("fp: %lf %lf %lf %lf %lf %i %lf %lf\n",fp[i],p,frho_spline[type_ij*nrho_tot + m*7+0],frho_spline[type_ij*nrho_tot + m*7+1],frho_spline[type_ij*nrho_tot + m*7+2],m,rdrho,rho[i]);
    if(evflag) {
      evdwl += ((frho_spline[type_ii*nrho_tot + m * 7 + 3] * p + frho_spline[type_ii*nrho_tot + m * 7 + 4]) * p + frho_spline[type_ii*nrho_tot + m * 7 + 5]) * p + frho_spline[type_ii*nrho_tot + m * 7 + 6];
    }
  }

  // communicate derivative of embedding function

  communicate(atom, comm);


  // compute forces on each atom
  // loop over neighbors of my atoms
  for(MMD_int i = 0; i < nlocal; i++) {
    int* neighs = &neighbor.neighbors[i * neighbor.maxneighs];
    const int numneigh = neighbor.numneigh[i];
    const MMD_float xtmp = x[i * PAD + 0];
    const MMD_float ytmp = x[i * PAD + 1];
    const MMD_float ztmp = x[i * PAD + 2];
    const int type_i = type[i];
    MMD_float fx = 0;
    MMD_float fy = 0;
    MMD_float fz = 0;

    for(MMD_int jj = 0; jj < numneigh; jj++) {
      const MMD_int j = neighs[jj];

      const MMD_float delx = xtmp - x[j * PAD + 0];
      const MMD_float dely = ytmp - x[j * PAD + 1];
      const MMD_float delz = ztmp - x[j * PAD + 2];
      const int type_j = type[j];
      const MMD_float rsq = delx * delx + dely * dely + delz * delz;

      const int type_ij = type_i*ntypes+type_j;

      if(rsq < cutforcesq[type_ij]) {
        MMD_float r = sqrt(rsq);
        MMD_float p = r * rdr + 1.0;
        MMD_int m = static_cast<int>(p);
        m = m < nr - 1 ? m : nr - 1;
        p -= m;
        p = p < 1.0 ? p : 1.0;


        // rhoip = derivative of (density at atom j due to atom i)
        // rhojp = derivative of (density at atom i due to atom j)
        // phi = pair potential energy
        // phip = phi'
        // z2 = phi * r
        // z2p = (phi * r)' = (phi' r) + phi
        // psip needs both fp[i] and fp[j] terms since r_ij appears in two
        //   terms of embed eng: Fi(sum rho_ij) and Fj(sum rho_ji)
        //   hence embed' = Fi(sum rho_ij) rhojp + Fj(sum rho_ji) rhoip

        MMD_float rhoip = (rhor_spline[type_ij*nr_tot + m * 7 + 0] * p + rhor_spline[type_ij*nr_tot + m * 7 + 1]) * p + rhor_spline[type_ij*nr_tot + m * 7 + 2];
        MMD_float z2p = (z2r_spline[type_ij*nr_tot + m * 7 + 0] * p + z2r_spline[type_ij*nr_tot + m * 7 + 1]) * p + z2r_spline[type_ij*nr_tot + m * 7 + 2];
        MMD_float z2 = ((z2r_spline[type_ij*nr_tot + m * 7 + 3] * p + z2r_spline[type_ij*nr_tot + m * 7 + 4]) * p + z2r_spline[type_ij*nr_tot + m * 7 + 5]) * p + z2r_spline[type_ij*nr_tot + m * 7 + 6];

        MMD_float recip = 1.0 / r;
        MMD_float phi = z2 * recip;
        MMD_float phip = z2p * recip - phi * recip;
        MMD_float psip = fp[i] * rhoip + fp[j] * rhoip + phip;
        MMD_float fpair = -psip * recip;

        fx += delx * fpair;
        fy += dely * fpair;
        fz += delz * fpair;

        if(j < nlocal) {
          f[j * PAD + 0] -= delx * fpair;
          f[j * PAD + 1] -= dely * fpair;
          f[j * PAD + 2] -= delz * fpair;
        } else fpair *= 0.5;

        if(evflag) {
          virial += delx * delx * fpair + dely * dely * fpair + delz * delz * fpair;
        }

        if(j < nlocal) evdwl += phi;
        else evdwl += 0.5 * phi;
      }
    }

    f[i * PAD + 0] += fx;
    f[i * PAD + 1] += fy;
    f[i * PAD + 2] += fz;
  }

  eng_vdwl = evdwl;
}

/* ---------------------------------------------------------------------- */

void ForceEAM::compute_fullneigh(Atom &atom, Neighbor &neighbor, Comm &comm, int me)
{

  MMD_float evdwl = 0.0;

  // grow energy and fp arrays if necessary
  // need to be atom->nmax in length

  #pragma omp master
  {
    eng_vdwl = 0;
    virial = 0;
    if(atom.nmax > nmax) {
      nmax = atom.nmax;
      rho = new MMD_float[nmax];
      fp = new MMD_float[nmax];
    }
  }

  #pragma omp barrier
  const MMD_float* const x = atom.x;
  MMD_float* const f = atom.f;
  const int* const type = atom.type;
  const int nlocal = atom.nlocal;

  // zero out density

  // rho = density at each atom
  // loop over neighbors of my atoms

  OMPFORSCHEDULE
  for(MMD_int i = 0; i < nlocal; i++) {
    int* neighs = &neighbor.neighbors[i * neighbor.maxneighs];
    const int jnum = neighbor.numneigh[i];
    const MMD_float xtmp = x[i * PAD + 0];
    const MMD_float ytmp = x[i * PAD + 1];
    const MMD_float ztmp = x[i * PAD + 2];
    const int type_i = type[i];
    MMD_float rhoi = 0;

    #pragma ivdep
    for(MMD_int jj = 0; jj < jnum; jj++) {
      const MMD_int j = neighs[jj];

      const MMD_float delx = xtmp - x[j * PAD + 0];
      const MMD_float dely = ytmp - x[j * PAD + 1];
      const MMD_float delz = ztmp - x[j * PAD + 2];
      const int type_j = type[j];
      const MMD_float rsq = delx * delx + dely * dely + delz * delz;

      const int type_ij = type_i*ntypes+type_j;

      if(rsq < cutforcesq[type_ij]) {
        MMD_float p = sqrt(rsq) * rdr + 1.0;
        MMD_int m = static_cast<int>(p);
        m = m < nr - 1 ? m : nr - 1;
        p -= m;
        p = p < 1.0 ? p : 1.0;

        rhoi += ((rhor_spline[type_ij*nr_tot + m * 7 + 3] * p + rhor_spline[type_ij*nr_tot + m * 7 + 4]) * p + rhor_spline[type_ij*nr_tot + m * 7 + 5]) * p + rhor_spline[type_ij*nr_tot + m * 7 + 6];
      }
    }

    const int type_ii = type_i*type_i;
    MMD_float p = 1.0 * rhoi * rdrho + 1.0;
    MMD_int m = static_cast<int>(p);
    m = MAX(1, MIN(m, nrho - 1));
    p -= m;
    p = MIN(p, 1.0);
    fp[i] = (frho_spline[type_ii*nrho_tot + m * 7 + 0] * p + frho_spline[type_ii*nrho_tot + m * 7 + 1]) * p + frho_spline[type_ii*nrho_tot + m * 7 + 2];

    if(evflag) {
      evdwl += ((frho_spline[type_ii*nrho_tot + m * 7 + 3] * p + frho_spline[type_ii*nrho_tot + m * 7 + 4]) * p + frho_spline[type_ii*nrho_tot + m * 7 + 5]) * p + frho_spline[type_ii*nrho_tot + m * 7 + 6];
    }

  }

  // #pragma omp barrier
  // fp = derivative of embedding energy at each atom
  // phi = embedding energy at each atom

  // communicate derivative of embedding function

  #pragma omp master
  {
    communicate(atom, comm);
  }

  #pragma omp barrier

  MMD_float t_virial = 0;
  // compute forces on each atom
  // loop over neighbors of my atoms

  OMPFORSCHEDULE
  for(MMD_int i = 0; i < nlocal; i++) {
    int* neighs = &neighbor.neighbors[i * neighbor.maxneighs];
    const int numneigh = neighbor.numneigh[i];
    const MMD_float xtmp = x[i * PAD + 0];
    const MMD_float ytmp = x[i * PAD + 1];
    const MMD_float ztmp = x[i * PAD + 2];
    const int type_i = type[i];

    MMD_float fx = 0.0;
    MMD_float fy = 0.0;
    MMD_float fz = 0.0;

    #pragma ivdep
    for(MMD_int jj = 0; jj < numneigh; jj++) {
      const MMD_int j = neighs[jj];

      const MMD_float delx = xtmp - x[j * PAD + 0];
      const MMD_float dely = ytmp - x[j * PAD + 1];
      const MMD_float delz = ztmp - x[j * PAD + 2];
      const int type_j = type[j];
      const MMD_float rsq = delx * delx + dely * dely + delz * delz;

      const int type_ij = type_i*ntypes+type_j;

      if(rsq < cutforcesq[type_ij]) {
        MMD_float r = sqrt(rsq);
        MMD_float p = r * rdr + 1.0;
        MMD_int m = static_cast<int>(p);
        m = m < nr - 1 ? m : nr - 1;
        p -= m;
        p = p < 1.0 ? p : 1.0;


        // rhoip = derivative of (density at atom j due to atom i)
        // rhojp = derivative of (density at atom i due to atom j)
        // phi = pair potential energy
        // phip = phi'
        // z2 = phi * r
        // z2p = (phi * r)' = (phi' r) + phi
        // psip needs both fp[i] and fp[j] terms since r_ij appears in two
        //   terms of embed eng: Fi(sum rho_ij) and Fj(sum rho_ji)
        //   hence embed' = Fi(sum rho_ij) rhojp + Fj(sum rho_ji) rhoip

        MMD_float rhoip = (rhor_spline[type_ij*nr_tot + m * 7 + 0] * p + rhor_spline[type_ij*nr_tot + m * 7 + 1]) * p + rhor_spline[type_ij*nr_tot + m * 7 + 2];
        MMD_float z2p = (z2r_spline[type_ij*nr_tot + m * 7 + 0] * p + z2r_spline[type_ij*nr_tot + m * 7 + 1]) * p + z2r_spline[type_ij*nr_tot + m * 7 + 2];
        MMD_float z2 = ((z2r_spline[type_ij*nr_tot + m * 7 + 3] * p + z2r_spline[type_ij*nr_tot + m * 7 + 4]) * p + z2r_spline[type_ij*nr_tot + m * 7 + 5]) * p + z2r_spline[type_ij*nr_tot + m * 7 + 6];

        MMD_float recip = 1.0 / r;
        MMD_float phi = z2 * recip;
        MMD_float phip = z2p * recip - phi * recip;
        MMD_float psip = fp[i] * rhoip + fp[j] * rhoip + phip;
        MMD_float fpair = -psip * recip;

        fx += delx * fpair;
        fy += dely * fpair;
        fz += delz * fpair;
        //  	if(i==0&&j<20)
        //      printf("fpair: %i %i %lf %lf %lf %lf\n",i,j,fpair,delx,dely,delz);
        fpair *= 0.5;

        if(evflag) {
          t_virial += delx * delx * fpair + dely * dely * fpair + delz * delz * fpair;
          evdwl += 0.5 * phi;
        }

      }
    }

    f[i * PAD + 0] = fx;
    f[i * PAD + 1] = fy;
    f[i * PAD + 2] = fz;

  }

  #pragma omp atomic
  virial += t_virial;
  #pragma omp atomic
  eng_vdwl += 2.0 * evdwl;

  #pragma omp barrier
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
   read DYNAMO funcfl file
------------------------------------------------------------------------- */

void ForceEAM::coeff(const char* arg)
{



  // read funcfl file if hasn't already been read
  // store filename in Funcfl data struct


  read_file(arg);
  int n = strlen(arg) + 1;
  funcfl.file = new char[n];

  // set setflag and map only for i,i type pairs
  // set mass of atom type if i = j

  //atom->mass = funcfl.mass;
  cutmax = funcfl.cut;

  for(int i=0; i<ntypes*ntypes; i++)
    cutforcesq[i] = cutmax * cutmax;
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void ForceEAM::init_style()
{
  // convert read-in file(s) to arrays and spline them

  file2array();
  array2spline();
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */



/* ----------------------------------------------------------------------
   read potential values from a DYNAMO single element funcfl file
------------------------------------------------------------------------- */

void ForceEAM::read_file(const char* filename)
{
  Funcfl* file = &funcfl;

  //me = 0;
  FILE* fptr;
  char line[MAXLINE];

  int flag = 0;

  if(me == 0) {
    fptr = fopen(filename, "r");

    if(fptr == NULL) {
      printf("Can't open EAM Potential file: %s\n", filename);
      flag = 1;
    }
  }

  MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if(flag) {
    MPI_Finalize();
    exit(0);
  }

  int tmp;

  if(me == 0) {
    fgets(line, MAXLINE, fptr);
    fgets(line, MAXLINE, fptr);
    sscanf(line, "%d %lg", &tmp, &file->mass);
    fgets(line, MAXLINE, fptr);
    sscanf(line, "%d %lg %d %lg %lg",
           &file->nrho, &file->drho, &file->nr, &file->dr, &file->cut);
  }

  //printf("Read: %lf %i %lf %i %lf %lf\n",file->mass,file->nrho,file->drho,file->nr,file->dr,file->cut);
  MPI_Bcast(&file->mass, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&file->nrho, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&file->drho, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&file->nr, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&file->dr, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&file->cut, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  mass = file->mass;
  file->frho = new MMD_float[file->nrho + 1];
  file->rhor = new MMD_float[file->nr + 1];
  file->zr = new MMD_float[file->nr + 1];

  if(me == 0) grab(fptr, file->nrho, file->frho);

  if(sizeof(MMD_float) == 4)
    MPI_Bcast(file->frho, file->nrho, MPI_FLOAT, 0, MPI_COMM_WORLD);
  else
    MPI_Bcast(file->frho, file->nrho, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if(me == 0) grab(fptr, file->nr, file->zr);

  if(sizeof(MMD_float) == 4)
    MPI_Bcast(file->zr, file->nr, MPI_FLOAT, 0, MPI_COMM_WORLD);
  else
    MPI_Bcast(file->zr, file->nr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if(me == 0) grab(fptr, file->nr, file->rhor);

  if(sizeof(MMD_float) == 4)
    MPI_Bcast(file->rhor, file->nr, MPI_FLOAT, 0, MPI_COMM_WORLD);
  else
    MPI_Bcast(file->rhor, file->nr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  for(int i = file->nrho; i > 0; i--) file->frho[i] = file->frho[i - 1];

  for(int i = file->nr; i > 0; i--) file->rhor[i] = file->rhor[i - 1];

  for(int i = file->nr; i > 0; i--) file->zr[i] = file->zr[i - 1];

  if(me == 0) fclose(fptr);
}

/* ----------------------------------------------------------------------
   convert read-in funcfl potential(s) to standard array format
   interpolate all file values to a single grid and cutoff
------------------------------------------------------------------------- */

void ForceEAM::file2array()
{
  int i, j, k, m, n;
  int ntypes = 1;
  double sixth = 1.0 / 6.0;

  // determine max function params from all active funcfl files
  // active means some element is pointing at it via map

  int active;
  double rmax, rhomax;
  dr = drho = rmax = rhomax = 0.0;

  active = 0;
  Funcfl* file = &funcfl;
  dr = MAX(dr, file->dr);
  drho = MAX(drho, file->drho);
  rmax = MAX(rmax, (file->nr - 1) * file->dr);
  rhomax = MAX(rhomax, (file->nrho - 1) * file->drho);

  // set nr,nrho from cutoff and spacings
  // 0.5 is for round-off in divide

  nr = static_cast<int>(rmax / dr + 0.5);
  nrho = static_cast<int>(rhomax / drho + 0.5);

  // ------------------------------------------------------------------
  // setup frho arrays
  // ------------------------------------------------------------------

  // allocate frho arrays
  // nfrho = # of funcfl files + 1 for zero array

  frho = new MMD_float[nrho + 1];

  // interpolate each file's frho to a single grid and cutoff

  double r, p, cof1, cof2, cof3, cof4;

  n = 0;

  for(m = 1; m <= nrho; m++) {
    r = (m - 1) * drho;
    p = r / file->drho + 1.0;
    k = static_cast<int>(p);
    k = MIN(k, file->nrho - 2);
    k = MAX(k, 2);
    p -= k;
    p = MIN(p, 2.0);
    cof1 = -sixth * p * (p - 1.0) * (p - 2.0);
    cof2 = 0.5 * (p * p - 1.0) * (p - 2.0);
    cof3 = -0.5 * p * (p + 1.0) * (p - 2.0);
    cof4 = sixth * p * (p * p - 1.0);
    frho[m] = cof1 * file->frho[k - 1] + cof2 * file->frho[k] +
              cof3 * file->frho[k + 1] + cof4 * file->frho[k + 2];
  }


  // ------------------------------------------------------------------
  // setup rhor arrays
  // ------------------------------------------------------------------

  // allocate rhor arrays
  // nrhor = # of funcfl files

  rhor = new MMD_float[nr + 1];

  // interpolate each file's rhor to a single grid and cutoff

  for(m = 1; m <= nr; m++) {
    r = (m - 1) * dr;
    p = r / file->dr + 1.0;
    k = static_cast<int>(p);
    k = MIN(k, file->nr - 2);
    k = MAX(k, 2);
    p -= k;
    p = MIN(p, 2.0);
    cof1 = -sixth * p * (p - 1.0) * (p - 2.0);
    cof2 = 0.5 * (p * p - 1.0) * (p - 2.0);
    cof3 = -0.5 * p * (p + 1.0) * (p - 2.0);
    cof4 = sixth * p * (p * p - 1.0);
    rhor[m] = cof1 * file->rhor[k - 1] + cof2 * file->rhor[k] +
              cof3 * file->rhor[k + 1] + cof4 * file->rhor[k + 2];
    //if(m==119)printf("BuildRho: %e %e %e %e %e %e\n",rhor[m],cof1,cof2,cof3,cof4,file->rhor[k]);
  }

  // type2rhor[i][j] = which rhor array (0 to nrhor-1) each type pair maps to
  // for funcfl files, I,J mapping only depends on I
  // OK if map = -1 (non-EAM atom in pair hybrid) b/c type2rhor not used

  // ------------------------------------------------------------------
  // setup z2r arrays
  // ------------------------------------------------------------------

  // allocate z2r arrays
  // nz2r = N*(N+1)/2 where N = # of funcfl files

  z2r = new MMD_float[nr + 1];

  // create a z2r array for each file against other files, only for I >= J
  // interpolate zri and zrj to a single grid and cutoff

  double zri, zrj;

  Funcfl* ifile = &funcfl;
  Funcfl* jfile = &funcfl;

  for(m = 1; m <= nr; m++) {
    r = (m - 1) * dr;

    p = r / ifile->dr + 1.0;
    k = static_cast<int>(p);
    k = MIN(k, ifile->nr - 2);
    k = MAX(k, 2);
    p -= k;
    p = MIN(p, 2.0);
    cof1 = -sixth * p * (p - 1.0) * (p - 2.0);
    cof2 = 0.5 * (p * p - 1.0) * (p - 2.0);
    cof3 = -0.5 * p * (p + 1.0) * (p - 2.0);
    cof4 = sixth * p * (p * p - 1.0);
    zri = cof1 * ifile->zr[k - 1] + cof2 * ifile->zr[k] +
          cof3 * ifile->zr[k + 1] + cof4 * ifile->zr[k + 2];

    p = r / jfile->dr + 1.0;
    k = static_cast<int>(p);
    k = MIN(k, jfile->nr - 2);
    k = MAX(k, 2);
    p -= k;
    p = MIN(p, 2.0);
    cof1 = -sixth * p * (p - 1.0) * (p - 2.0);
    cof2 = 0.5 * (p * p - 1.0) * (p - 2.0);
    cof3 = -0.5 * p * (p + 1.0) * (p - 2.0);
    cof4 = sixth * p * (p * p - 1.0);
    zrj = cof1 * jfile->zr[k - 1] + cof2 * jfile->zr[k] +
          cof3 * jfile->zr[k + 1] + cof4 * jfile->zr[k + 2];

    z2r[m] = 27.2 * 0.529 * zri * zrj;
  }

}

/* ---------------------------------------------------------------------- */

void ForceEAM::array2spline()
{
  rdr = 1.0 / dr;
  rdrho = 1.0 / drho;

  nrho_tot = (nrho + 1) * 7 + 64;
  nr_tot = (nr + 1) * 7 + 64;
  nrho_tot -= nrho_tot%64;
  nr_tot -= nr_tot%64;

  frho_spline = new MMD_float[ntypes * ntypes * nrho_tot];
  rhor_spline = new MMD_float[ntypes * ntypes * nr_tot];
  z2r_spline = new MMD_float[ntypes * ntypes * nr_tot];

  interpolate(nrho, drho, frho, frho_spline);

  interpolate(nr, dr, rhor, rhor_spline);

  interpolate(nr, dr, z2r, z2r_spline);

  // replicate data for multiple types;
  for(int tt = 0 ; tt<ntypes*ntypes; tt++) {
    for(int k = 0; k<nrho_tot; k++)
      frho_spline[tt*nrho_tot + k] = frho_spline[k];
    for(int k = 0; k<nr_tot; k++)
      rhor_spline[tt*nr_tot + k] = rhor_spline[k];
    for(int k = 0; k<nr_tot; k++)
      z2r_spline[tt*nr_tot + k] = z2r_spline[k];
  }
}

/* ---------------------------------------------------------------------- */

void ForceEAM::interpolate(MMD_int n, MMD_float delta, MMD_float* f, MMD_float* spline)
{
  for(int m = 1; m <= n; m++) spline[m * 7 + 6] = f[m];

  spline[1 * 7 + 5] = spline[2 * 7 + 6] - spline[1 * 7 + 6];
  spline[2 * 7 + 5] = 0.5 * (spline[3 * 7 + 6] - spline[1 * 7 + 6]);
  spline[(n - 1) * 7 + 5] = 0.5 * (spline[n * 7 + 6] - spline[(n - 2) * 7 + 6]);
  spline[n * 7 + 5] = spline[n * 7 + 6] - spline[(n - 1) * 7 + 6];

  for(int m = 3; m <= n - 2; m++)
    spline[m * 7 + 5] = ((spline[(m - 2) * 7 + 6] - spline[(m + 2) * 7 + 6]) +
                         8.0 * (spline[(m + 1) * 7 + 6] - spline[(m - 1) * 7 + 6])) / 12.0;

  for(int m = 1; m <= n - 1; m++) {
    spline[m * 7 + 4] = 3.0 * (spline[(m + 1) * 7 + 6] - spline[m * 7 + 6]) -
                        2.0 * spline[m * 7 + 5] - spline[(m + 1) * 7 + 5];
    spline[m * 7 + 3] = spline[m * 7 + 5] + spline[(m + 1) * 7 + 5] -
                        2.0 * (spline[(m + 1) * 7 + 6] - spline[m * 7 + 6]);
  }

  spline[n * 7 + 4] = 0.0;
  spline[n * 7 + 3] = 0.0;

  for(int m = 1; m <= n; m++) {
    spline[m * 7 + 2] = spline[m * 7 + 5] / delta;
    spline[m * 7 + 1] = 2.0 * spline[m * 7 + 4] / delta;
    spline[m * 7 + 0] = 3.0 * spline[m * 7 + 3] / delta;
  }
}

/* ----------------------------------------------------------------------
   grab n values from file fp and put them in list
   values can be several to a line
   only called by proc 0
------------------------------------------------------------------------- */

void ForceEAM::grab(FILE* fptr, MMD_int n, MMD_float* list)
{
  char* ptr;
  char line[MAXLINE];

  int i = 0;

  while(i < n) {
    fgets(line, MAXLINE, fptr);
    ptr = strtok(line, " \t\n\r\f");
    list[i++] = atof(ptr);

    while(ptr = strtok(NULL, " \t\n\r\f")) list[i++] = atof(ptr);
  }
}

/* ---------------------------------------------------------------------- */

MMD_float ForceEAM::single(int i, int j, int itype, int jtype,
                           MMD_float rsq, MMD_float factor_coul, MMD_float factor_lj,
                           MMD_float &fforce)
{
  int m;
  MMD_float r, p, rhoip, rhojp, z2, z2p, recip, phi, phip, psip;
  MMD_float* coeff;

  r = sqrt(rsq);
  p = r * rdr + 1.0;
  m = static_cast<int>(p);
  m = MIN(m, nr - 1);
  p -= m;
  p = MIN(p, 1.0);

  coeff = &rhor_spline[m * 7 + 0];
  rhoip = (coeff[0] * p + coeff[1]) * p + coeff[2];
  coeff = &rhor_spline[m * 7 + 0];
  rhojp = (coeff[0] * p + coeff[1]) * p + coeff[2];
  coeff = &z2r_spline[m * 7 + 0];
  z2p = (coeff[0] * p + coeff[1]) * p + coeff[2];
  z2 = ((coeff[3] * p + coeff[4]) * p + coeff[5]) * p + coeff[6];

  recip = 1.0 / r;
  phi = z2 * recip;
  phip = z2p * recip - phi * recip;
  psip = fp[i] * rhojp + fp[j] * rhoip + phip;
  fforce = -psip * recip;

  return phi;
}

void ForceEAM::communicate(Atom &atom, Comm &comm)
{

  int iswap;
  int pbc_flags[4];
  MMD_float* buf;

  for(iswap = 0; iswap < comm.nswap; iswap++) {

    /* pack buffer */

    pbc_flags[0] = comm.pbc_any[iswap];
    pbc_flags[1] = comm.pbc_flagx[iswap];
    pbc_flags[2] = comm.pbc_flagy[iswap];
    pbc_flags[3] = comm.pbc_flagz[iswap];
    //timer->stamp_extra_start();

    int size = pack_comm(comm.sendnum[iswap], iswap, comm.buf_send, comm.sendlist);
    //timer->stamp_extra_stop(TIME_TEST);


    /* exchange with another proc
       if self, set recv buffer to send buffer */

    if(comm.sendproc[iswap] != me) {
      MPI_Datatype type = (sizeof(MMD_float) == 4) ? MPI_FLOAT : MPI_DOUBLE;
      MPI_Sendrecv(comm.buf_send, comm.comm_send_size[iswap], MPI_FLOAT, comm.sendproc[iswap], 0,
                   comm.buf_recv, comm.comm_recv_size[iswap], MPI_FLOAT, comm.recvproc[iswap], 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      buf = comm.buf_recv;
    } else buf = comm.buf_send;

    /* unpack buffer */

    unpack_comm(comm.recvnum[iswap], comm.firstrecv[iswap], buf);
  }
}
/* ---------------------------------------------------------------------- */

int ForceEAM::pack_comm(int n, int iswap, MMD_float* buf, int** asendlist)
{
  int i, j, m;

  m = 0;

  for(i = 0; i < n; i++) {
    j = asendlist[iswap][i];
    buf[i] = fp[j];
  }

  return 1;
}

/* ---------------------------------------------------------------------- */

void ForceEAM::unpack_comm(int n, int first, MMD_float* buf)
{
  int i, m, last;

  m = 0;
  last = first + n;

  for(i = first; i < last; i++) fp[i] = buf[m++];
}

/* ---------------------------------------------------------------------- */

int ForceEAM::pack_reverse_comm(int n, int first, MMD_float* buf)
{
  int i, m, last;

  m = 0;
  last = first + n;

  for(i = first; i < last; i++) buf[m++] = rho[i];

  return 1;
}

/* ---------------------------------------------------------------------- */

void ForceEAM::unpack_reverse_comm(int n, int* list, MMD_float* buf)
{
  int i, j, m;

  m = 0;

  for(i = 0; i < n; i++) {
    j = list[i];
    rho[j] += buf[m++];
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

MMD_float ForceEAM::memory_usage()
{
  MMD_int bytes = 2 * nmax * sizeof(MMD_float);
  return bytes;
}


void ForceEAM::bounds(char* str, int nmax, int &nlo, int &nhi)
{
  char* ptr = strchr(str, '*');

  if(ptr == NULL) {
    nlo = nhi = atoi(str);
  } else if(strlen(str) == 1) {
    nlo = 1;
    nhi = nmax;
  } else if(ptr == str) {
    nlo = 1;
    nhi = atoi(ptr + 1);
  } else if(strlen(ptr + 1) == 0) {
    nlo = atoi(str);
    nhi = nmax;
  } else {
    nlo = atoi(str);
    nhi = atoi(ptr + 1);
  }

  if(nlo < 1 || nhi > nmax) printf("Numeric index is out of bounds");
}
