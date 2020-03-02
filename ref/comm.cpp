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

#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"
#include "comm.h"
#include "openmp.h"

#define BUFFACTOR 1.5
#define BUFMIN 1000
#define BUFEXTRA 100
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

Comm::Comm()
{
  maxsend = BUFMIN;
  buf_send = (MMD_float*) malloc((maxsend + BUFMIN) * sizeof(MMD_float));
  maxrecv = BUFMIN;
  buf_recv = (MMD_float*) malloc(maxrecv * sizeof(MMD_float));
  check_safeexchange = 0;
  do_safeexchange = 0;
  maxthreads = 0;
  maxnlocal = 0;
}

Comm::~Comm() {}

/* setup spatial-decomposition communication patterns */

int Comm::setup(MMD_float cutneigh, Atom &atom)
{
  int i;
  int nprocs;
  int periods[3];
  MMD_float prd[3];
  int myloc[3];
  MPI_Comm cartesian;
  MMD_float lo, hi;
  int ineed, idim, nbox;

  prd[0] = atom.box.xprd;
  prd[1] = atom.box.yprd;
  prd[2] = atom.box.zprd;

  /* setup 3-d grid of procs */

  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  MMD_float area[3];

  area[0] = prd[0] * prd[1];
  area[1] = prd[0] * prd[2];
  area[2] = prd[1] * prd[2];

  MMD_float bestsurf = 2.0 * (area[0] + area[1] + area[2]);

  // loop thru all possible factorizations of nprocs
  // surf = surface area of a proc sub-domain
  // for 2d, insure ipz = 1

  int ipx, ipy, ipz, nremain;
  MMD_float surf;

  ipx = 1;

  while(ipx <= nprocs) {
    if(nprocs % ipx == 0) {
      nremain = nprocs / ipx;
      ipy = 1;

      while(ipy <= nremain) {
        if(nremain % ipy == 0) {
          ipz = nremain / ipy;
          surf = area[0] / ipx / ipy + area[1] / ipx / ipz + area[2] / ipy / ipz;

          if(surf < bestsurf) {
            bestsurf = surf;
            procgrid[0] = ipx;
            procgrid[1] = ipy;
            procgrid[2] = ipz;
          }
        }

        ipy++;
      }
    }

    ipx++;
  }

  if(procgrid[0]*procgrid[1]*procgrid[2] != nprocs) {
    if(me == 0) printf("ERROR: Bad grid of processors\n");

    return 1;
  }

  /* determine where I am and my neighboring procs in 3d grid of procs */

  int reorder = 0;
  periods[0] = periods[1] = periods[2] = 1;

  MPI_Cart_create(MPI_COMM_WORLD, 3, procgrid, periods, reorder, &cartesian);
  MPI_Cart_get(cartesian, 3, procgrid, periods, myloc);
  MPI_Cart_shift(cartesian, 0, 1, &procneigh[0][0], &procneigh[0][1]);
  MPI_Cart_shift(cartesian, 1, 1, &procneigh[1][0], &procneigh[1][1]);
  MPI_Cart_shift(cartesian, 2, 1, &procneigh[2][0], &procneigh[2][1]);

  /* lo/hi = my local box bounds */

  atom.box.xlo = myloc[0] * prd[0] / procgrid[0];
  atom.box.xhi = (myloc[0] + 1) * prd[0] / procgrid[0];
  atom.box.ylo = myloc[1] * prd[1] / procgrid[1];
  atom.box.yhi = (myloc[1] + 1) * prd[1] / procgrid[1];
  atom.box.zlo = myloc[2] * prd[2] / procgrid[2];
  atom.box.zhi = (myloc[2] + 1) * prd[2] / procgrid[2];

  /* need = # of boxes I need atoms from in each dimension */

  need[0] = static_cast<int>(cutneigh * procgrid[0] / prd[0] + 1);
  need[1] = static_cast<int>(cutneigh * procgrid[1] / prd[1] + 1);
  need[2] = static_cast<int>(cutneigh * procgrid[2] / prd[2] + 1);

  /* alloc comm memory */

  int maxswap = 2 * (need[0] + need[1] + need[2]);

  slablo = (MMD_float*) malloc(maxswap * sizeof(MMD_float));
  slabhi = (MMD_float*) malloc(maxswap * sizeof(MMD_float));
  pbc_any = (int*) malloc(maxswap * sizeof(int));
  pbc_flagx = (int*) malloc(maxswap * sizeof(int));
  pbc_flagy = (int*) malloc(maxswap * sizeof(int));
  pbc_flagz = (int*) malloc(maxswap * sizeof(int));
  sendproc = (int*) malloc(maxswap * sizeof(int));
  recvproc = (int*) malloc(maxswap * sizeof(int));
  sendproc_exc = (int*) malloc(maxswap * sizeof(int));
  recvproc_exc = (int*) malloc(maxswap * sizeof(int));
  sendnum = (int*) malloc(maxswap * sizeof(int));
  recvnum = (int*) malloc(maxswap * sizeof(int));
  comm_send_size = (int*) malloc(maxswap * sizeof(int));
  comm_recv_size = (int*) malloc(maxswap * sizeof(int));
  reverse_send_size = (int*) malloc(maxswap * sizeof(int));
  reverse_recv_size = (int*) malloc(maxswap * sizeof(int));
  int iswap = 0;

  for(int idim = 0; idim < 3; idim++)
    for(int i = 1; i <= need[idim]; i++, iswap += 2) {
      MPI_Cart_shift(cartesian, idim, i, &sendproc_exc[iswap], &sendproc_exc[iswap + 1]);
      MPI_Cart_shift(cartesian, idim, i, &recvproc_exc[iswap + 1], &recvproc_exc[iswap]);
    }

  MPI_Comm_free(&cartesian);

  firstrecv = (int*) malloc(maxswap * sizeof(int));
  maxsendlist = (int*) malloc(maxswap * sizeof(int));

  for(i = 0; i < maxswap; i++) maxsendlist[i] = BUFMIN;

  sendlist = (int**) malloc(maxswap * sizeof(int*));

  for(i = 0; i < maxswap; i++)
    sendlist[i] = (int*) malloc(BUFMIN * sizeof(int));

  /* setup 4 parameters for each exchange: (spart,rpart,slablo,slabhi)
     sendproc(nswap) = proc to send to at each swap
     recvproc(nswap) = proc to recv from at each swap
     slablo/slabhi(nswap) = slab boundaries (in correct dimension) of atoms
                            to send at each swap
     1st part of if statement is sending to the west/south/down
     2nd part of if statement is sending to the east/north/up
     nbox = atoms I send originated in this box */

  /* set commflag if atoms are being exchanged across a box boundary
     commflag(idim,nswap) =  0 -> not across a boundary
                          =  1 -> add box-length to position when sending
                          = -1 -> subtract box-length from pos when sending */

  nswap = 0;

  for(idim = 0; idim < 3; idim++) {
    for(ineed = 0; ineed < 2 * need[idim]; ineed++) {
      pbc_any[nswap] = 0;
      pbc_flagx[nswap] = 0;
      pbc_flagy[nswap] = 0;
      pbc_flagz[nswap] = 0;

      if(ineed % 2 == 0) {
        sendproc[nswap] = procneigh[idim][0];
        recvproc[nswap] = procneigh[idim][1];
        nbox = myloc[idim] + ineed / 2;
        lo = nbox * prd[idim] / procgrid[idim];

        if(idim == 0) hi = atom.box.xlo + cutneigh;

        if(idim == 1) hi = atom.box.ylo + cutneigh;

        if(idim == 2) hi = atom.box.zlo + cutneigh;

        hi = MIN(hi, (nbox + 1) * prd[idim] / procgrid[idim]);

        if(myloc[idim] == 0) {
          pbc_any[nswap] = 1;

          if(idim == 0) pbc_flagx[nswap] = 1;

          if(idim == 1) pbc_flagy[nswap] = 1;

          if(idim == 2) pbc_flagz[nswap] = 1;
        }
      } else {
        sendproc[nswap] = procneigh[idim][1];
        recvproc[nswap] = procneigh[idim][0];
        nbox = myloc[idim] - ineed / 2;
        hi = (nbox + 1) * prd[idim] / procgrid[idim];

        if(idim == 0) lo = atom.box.xhi - cutneigh;

        if(idim == 1) lo = atom.box.yhi - cutneigh;

        if(idim == 2) lo = atom.box.zhi - cutneigh;

        lo = MAX(lo, nbox * prd[idim] / procgrid[idim]);

        if(myloc[idim] == procgrid[idim] - 1) {
          pbc_any[nswap] = 1;

          if(idim == 0) pbc_flagx[nswap] = -1;

          if(idim == 1) pbc_flagy[nswap] = -1;

          if(idim == 2) pbc_flagz[nswap] = -1;
        }
      }

      slablo[nswap] = lo;
      slabhi[nswap] = hi;
      nswap++;
    }
  }

  return 0;
}

/* communication of atom info every timestep */

void Comm::communicate(Atom &atom)
{

  int iswap;
  int pbc_flags[4];
  MMD_float* buf;

  for(iswap = 0; iswap < nswap; iswap++) {

    /* pack buffer */

    pbc_flags[0] = pbc_any[iswap];
    pbc_flags[1] = pbc_flagx[iswap];
    pbc_flags[2] = pbc_flagy[iswap];
    pbc_flags[3] = pbc_flagz[iswap];

    //#pragma omp barrier
    atom.pack_comm(sendnum[iswap], sendlist[iswap], buf_send, pbc_flags);

    //#pragma omp barrier

    /* exchange with another proc
       if self, set recv buffer to send buffer */

    if(sendproc[iswap] != me) {
      #pragma omp master
      {
        MPI_Datatype type = (sizeof(MMD_float) == 4) ? MPI_FLOAT : MPI_DOUBLE;
        MPI_Sendrecv(buf_send, comm_send_size[iswap], type, sendproc[iswap], 0,
                     buf_recv, comm_recv_size[iswap], type, recvproc[iswap], 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      buf = buf_recv;
    } else buf = buf_send;

    #pragma omp barrier
    /* unpack buffer */

    atom.unpack_comm(recvnum[iswap], firstrecv[iswap], buf);
    //#pragma omp barrier
  }
}

/* reverse communication of atom info every timestep */

void Comm::reverse_communicate(Atom &atom)
{
  int iswap;
  MMD_float* buf;

  for(iswap = nswap - 1; iswap >= 0; iswap--) {

    /* pack buffer */

    // #pragma omp barrier
    atom.pack_reverse(recvnum[iswap], firstrecv[iswap], buf_send);

    // #pragma omp barrier
    /* exchange with another proc
       if self, set recv buffer to send buffer */

    if(sendproc[iswap] != me) {

      #pragma omp master
      {
        MPI_Datatype type = (sizeof(MMD_float) == 4) ? MPI_FLOAT : MPI_DOUBLE;
        MPI_Sendrecv(buf_send, reverse_send_size[iswap], type, recvproc[iswap], 0,
                     buf_recv, reverse_recv_size[iswap], type, sendproc[iswap], 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      buf = buf_recv;
    } else buf = buf_send;

    /* unpack buffer */

    #pragma omp barrier
    atom.unpack_reverse(sendnum[iswap], sendlist[iswap], buf);
    // #pragma omp barrier
  }
}

/* exchange:
   move atoms to correct proc boxes
   send out atoms that have left my box, receive ones entering my box
   this routine called before every reneighboring
   atoms exchanged with all 6 stencil neighbors
*/

void Comm::exchange(Atom &atom)
{
  if(do_safeexchange)
    return exchange_all(atom);

  int i, m, n, idim, nsend, nrecv, nrecv1, nrecv2, nlocal;
  MMD_float lo, hi, value;
  MMD_float* x;

  /* enforce PBC */

  atom.pbc();

  /* loop over dimensions */
  int tid = omp_get_thread_num();

  for(idim = 0; idim < 3; idim++) {

    /* only exchange if more than one proc in this dimension */

    if(procgrid[idim] == 1) continue;

    /* fill buffer with atoms leaving my box
       when atom is deleted, fill it in with last atom */

    i = nsend = 0;

    if(idim == 0) {
      lo = atom.box.xlo;
      hi = atom.box.xhi;
    } else if(idim == 1) {
      lo = atom.box.ylo;
      hi = atom.box.yhi;
    } else {
      lo = atom.box.zlo;
      hi = atom.box.zhi;
    }

    x = atom.x;

    nlocal = atom.nlocal;

    #pragma omp master
    {
      if(nlocal > maxnlocal) {
        send_flag = new int[nlocal];
        maxnlocal = nlocal;
      }

      if(maxthreads < threads->omp_num_threads) {
        maxthreads = threads->omp_num_threads;
        nsend_thread = new int [maxthreads];
        nrecv_thread = new int [maxthreads];
        nholes_thread = new int [maxthreads];
        maxsend_thread = new int [maxthreads];
        exc_sendlist_thread = new int*[maxthreads];

        for(int i = 0; i < maxthreads; i++) {
          maxsend_thread[i] = maxsend;
          exc_sendlist_thread[i] = (int*) malloc(maxsend * sizeof(int));
        }
      }
    }

    #pragma omp barrier

    nsend = 0;
    #pragma omp for

    for(int i = 0; i < threads->omp_num_threads; i++) {
      nsend_thread[i] = 0;
      nholes_thread[i] = 0;
    }

    #pragma omp for
    for(int i = 0; i < nlocal; i++) {
      if(x[i * PAD + idim] < lo || x[i * PAD + idim] >= hi) {
        if(nsend >= maxsend_thread[tid])  {
          maxsend_thread[tid] = nsend + 100;
          exc_sendlist_thread[tid] = (int*) realloc(exc_sendlist_thread[tid], (nsend + 100) * sizeof(int));
        }

        exc_sendlist_thread[tid][nsend++] = i;
        send_flag[i] = 0;
      } else
        send_flag[i] = 1;
    }

    nsend_thread[tid] = nsend;

    #pragma omp barrier

    #pragma omp master
    {
      int total_nsend = 0;

      for(int i = 0; i < threads->omp_num_threads; i++) {
        total_nsend += nsend_thread[i];
        nsend_thread[i] = total_nsend;
      }

      if(total_nsend * 7 > maxsend) growsend(total_nsend * 7);
    }

    #pragma omp barrier

    int total_nsend = nsend_thread[threads->omp_num_threads - 1];
    int nholes = 0;

    for(int i = 0; i < nsend; i++)
      if(exc_sendlist_thread[tid][i] < nlocal - total_nsend)
        nholes++;

    nholes_thread[tid] = nholes;
    #pragma omp barrier

    #pragma omp master
    {
      int total_nholes = 0;

      for(int i = 0; i < threads->omp_num_threads; i++) {
        total_nholes += nholes_thread[i];
        nholes_thread[i] = total_nholes;
      }
    }
    #pragma omp barrier

    int j = nlocal;
    int holes = 0;

    while(holes < nholes_thread[tid]) {
      j--;

      if(send_flag[j]) holes++;
    }


    for(int k = 0; k < nsend; k++) {
      atom.pack_exchange(exc_sendlist_thread[tid][k], &buf_send[(k + nsend_thread[tid] - nsend) * 7]);

      if(exc_sendlist_thread[tid][k] < nlocal - total_nsend) {
        while(!send_flag[j]) j++;

        atom.copy(j++, exc_sendlist_thread[tid][k]);
      }
    }

    nsend *= 7;
    #pragma omp barrier
    #pragma omp master
    {
      atom.nlocal = nlocal - total_nsend;
      nsend = total_nsend * 7;

      /* send/recv atoms in both directions
         only if neighboring procs are different */

      MPI_Sendrecv(&nsend, 1, MPI_INT, procneigh[idim][0], 0,
                   &nrecv1, 1, MPI_INT, procneigh[idim][1], 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      nrecv = nrecv1;

      if(procgrid[idim] > 2) {
        MPI_Sendrecv(&nsend, 1, MPI_INT, procneigh[idim][1], 0,
                     &nrecv2, 1, MPI_INT, procneigh[idim][0], 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        nrecv += nrecv2;
      }

      if(nrecv > maxrecv) growrecv(nrecv);

      MPI_Datatype type = (sizeof(MMD_float) == 4) ? MPI_FLOAT : MPI_DOUBLE;
      MPI_Sendrecv(buf_send, nsend, type, procneigh[idim][0], 0,
                   buf_recv, nrecv1, type, procneigh[idim][1], 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      if(procgrid[idim] > 2) {
        MPI_Sendrecv(buf_send, nsend, type, procneigh[idim][1], 0,
                     buf_recv+nrecv1, nrecv2, type, procneigh[idim][0], 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }

      nrecv_atoms = nrecv / 7;

      for(int i = 0; i < threads->omp_num_threads; i++)
        nrecv_thread[i] = 0;

    }
    /* check incoming atoms to see if they are in my box
       if they are, add to my list */

    #pragma omp barrier

    nrecv = 0;

    #pragma omp for
    for(int i = 0; i < nrecv_atoms; i++) {
      value = buf_recv[i * 7 + idim];

      if(value >= lo && value < hi)
        nrecv++;
    }

    nrecv_thread[tid] = nrecv;
    nlocal = atom.nlocal;
    #pragma omp barrier

    #pragma omp master
    {
      int total_nrecv = 0;

      for(int i = 0; i < threads->omp_num_threads; i++) {
        total_nrecv += nrecv_thread[i];
        nrecv_thread[i] = total_nrecv;
      }

      atom.nlocal += total_nrecv;
    }
    #pragma omp barrier

    int copyinpos = nlocal + nrecv_thread[tid] - nrecv;

    #pragma omp for
    for(int i = 0; i < nrecv_atoms; i++) {
      value = buf_recv[i * 7 + idim];

      if(value >= lo && value < hi)
        atom.unpack_exchange(copyinpos++, &buf_recv[i * 7]);
    }

    // #pragma omp barrier

  }
}

void Comm::exchange_all(Atom &atom)
{
  int i, m, n, idim, nsend, nrecv, nrecv1, nrecv2, nlocal;
  MMD_float lo, hi, value;
  MMD_float* x;

  /* enforce PBC */

  atom.pbc();

  /* loop over dimensions */
  int iswap = 0;

  for(idim = 0; idim < 3; idim++) {

    /* only exchange if more than one proc in this dimension */

    if(procgrid[idim] == 1) {
      iswap += 2 * need[idim];
      continue;
    }

    /* fill buffer with atoms leaving my box
    *        when atom is deleted, fill it in with last atom */

    i = nsend = 0;

    if(idim == 0) {
      lo = atom.box.xlo;
      hi = atom.box.xhi;
    } else if(idim == 1) {
      lo = atom.box.ylo;
      hi = atom.box.yhi;
    } else {
      lo = atom.box.zlo;
      hi = atom.box.zhi;
    }

    x = atom.x;

    nlocal = atom.nlocal;

    while(i < nlocal) {
      if(x[i * PAD + idim] < lo || x[i * PAD + idim] >= hi) {
        if(nsend > maxsend) growsend(nsend);

        nsend += atom.pack_exchange(i, &buf_send[nsend]);
        atom.copy(nlocal - 1, i);
        nlocal--;
      } else i++;
    }

    atom.nlocal = nlocal;

    /* send/recv atoms in both directions
    *        only if neighboring procs are different */
    for(int ineed = 0; ineed < 2 * need[idim]; ineed += 1) {
      if(ineed < procgrid[idim] - 1) {
        MPI_Sendrecv(&nsend, 1, MPI_INT, sendproc_exc[iswap], 0,
                     &nrecv, 1, MPI_INT, recvproc_exc[iswap], 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if(nrecv > maxrecv) growrecv(nrecv);

        MPI_Datatype type = (sizeof(MMD_float) == 4) ? MPI_FLOAT : MPI_DOUBLE;
        MPI_Sendrecv(buf_send, nsend, type, sendproc_exc[iswap], 0,
                     buf_recv, nrecv, type, recvproc_exc[iswap], 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* check incoming atoms to see if they are in my box
        *        if they are, add to my list */

        n = atom.nlocal;
        m = 0;

        while(m < nrecv) {
          value = buf_recv[m + idim];

          if(value >= lo && value < hi)
            m += atom.unpack_exchange(n++, &buf_recv[m]);
          else m += atom.skip_exchange(&buf_recv[m]);
        }

        atom.nlocal = n;
      }

      iswap += 1;

    }
  }
}

/* borders:
   make lists of nearby atoms to send to neighboring procs at every timestep
   one list is created for every swap that will be made
   as list is made, actually do swaps
   this does equivalent of a communicate (so don't need to explicitly
     call communicate routine on reneighboring timestep)
   this routine is called before every reneighboring
*/

void Comm::borders(Atom &atom)
{
  int i, m, n, iswap, idim, ineed, nsend, nrecv, nall, nfirst, nlast;
  MMD_float lo, hi;
  int pbc_flags[4];
  MMD_float* x;

  /* erase all ghost atoms */

  atom.nghost = 0;

  /* do swaps over all 3 dimensions */

  iswap = 0;

  int tid = omp_get_thread_num();

    #pragma omp master
    {
      if(atom.nlocal > maxnlocal) {
        send_flag = new int[atom.nlocal];
        maxnlocal = atom.nlocal;
      }

      if(maxthreads < threads->omp_num_threads) {
        maxthreads = threads->omp_num_threads;
        nsend_thread = new int [maxthreads];
        nrecv_thread = new int [maxthreads];
        nholes_thread = new int [maxthreads];
        maxsend_thread = new int [maxthreads];
        exc_sendlist_thread = new int*[maxthreads];

        for(int i = 0; i < maxthreads; i++) {
          maxsend_thread[i] = maxsend;
          exc_sendlist_thread[i] = (int*) malloc(maxsend * sizeof(int));
        }
      }
    }

  for(idim = 0; idim < 3; idim++) {
    nlast = 0;

    for(ineed = 0; ineed < 2 * need[idim]; ineed++) {

      // find atoms within slab boundaries lo/hi using <= and >=
      // check atoms between nfirst and nlast
      //   for first swaps in a dim, check owned and ghost
      //   for later swaps in a dim, only check newly arrived ghosts
      // store sent atom indices in list for use in future timesteps

      lo = slablo[iswap];
      hi = slabhi[iswap];
      pbc_flags[0] = pbc_any[iswap];
      pbc_flags[1] = pbc_flagx[iswap];
      pbc_flags[2] = pbc_flagy[iswap];
      pbc_flags[3] = pbc_flagz[iswap];

      x = atom.x;

      if(ineed % 2 == 0) {
        nfirst = nlast;
        nlast = atom.nlocal + atom.nghost;
      }

      #pragma omp for

      for(int i = 0; i < threads->omp_num_threads; i++) {
        nsend_thread[i] = 0;
      }

      //#pragma omp barrier
      nsend = 0;
      m = 0;

      #pragma omp for
      for(int i = nfirst; i < nlast; i++) {
        if(x[i * PAD + idim] >= lo && x[i * PAD + idim] <= hi) {
          if(nsend >= maxsend_thread[tid])  {
            maxsend_thread[tid] = nsend + 100;
            exc_sendlist_thread[tid] = (int*) realloc(exc_sendlist_thread[tid], (nsend + 100) * sizeof(int));
          }

          exc_sendlist_thread[tid][nsend++] = i;
        }
      }

      nsend_thread[tid] = nsend;

      #pragma omp barrier

      #pragma omp master
      {
        int total_nsend = 0;

        for(int i = 0; i < threads->omp_num_threads; i++) {
          total_nsend += nsend_thread[i];
          nsend_thread[i] = total_nsend;
        }

        if(total_nsend > maxsendlist[iswap]) growlist(iswap, total_nsend);

        if(total_nsend * 4 > maxsend) growsend(total_nsend * 4);
      }
      #pragma omp barrier

      for(int k = 0; k < nsend; k++) {
        atom.pack_border(exc_sendlist_thread[tid][k], &buf_send[(k + nsend_thread[tid] - nsend) * 4], pbc_flags);
        sendlist[iswap][k + nsend_thread[tid] - nsend] = exc_sendlist_thread[tid][k];
      }

      #pragma omp barrier


      /* swap atoms with other proc
      put incoming ghosts at end of my atom arrays
      if swapping with self, simply copy, no messages */

      #pragma omp master
      {
        nsend = nsend_thread[threads->omp_num_threads - 1];

        if(sendproc[iswap] != me) {
          MPI_Sendrecv(&nsend, 1, MPI_INT, sendproc[iswap], 0,
                       &nrecv, 1, MPI_INT, recvproc[iswap], 0,
                       MPI_COMM_WORLD, MPI_STATUS_IGNORE);

          if(nrecv * atom.border_size > maxrecv) growrecv(nrecv * atom.border_size);

          MPI_Datatype type = (sizeof(MMD_float) == 4) ? MPI_FLOAT : MPI_DOUBLE;
          MPI_Sendrecv(buf_send, nsend * atom.border_size, type, sendproc[iswap], 0,
                       buf_recv, nrecv * atom.border_size, type, recvproc[iswap], 0,
                       MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          buf = buf_recv;
        } else {
          nrecv = nsend;
          buf = buf_send;
        }

        nrecv_atoms = nrecv;
      }
      /* unpack buffer */

      #pragma omp barrier
      n = atom.nlocal + atom.nghost;
      nrecv = nrecv_atoms;

      #pragma omp for
      for(int i = 0; i < nrecv; i++)
        atom.unpack_border(n + i, &buf[i * 4]);

      // #pragma omp barrier

      /* set all pointers & counters */

      #pragma omp master
      {
        sendnum[iswap] = nsend;
        recvnum[iswap] = nrecv;
        comm_send_size[iswap] = nsend * atom.comm_size;
        comm_recv_size[iswap] = nrecv * atom.comm_size;
        reverse_send_size[iswap] = nrecv * atom.reverse_size;
        reverse_recv_size[iswap] = nsend * atom.reverse_size;
        firstrecv[iswap] = atom.nlocal + atom.nghost;
        atom.nghost += nrecv;
      }
      #pragma omp barrier
      iswap++;
    }
  }

  /* insure buffers are large enough for reverse comm */

  int max1, max2;
  max1 = max2 = 0;

  for(iswap = 0; iswap < nswap; iswap++) {
    max1 = MAX(max1, reverse_send_size[iswap]);
    max2 = MAX(max2, reverse_recv_size[iswap]);
  }

  if(max1 > maxsend) growsend(max1);

  if(max2 > maxrecv) growrecv(max2);
}

/* realloc the size of the send buffer as needed with BUFFACTOR & BUFEXTRA */

void Comm::growsend(int n)
{
  maxsend = static_cast<int>(BUFFACTOR * n);
  buf_send = (MMD_float*) realloc(buf_send, (maxsend + BUFEXTRA) * sizeof(MMD_float));
}

/* free/malloc the size of the recv buffer as needed with BUFFACTOR */

void Comm::growrecv(int n)
{
  maxrecv = static_cast<int>(BUFFACTOR * n);
  free(buf_recv);
  buf_recv = (MMD_float*) malloc(maxrecv * sizeof(MMD_float));
}

/* realloc the size of the iswap sendlist as needed with BUFFACTOR */

void Comm::growlist(int iswap, int n)
{
  maxsendlist[iswap] = static_cast<int>(BUFFACTOR * n);
  sendlist[iswap] =
    (int*) realloc(sendlist[iswap], maxsendlist[iswap] * sizeof(int));
}
