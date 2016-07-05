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


#ifndef FORCEEAM_H
#define FORCEEAM_H

#include "stdio.h"
#include "atom.h"
#include "neighbor.h"
#include "threadData.h"
#include "types.h"
#include "mpi.h"
#include "comm.h"
#include "force.h"

class ForceEAM : Force
{
  public:

    // public variables so USER-ATC package can access them

    MMD_float cutmax;

    // potentials as array data

    MMD_int nrho, nr;
    MMD_int nrho_tot, nr_tot;
    MMD_float* frho, *rhor, *z2r;

    // potentials in spline form used for force computation

    MMD_float dr, rdr, drho, rdrho;
    MMD_float* rhor_spline, *frho_spline, *z2r_spline;

    ForceEAM(int ntypes_);
    virtual ~ForceEAM();
    virtual void compute(Atom &atom, Neighbor &neighbor, Comm &comm, int me);
    virtual void coeff(const char*);
    virtual void setup();
    void init_style();
    MMD_float single(MMD_int, MMD_int, MMD_int, MMD_int, MMD_float, MMD_float, MMD_float, MMD_float &);

    virtual MMD_int pack_comm(int n, int iswap, MMD_float* buf, MMD_int** asendlist);
    virtual void unpack_comm(int n, int first, MMD_float* buf);
    MMD_int pack_reverse_comm(MMD_int, MMD_int, MMD_float*);
    void unpack_reverse_comm(MMD_int, MMD_int*, MMD_float*);
    MMD_float memory_usage();

  protected:
    void compute_halfneigh(Atom &atom, Neighbor &neighbor, Comm &comm, int me);
    void compute_fullneigh(Atom &atom, Neighbor &neighbor, Comm &comm, int me);

    // per-atom arrays

    MMD_float* rho, *fp;

    MMD_int nmax;

    // potentials as file data

    MMD_int* map;                   // which element each atom type maps to

    struct Funcfl {
      char* file;
      MMD_int nrho, nr;
      double drho, dr, cut, mass;
      MMD_float* frho, *rhor, *zr;
    };
    Funcfl funcfl;

    void array2spline();
    void interpolate(MMD_int n, MMD_float delta, MMD_float* f, MMD_float* spline);
    void grab(FILE*, MMD_int, MMD_float*);

    virtual void read_file(const char*);
    virtual void file2array();

    void bounds(char* str, int nmax, int &nlo, int &nhi);

    void communicate(Atom &atom, Comm &comm);
};



#endif
