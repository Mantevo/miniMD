#include<force_lj.h>

template<class ExecutionSpace>
class ComputeLJ {
public:
  static bool compute(const ForceLJ& force,  MMD_int half_neigh, MMD_int ghost_newton) { return false; }
};

#ifdef KOKKOS_ENABLE_OPENMP
template<>
class ComputeLJ<Kokkos::OpenMP> {
public:
  static bool compute(const ForceLJ& force,  MMD_int half_neigh, MMD_int ghost_newton) {
    if(half_neigh!=0) return false;
    if(force.evflag) return false;

    const int nlocal = force.nlocal;
    const int nall =   force.nall;
    const int ntypes = force.ntypes;

    const MMD_float* const x = force.x.data();
    MMD_float* const f =       force.f.data();
    const int* const type =    force.type.data();

    const MMD_int* const neighbors =    force.neighbors.data();
    const MMD_int* const numneigh =     force.numneigh.data();
    const MMD_int maxneighs =           force.neighbors.extent(1);
    const MMD_float* const cutforcesq = ntypes>MAX_STACK_TYPES?force.cutforcesq.data():force.cutforcesq_s;
    const MMD_float* const sigma6 =     ntypes>MAX_STACK_TYPES?force.sigma6.data():force.sigma6_s;
    const MMD_float* const epsilon =    ntypes>MAX_STACK_TYPES?force.epsilon.data():force.epsilon_s;

    #pragma omp parallel for simd
    for(int i = 0; i < nlocal; i++) {
      f[i * PAD + 0] = 0.0;
      f[i * PAD + 1] = 0.0;
      f[i * PAD + 2] = 0.0;
    }

    // loop over all neighbors of my atoms
    // store force on atom i

    #pragma omp parallel for
    for(int i = 0; i < nlocal; i++) {
      const int* const neighs = &neighbors[i * maxneighs];
      const int numneighs = numneigh[i];
      const MMD_float xtmp = x[i * PAD + 0];
      const MMD_float ytmp = x[i * PAD + 1];
      const MMD_float ztmp = x[i * PAD + 2];
      const int type_i = type[i];
      MMD_float fix = 0;
      MMD_float fiy = 0;
      MMD_float fiz = 0;

      #pragma omp simd reduction (+: fix,fiy,fiz)
      for(int k = 0; k < numneighs; k++) {
        const int j = neighs[k];
        const MMD_float delx = xtmp - x[j * PAD + 0];
        const MMD_float dely = ytmp - x[j * PAD + 1];
        const MMD_float delz = ztmp - x[j * PAD + 2];
        const int type_j = type[j];
        const MMD_float rsq = delx * delx + dely * dely + delz * delz;

        int type_ij = type_i*ntypes+type_j;
        if(rsq < cutforcesq[type_ij]) {
          const MMD_float sr2 = 1.0 / rsq;
          const MMD_float sr6 = sr2 * sr2 * sr2 * sigma6[type_ij];
          const MMD_float force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon[type_ij];
          fix += delx * force;
          fiy += dely * force;
          fiz += delz * force;
        }

      }

      f[i * PAD + 0] += fix;
      f[i * PAD + 1] += fiy;
      f[i * PAD + 2] += fiz;
    }
    return true;
  }

};
#endif

bool compute_lj(const ForceLJ& force,  MMD_int half_neigh, MMD_int ghost_newton) {
  return ComputeLJ<Kokkos::DefaultExecutionSpace>::compute(force,half_neigh,ghost_newton);
}

