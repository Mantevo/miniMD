mkdir minimd-submission
cd minimd-submission
git clone https://github.com/kokkos/kokkos
cd kokkos
git checkout 2.7.24
cd ..
mkdir submission
./kokkos/scripts/snapshot.py --small ${PWD}/kokkos submission
rm -rf submission/kokkos/core/src/eti/*/*.cpp

git clone https://github.com/mantevo/miniMD
cp -r miniMD/mpi-spec/* submission


