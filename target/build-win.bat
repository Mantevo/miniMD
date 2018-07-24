set INCLUDE=%INCLUDE%c:\Program Files (x86)\OpenMPI_v1.6.1-win32\include;
mpicxx /O2 /arch:AVX -c *.cpp
del ljs.obj
mpicxx /O2 /arch:AVX ljs.cpp /link *.obj
mpirun -np 8 ljs.exe