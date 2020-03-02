# Default Makefile: just call 'make -f Makefile.default'
SHELL = /bin/sh

#Compiler and Linker
CC =    mpicxx
LINK =    mpicxx

# MiniMD Files

SRC =	ljs.cpp input.cpp integrate.cpp atom.cpp force_lj.cpp neighbor.cpp thermo.cpp comm.cpp timer.cpp output.cpp setup.cpp force_eam.cpp  
INC = ljs.h atom.h force.h neighbor.h thermo.h timer.h comm.h integrate.h threadData.h variant.h types.h force_eam.h force_lj.h openmp.h

	
# Definitions

ROOT =	miniMD
EXE =	$(ROOT)
OBJ =	$(SRC:.cpp=.o) $(KOKKOS_SRC:.cpp=.o) 

CCFLAGS = -O3 -DMPICH_IGNORE_CXX_SEEK -DDEVICE=1  -I./ -g -mavx -fopenmp -DUSE_SIMD -DNOCHUNK
LINKFLAGS = -O3 -g -mavx -fopenmp 
USRLIB =  
SYSLIB =  

#First copy all files, then call the same makefile again with the actual build target
#Can't be done in single pass, because the actual compilation rule for .o files, need the 
#copied cpp files to be in place. 

copy: 
	@if [ ! -d Obj_default ]; then mkdir Obj_default; fi
	@cp -p $(SRC) $(INC) Obj_default
	@cp Makefile.default Obj_default/Makefile
	@cd Obj_default; $(MAKE) all
  
all: $(OBJ)
	$(LINK) $(OBJ) $(LINKFLAGS) $(USRLIB) $(SYSLIB) -o ../$(EXE)
  
# Compilation rules

.cpp.o:
	$(CC) $(CCFLAGS) -E $*.cpp > $*.2.cpp
	$(CC) $(CCFLAGS) -c $*.2.cpp -o $*.o

# Individual dependencies

$(OBJ): $(INC)

# Clean
clean:
	rm -r Obj_*
	
# Test

scope=0
input=lj
halfneigh=0
path=""
test:
	bash run_tests ${scope} ${input} ${halfneigh} ${path} 
  
