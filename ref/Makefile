# Multiple-machine Makefile

SHELL = /bin/sh

# Files

SRC =	ljs.cpp input.cpp integrate.cpp atom.cpp force_lj.cpp force_eam.cpp neighbor.cpp \
	thermo.cpp comm.cpp timer.cpp output.cpp setup.cpp
INC =	ljs.h atom.h force.h neighbor.h thermo.h timer.h comm.h integrate.h threadData.h variant.h openmp.h \
	force_lj.h force_eam.h types.h

# Definitions

ROOT =	miniMD
EXE =	$(ROOT)_$@
OBJ =	$(SRC:.cpp=.o)

# Help

help:
	@echo 'Type "make target {Options}" where target is one of:'
	@echo '      openmpi     (using OpenMPI)'
	@echo '      intel       (using intelMPI)'
	@echo '      cray        (using CrayCompiler for CPUs)'
	@echo '      cray_intel  (using Cray Wrapper for Intel compiler for CPUs)'
	@echo 'Options (not all of which apply for each target):'
	@echo '      KNC=yes     (compile for XeonPhi)'
	@echo '      SP=yes      (compile for single precision [32bit floats])'
	@echo '      DEBUG=yes   (compile debug mode)'
	@echo '      AVX=yes     (compile for avx [DEFAULT])'
	@echo '      SIMD=yes    (use "pragma simd" [DEFAULT])'
	@echo '      RED_PREC=yes   (use reduced precision math intrinsics)'
	@echo '      ANSI_ALIAS=yes (compile with ansi-alias flag)'
	@echo '      GSUNROLL=yes   (compile with flags for XeonPhi which unroll gather/scatter operations [DEFAULT])'
	@echo '      LIBRT=yes   (use librt for timing [more precise])'
	@echo '      PAD=[3,4]   (pad data to 3 or 4 elements [default 3 = no padding])'
	@echo '(Good) Examples:'
	@echo '    Modern CPUs (Intel Sandy Bridge, AMD Interlagos or newer):'
	@echo '      make openmpi -j 32'
	@echo '    Intel Xeon Phi:'
	@echo '      make intel -j 32 KNC=yes ANSI_ALIAS=yes SIMD=no'

# Targets

openmpi:
	@if [ ! -d Obj_$@ ]; then mkdir Obj_$@; fi
	@cp -p $(SRC) $(INC) Obj_$@
	@cp Makefile.$@ Obj_$@/Makefile
	@cd Obj_$@; \
	$(MAKE)  "OBJ = $(OBJ)" "INC = $(INC)" "EXE = ../$(EXE)" ../$(EXE)
#	@if [ -d Obj_$@ ]; then cd Obj_$@; rm $(SRC) $(INC) Makefile*; fi

cray:
	@if [ ! -d Obj_$@ ]; then mkdir Obj_$@; fi
	@cp -p $(SRC) $(INC) Obj_$@
	@cp Makefile.$@ Obj_$@/Makefile
	@cd Obj_$@; \
	$(MAKE)  "OBJ = $(OBJ)" "INC = $(INC)" "EXE = ../$(EXE)" ../$(EXE)

cray_intel:
	@if [ ! -d Obj_$@ ]; then mkdir Obj_$@; fi
	@cp -p $(SRC) $(INC) Obj_$@
	@cp Makefile.$@ Obj_$@/Makefile
	@cd Obj_$@; \
	$(MAKE)  "OBJ = $(OBJ)" "INC = $(INC)" "EXE = ../$(EXE)" ../$(EXE)

intel:
	@if [ ! -d Obj_$@ ]; then mkdir Obj_$@; fi
	@cp -p $(SRC) $(INC) Obj_$@
	@cp Makefile.$@ Obj_$@/Makefile
	@cd Obj_$@; \
	$(MAKE)  "OBJ = $(OBJ)" "INC = $(INC)" "EXE = ../$(EXE)" ../$(EXE)
#       @if [ -d Obj_$@ ]; then cd Obj_$@; rm $(SRC) $(INC) Makefile*; fi


# Clean

clean:
	rm -r Obj_*

clean_openmpi:
	rm -r Obj_openmpi

clean_intel:
	rm -r Obj_intel

clean_cray:
	rm -r Obj_cray
	
clean_cray_intel:
	rm -r Obj_cray_intel

# Test

scope=0
input=lj
halfneigh=0
path=""

test:
	bash run_tests ${scope} ${input} ${halfneigh} ${path}   
