# Makefile for mpicxx compiler

SHELL = /bin/sh

# Default Options

AVX ?= yes
SIMD ?= yes

#.IGNORE:

# System-specific settings

CC =		mpicxx
CCFLAGS =	-O3 -DMPICH_IGNORE_CXX_SEEK -fopenmp -DNOCHUNK
LINK =		mpicxx 
LINKFLAGS =	-O3 -fopenmp
USRLIB = 	
SYSLIB =	
SIZE =		size

# Check for AVX compile
ifeq ($(AVX), yes)
CCFLAGS += -mavx 
LINKFLAGS += -mavx
endif

ifeq ($(SIMD), yes)
CCFLAGS += -DUSE_SIMD
endif

ifeq ($(ANSI_ALIAS), yes)
CCFLAGS += -ansi-alias 
LINKFLAGS += -ansi-alias
endif

#Check for PAD
ifeq ($(PAD4), yes)
CCFLAGS += -DPAD4 
LINKFLAGS += -DPAD4 
endif

#Check for single precision
ifeq ($(SP), yes)
CCFLAGS += -DPRECISION=1 
LINKFLAGS += -DPRECISION=1 
else
CCFLAGS += -DPRECISION=2
LINKFLAGS += -DPRECISION=2
endif

#Check if debug on
ifeq ($(DEBUG), yes)
CCFLAGS += -g  
LINKFLAGS += -g 
endif

#Use LIBRT for precision timers
ifeq ($(LIBRT),yes)
CCFLAGS += -DPREC_TIMER
USRLIB += -lrt
endif

# Link rule

$(EXE):	$(OBJ)
	$(LINK) $(LINKFLAGS) $(OBJ) $(USRLIB) $(SYSLIB) -o $(EXE)
	$(SIZE) $(EXE)

# Compilation rules

.cpp.o:
	$(CC) $(CCFLAGS) -E $*.cpp > $*.2.cpp
	$(CC) $(CCFLAGS) -c $*.2.cpp -o $*.o

# Individual dependencies

$(OBJ): $(INC)
