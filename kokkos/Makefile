#Set the path to MPI
MPI_PATH = 
#Use MPI
HAVE_MPI = yes

#Set the path to Kokkos
KOKKOS_PATH =${HOME}/Kokkos/kokkos
#Set the Devices to compile for
KOKKOS_DEVICES=OpenMP
#Set the Architecture to compiler for
KOKKOS_ARCH=SNB
#Set third party library usage
KOKKOS_USE_TPL=

CXXFLAGS = -O3 -g 
LINKFLAGS = 
LIB =

ifeq ($(HAVE_MPI), yes)
CXX = mpicxx
else
CXX = g++
override CXXFLAGS += -I./MPI-Stubs
LIB += MPI-Stubs/libmpi_stubs.a
endif

LINK = ${CXX}
EXE = miniMD

#CXXFLAGS += -DTEST_LAMBDA_BYCOPY
#CXXFLAGS += -DTEST_LAMBDA_BYPTR

#Check for single precision
ifeq ($(SP), yes)
override CXXFLAGS += -DPRECISION=1 
LINKFLAGS += -DPRECISION=1 
else
CXXFLAGS += -DPRECISION=2
LINKFLAGS += -DPRECISION=2
endif

#Check if debug on
ifeq ($(DEBUG), yes)
override CXXFLAGS += -g -G 
LINKFLAGS += -g 
endif


#Use LIBRT for precision timers
ifeq ($(LIBRT),yes)
override CXXFLAGS += -DPREC_TIMER
KOKKOS_USE_TPL += ,librt
endif

#Check for PAD
ifeq ($(PAD), 3)
override CXXFLAGS += -DPAD3 
endif
ifeq ($(PAD), 4)
override CXXFLAGS += -DPAD4 
endif

#Check for use of explicit SIMD pragma
ifeq ($(SIMD), yes)
override CXXFLAGS += -DUSE_SIMD
endif

#Check for request of ANSI_ALIAS
ifeq ($(ANSI_ALIAS), yes)
override CXXFLAGS += -ansi-alias 
endif


default: all

MAKEFILE_PATH := $(subst Makefile,,$(abspath $(lastword $(MAKEFILE_LIST))))
include $(KOKKOS_PATH)/Makefile.kokkos

SRC = $(wildcard $(MAKEFILE_PATH)*.cpp)
HEADERS = $(wildcard $(MAKEFILE_PATH)*.hpp)

vpath %.cpp $(sort $(dir $(SRC)))
$(warning "MakefilePath: $(MAKEFILE_PATH) SRC: $(SRC) HEADERS: $(HEADERS)")
DEPFLAGS = -M

OBJ = $(notdir $(SRC:.cpp=.o))

all: $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LDFLAGS) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean
	rm *.o *.d libkokkos.a *.cuda *.host

EXTRA_FLAGS = $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS)
# Compilation rules

%.o:%.cpp
	$(CXX) $(CXXFLAGS) $(EXTRA_FLAGS) -c $< -o $(notdir $@)


# Test
scope=0
input=lj
halfneigh=0
path=""

test:
	bash run_tests ${scope} ${input} ${halfneigh} ${path}
