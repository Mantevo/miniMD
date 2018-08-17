srcdir=.
include $(srcdir)/Makefile.$(TOOLS)

PREFIX?=.

CPPCHECK?=cppcheck

ROOT=miniMD
TARGETS=
TARGETS+=$(PREFIX)/$(ROOT)_$(TOOLS)
CLEANFILES=$(TARGETS)

OBJ_DIR=$(PREFIX)/Obj_$(TOOLS)

all: $(TARGETS)

#=======================================================================

SRC=ljs.cpp input.cpp integrate.cpp atom.cpp force_lj.cpp neighbor.cpp \
	thermo.cpp comm.cpp timer.cpp output.cpp setup.cpp
INC=ljs.h atom.h force.h neighbor.h thermo.h timer.h comm.h integrate.h threadData.h variant.h openmp.h \
	types.h miniMD_math.h

OBJ_base = $(SRC:.cpp=.o)
OBJ= $(addprefix $(OBJ_DIR)/,$(OBJ_base))
DEP_base = $(SRC:.cpp=.d)
DEP= $(addprefix $(OBJ_DIR)/,$(DEP_base))

LDFLAGS = $(COMMON_LDFLAGS)

$(OBJ): | $(OBJ_DIR)

$(DEP): | $(OBJ_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/%.o: $(srcdir)/%.cpp $(INC) $(srcdir)/Makefile  $(srcdir)/Makefile.$(TOOLS)
$(OBJ_DIR)/%.o: $(srcdir)/%.cpp $(OBJ_DIR)/%.d $(srcdir)/Makefile $(srcdir)/Makefile.$(TOOLS)
	$(CC) -o $@  -c $< $(CCFLAGS) -MD

.PHONY: cppcheck clean

cppcheck: $(SRC) $(INC)
	$(CPPCHECK) --enable=all --inconclusive --template=gcc --language=c++ --std=c++11 $(SRC) $(INC)

-include $(DEPS)

$(PREFIX)/$(ROOT)_$(TOOLS): $(OBJ) $(srcdir)/Makefile $(srcdir)/Makefile.$(TOOLS)
	$(LINK) $(OBJ) $(LINKFLAGS) $(USRLIB) $(SYSLIB)-o $@
	$(SIZE) $@

CLEANFILES += $(OBJ)
CLEANFILES += $(DEP)

clean:
	rm -rf  $(CLEANFILES)

#=======================================================================
