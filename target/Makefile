srcdir=.
include $(srcdir)/Makefile.$(TOOLS)

PREFIX?=.
OBJ_DIR=$(PREFIX)/Obj_$(TOOLS)

CPPCHECK?=cppcheck
BEAR?=bear
RUN-CLANG-TIDY?=run-clang-tidy.py

ROOT=miniMD
TARGETS=
TARGETS+=$(PREFIX)/$(ROOT)_$(TOOLS)
CLEANFILES=$(TARGETS) $(OBJ_DIR)/compile_commands.json

all: $(TARGETS)

#=======================================================================

SRC=ljs.cpp input.cpp integrate.cpp atom.cpp force_lj.cpp neighbor.cpp \
	thermo.cpp comm.cpp timer.cpp output.cpp setup.cpp
INC=ljs.h atom.h force.h neighbor.h thermo.h timer.h comm.h integrate.h threadData.h variant.h openmp.h \
	types.h miniMD_math.h util.h

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

.PHONY: clean cppcheck run-clang-tidy

cppcheck: $(SRC) $(INC)
	$(CPPCHECK) --enable=all --inconclusive --template=gcc --language=c++ --std=c++11 $(SRC) $(INC)

$(OBJ_DIR)/compile_commands.json: $(SRC) $(INC) $(srcdir)/Makefile $(srcdir)/Makefile.$(TOOLS)
	$(BEAR) --cdb $@ $(MAKE)

run-clang-tidy: $(OBJ_DIR)/compile_commands.json $(srcdir)/Makefile $(srcdir)/Makefile.$(TOOLS)
	$(RUN-CLANG-TIDY) -p $(OBJ_DIR) -checks '*'

-include $(DEPS)

$(PREFIX)/$(ROOT)_$(TOOLS): $(OBJ) $(srcdir)/Makefile $(srcdir)/Makefile.$(TOOLS)
	$(LINK) $(OBJ) $(LINKFLAGS) $(USRLIB) $(SYSLIB)-o $@
	$(SIZE) $@

CLEANFILES += $(OBJ)
CLEANFILES += $(DEP)

clean:
	rm -rf  $(CLEANFILES)

#=======================================================================
