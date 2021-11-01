ROOT_DIR := .
SRC_DIR := $(ROOT_DIR)/src
SPIKES_DIR := $(SRC_DIR)/spikes
OT_DIR := $(SRC_DIR)/ot

UNAME_S := $(shell uname -s)

CXX = g++
CFLAGS = \
	-std=c++1y \
	-MMD \
    -I${ARMADILLO_DIR}/include \
	-I${BOOST_DIR}/include \
	-I$(OT_DIR) \
	-I$(CGAL_DIR)/include \
	-I$(ROOT_DIR)/lib/stats/include \
	-I$(ROOT_DIR)/lib/gcem/include \
	-fPIC -DCGAL_DISABLE_ROUNDING_MATH_CHECK=ON \
	-O3 -ftree-vectorize -funroll-loops \
	-Wno-reorder -Wno-sign-compare 

LDLIBS = -lstdc++ -larmadillo -lblas -llapack -L${ARMADILLO_DIR}/lib `${PYTHON3}-config --libs`
LDFLAGS = -std=c++1y -D_REENTRANT -DARMA_DONT_USE_WRAPPER -DARMA_NO_DEBUG -DUSE_CGAL `${PYTHON3}-config --ldflags`

ifeq (${UNAME_S},Darwin)
	CXX = clang++
	CFLAGS += -I${HOMEBREW_PREFIX}/include  -DUSE_CGAL
	LDLIBS += -L${HOMEBREW_PREFIX}/lib 
else 
	CFLAGS += -fopenmp
endif

OUR_SRCS_T = $(wildcard $(SRC_DIR)/*.cpp)
OUR_SRCS_TT = $(filter-out $(SRC_DIR)/RcppExports.cpp $(SRC_DIR)/graph.cpp $(SRC_DIR)/abc_py_class.cpp, $(OUR_SRCS_T))
OUR_SRCS = $(filter-out $(SRC_DIR)/rcpp_functions.cpp, $(OUR_SRCS_TT))

OT_SRCS = $(wildcard $(OT_DIR)/*.cpp)
SPIKES_SRCS = $(wildcard $(SPIKES_DIR)/*.cpp)

SRCS = $(OUR_SRCS) $(OT_SRCS)
OBJS = $(subst .cpp,.o, $(SRCS))

SPIKES_OBJS = $(subst .cpp,.o, $(SPIKES_SRCS))
SPIKES_EXECS = $(subst .cpp,.out, $(SPIKES_SRCS))

info:
	@echo " Info..."
	@echo " SRC_DIR = $(SRC_DIR)"
	@echo " SPIKES_DIR = $(SPIKES_DIR)"
	@echo " SOURCES = $(SRCS)"
	@echo " OT_SRCS = $(OT_SRCS)"
	@echo "uname -s: $(UNAME_S)"


all: $(SPIKES_EXECS) generate_pybind

generate_lib: $(OBJS)
	g++ -shared $(OBJS) -o libabc.so $(LDLIBS)

generate_pybind: $(OBJS)
	$(CXX) -shared $(CFLAGS) -I$(SRC_DIR)/lib/carma/include/ \
		 `${PYTHON3} -m pybind11 --includes` \
		python_exports.cpp -o abcpp`${PYTHON3}-config --extension-suffix` \
		$(OBJS) $(LDLIBS)

run_gnk: run_gnk.o $(OBJS)
	$(CXX) $(CFLAGS) $(LDFLAGS) -o run_gnk.out $(OBJS) $< $(LDLIBS)
	

$(SPIKES_EXECS): %.out: %.o $(OBJS)
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJS) $< $(LDLIBS)

$(SPIKES_OBJS): %.o: %.cpp
		$(CXX) $(CFLAGS) -c $< -o $@

%.o : %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

# -include $(OBJS:%.o=%.d)

clean:
	rm $(OBJS) $(SPIKES_OBJS) $(OBJS:%.o=%.d)  run_gnk.out run_gnk.o
