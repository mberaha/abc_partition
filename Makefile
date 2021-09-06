ROOT_DIR := .
SRC_DIR := $(ROOT_DIR)/src
SPIKES_DIR := $(SRC_DIR)/spikes
OT_DIR := $(SRC_DIR)/ot
CGAL_DIR := ~/CGAL-5.0.2

CXX = g++
CFLAGS = \
	-std=c++1y \
	-MMD \
	-I${HOMEBREW_PREFIX}/include \
    -I${ARMADILLO_DIR}/include \
	-I${BOOST_DIR}/include \
	-I$(OT_DIR) \
	-I$(CGAL_DIR)/include \
	-I$(ROOT_DIR)/lib/stats/include \
	-I$(ROOT_DIR)/lib/gcem/include \
	-fPIC -DCGAL_DISABLE_ROUNDING_MATH_CHECK=ON \
	-O3  -ftree-vectorize -funroll-loops -fopenmp

LDLIBS = -lstdc++  -larmadillo -lblas -llapack -L${ARMADILLO_DIR}/lib -L${HOMEBREW_PREFIX}/lib
LDFLAGS = -std=c++14 -D_REENTRANT -DARMA_DONT_USE_WRAPPER -DARMA_NO_DEBUG \
		  -DARMA_USE_OPENMP

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

all: $(SPIKES_EXECS) generate_pybind

generate_lib: $(OBJS)
	g++ -shared $(OBJS) -o libabc.so $(LDLIBS)

generate_pybind: $(OBJS)
	$(CXX) -shared $(CFLAGS) -I$(SRC_DIR)/lib/carma/include/ \
		 `${PYTHON3} -m pybind11 --includes` \
		python_exports.cpp -o abcpp`${PYTHON3}-config --extension-suffix` \
		$(OBJS) $(LDLIBS)


$(SPIKES_EXECS): %.out: %.o $(OBJS)
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJS) $< $(LDLIBS)

$(SPIKES_OBJS): %.o: %.cpp
		$(CXX) $(CFLAGS) -c $< -o $@

%.o : %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

# -include $(OBJS:%.o=%.d)

clean:
	rm $(OBJS) $(SPIKES_OBJS) $(OBJS:%.o=%.d)
