#ifndef ABC_PY_CLASS
#define ABC_PY_CLASS

#include <vector> 
#include "distance.hpp"
#include "distributions.hpp"
#include "utils.hpp"
#include "kernels.hpp"


template<typename data_t, typename param_t, class Kernel>
class AbcPy {
 protected:
     Distance<data_t>* d;
     arma::imat part_results;
     arma::vec dist_results;
     double curr_dist;
     arma::ivec part;
     arma::ivec temp_part;
     arma::ivec tvec;
     arma::ivec uniq_temp;
     std::vector<data_t> data;
     std::vector<data_t> data_synt;

     std::vector<param_t> param;
     std::vector<param_t> tparam;

     // used only if log=true
     std::vector<std::vector<param_t>> param_log;

     Kernel kernel;

     // params
     double theta, sigma;
     int n_data;
     double eps0;

     int n_unique;
     arma::uvec sort_indices;
     arma::vec t_diff;
     double lEsum = 0;
     double lEsum2 = 0;
     double eps;
     double meanEps, meanEps2;

     bool log=false;
     int num_accepted = 0;

 public:
     ~AbcPy() {delete d;}

     AbcPy() {}

     AbcPy(std::vector<data_t> data, std::vector<param_t> inits,
           double theta, double sigma,
           double eps0, std::string distance, Kernel kernel,
           int max_iter = 100, double entropic_eps = 0.1, double threshold = 1e-4);
    
     void set_log() {log = true;}

     void updateUrn();

     void updateParams();

     void generateSyntData();

     void saveCurrParam();

     void step();

     double run(int nrep);

     arma::vec get_dists() const { return dist_results; }

     arma::imat get_parts() const { return part_results; }

     std::vector<std::vector<param_t>> get_params_log() const { return param_log; }
};

#include "abc_class_imp.hpp"

using UnivAbcPy = AbcPy<double, arma::vec, UnivGaussianKernel>;

using MultiAbcPy = AbcPy<arma::vec, mvnorm_param, MultiGaussianKernel>;

using TimeSeriesAbcPy = AbcPy<TimeSeries, arma::vec, TimeSeriesKernel>;

#ifdef USE_R

    using GraphAbcPy = AbcPy<Graph, arma::vec, GraphKernel>;

#endif

#endif
