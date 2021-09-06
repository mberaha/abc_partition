#ifndef KERNELS_HPP
#define KERNELS_HPP

#include <cmath>
#include <random>
#include "distributions.hpp"
#include "utils.hpp"
#include "stats.hpp"
#include "time_series.hpp"

using mvnorm_param = std::tuple<arma::vec, arma::mat>;

struct gandk_param {
    arma::vec a;
    arma::vec b;
    arma::vec g;
    arma::vec k;
};

class UnivGaussianKernel
{
protected:
    double m0, a0, b0, k0;

public:

    UnivGaussianKernel() {}
    ~UnivGaussianKernel() {}

    UnivGaussianKernel(double m0, double a0, double b0, double k0);

    arma::vec sample_prior();

    std::vector<double> generate_dataset(
        arma::ivec temp_part, std::vector<arma::vec> tparam);

    std::vector<arma::vec> make_default_init();
};

class MultiGaussianKernel {
protected:
    double df, k0;
    arma::vec m0;
    arma::mat prior_prec_chol;

public:
    MultiGaussianKernel() {}
    ~MultiGaussianKernel() {}

    MultiGaussianKernel(arma::vec m0, arma::mat prior_prec_chol, double df,
                        double k0);

    std::tuple<arma::vec, arma::mat> sample_prior();

    std::vector<arma::vec> generate_dataset(
        arma::ivec temp_part, std::vector<mvnorm_param> tparam);

    std::vector<mvnorm_param> make_default_init();
};

class MultiGandKKernel {
protected:
    arma::vec mean_a;
    arma::vec var_a;
    arma::vec shape_b;
    arma::vec rate_b;
    arma::vec mean_g;
    arma::vec var_g;
    arma::vec shape_k;
    arma::vec rate_k;

    double rho;
    arma::mat cov_chol;

public:
  MultiGandKKernel() {}
  ~MultiGandKKernel() {}

  MultiGandKKernel(double rho);

  MultiGandKKernel(double rho, arma::vec mean_a, arma::vec var_a,  arma::vec shape_b, arma::vec rate_b, 
                   arma::vec mean_g, arma::vec var_g, arma::vec shape_k, arma::vec rate_k);


  gandk_param sample_prior();

  std::vector<arma::vec> generate_dataset(
      arma::ivec temp_part, std::vector<gandk_param>);

  std::vector<gandk_param> make_default_init();

  arma::vec rand_from_param(gandk_param param);

};


class TimeSeriesKernel {
protected:
    int num_steps;

    // normal distributions
    double mu_mean, mu_sd, beta_mean, beta_sd;
    // exponential_distributions
    double xi_rate, omega_sq_rate, lambda_rate;

    std::mt19937_64 rng{23042020};

public:
    TimeSeriesKernel() {}
    ~TimeSeriesKernel() {}

    TimeSeriesKernel(int num_steps): num_steps(num_steps) {}

    TimeSeriesKernel(double mu_mean, double mu_sd, double beta_mean,
                     double beta_sd, double xi_rate, double omega_sq_rate,
                     double lambda_rate, int num_steps);

    arma::vec sample_prior();

    std::vector<TimeSeries> generate_dataset(
        arma::ivec temp_part, std::vector<arma::vec> tparam);

    TimeSeries generate_single(
        double mu, double beta, double xi, double omega_sq, double lambda);

    std::vector<arma::vec> make_default_init();
};


#ifdef USE_R
#include "Rcpp.h"
#include "graph.hpp"
#include "distributions.hpp"

class GraphKernel {
protected:
    GraphSimulator simulator;

    int n_nodes;
    arma::vec m0;
    arma::mat prior_var_chol;

public:
    GraphKernel() {}
    ~GraphKernel() {}

    GraphKernel(int n_nodes, arma::vec m0, arma::mat prior_var_chol);

    arma::vec sample_prior();

    std::vector<Graph> generate_dataset(
        arma::ivec temp_part, std::vector<arma::vec> tparam);

    std::vector<arma::vec> make_default_init();
};

#endif




#endif