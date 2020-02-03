#include <armadillo>
#include <tuple>
#include "wasserstein.hpp"
#include "distance.hpp"
#include "distributions.hpp"

void update_part_PY_univ(
    arma::vec &temp_part, arma::vec part, arma::mat param, arma::mat &tparam,
    double theta, double sigma, double m0, double k0, double a0, double b0);


void update_part_PY_multi(
    arma::vec &temp_part, arma::vec part, std::vector<arma::vec> mean,
    std::vector<arma::vec> &tmean, std::vector<arma::mat> prec,
    std::vector<arma::mat> &tprec, double theta, double sigma,
    const arma::vec &m0, double k0, double df,
    const arma::mat& prior_prec_chol);


std::tuple<arma::vec, arma::mat, double> ABC_MCMC_univ(
    arma::vec data, int nrep, double theta, double sigma, double m0,
    double k0, double a0, double b0, double eps0, int p,
    std::string dist="wasserstein");


std::tuple<arma::vec, arma::mat, double> ABC_MCMC_multi(
    arma::mat data, int nrep, double theta, double sigma,
    const arma::vec &m0, double k0, double df, const arma::mat& prior_prec_chol,
    double eps0, int p, std::string dist="wasserstein");
