#include <armadillo>
#include <tuple>
#include "wasserstein.hpp"
#include "distance.hpp"

void update_part_PY(arma::vec &temp_part,
                    arma::vec part,
                    arma::mat param,
                    arma::mat &tparam,
                    double theta,
                    double sigma,
                    double m0,
                    double k0,
                    double a0,
                    double b0);



std::tuple<arma::vec, arma::mat, double> ABC_MCMC(
    arma::vec data, int nrep, double theta, double sigma, double m0,
    double k0, double a0, double b0, double eps0, int p,
    std::string dist="wasserstein");
