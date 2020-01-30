#ifndef WASSERSTEIN_HPP
#define WASSERSTEIN_HPP

#include <armadillo>
#include <tuple>

#include "./ot/emd.hpp"


inline double lp_dist(arma::vec x, arma::vec y, double p=2.0) {
    return arma::accu(arma::square(x -y));
}

std::tuple<arma::umat, double, int> d_wasserstein(
        arma::mat atoms_x, arma::mat atoms_s, double p,
        int max_iter=100000);

#endif  // WASSERSTEIN_HPP
