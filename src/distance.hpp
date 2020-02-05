#ifndef DISTANCE_HPP
#define DISTANCE_HPP

#include <armadillo>
#include <string>
#include "wasserstein.hpp"
#include "ot/sinkhorn.hpp"

class Distance {
public:
    virtual ~Distance() {}

    virtual std::tuple<arma::uvec, double> compute(
        const arma::mat &atoms_x, const arma::mat &atoms_y) = 0;
};


class UniformDiscreteWassersteinDistance: public Distance {
protected:
    int max_iter = 1000;
    double p = 1.0;

public:
    std::tuple<arma::uvec, double> compute(
            const arma::mat &real_data, const arma::mat &synth_data);
};

// NB!! Assumes atoms_x already sorted
class SortingDistance1d: public Distance {
protected:
    double p = 1.0;

public:
    std::tuple<arma::uvec, double> compute(
        const arma::mat &real_data, const arma::mat &synth_data);
};


class UniformSinkhorn: public Distance {
protected:
    double eps = 0.1;
    double threshold = 1e-1;
    int max_iter = 1000;
    int p = 1;

    bool init_done = false;
    arma::vec w_in, w_out;
    int n_in, n_out;
    arma::mat cost_mat;
    arma::mat transport;
    arma::uvec perm;

public:
std::tuple<arma::uvec, double> compute(
    const arma::mat &real_data, const arma::mat &synth_data);

void compute_cost(const arma::mat &real_data, const arma::mat &synth_data);

void init(const arma::mat &real_data, const arma::mat &synth_data);
};

#endif
