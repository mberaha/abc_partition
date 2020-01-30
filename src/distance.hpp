#ifndef DISTANCE_HPP
#define DISTANCE_HPP

#include <armadillo>
#include <string>
#include "wasserstein.hpp"

class Distance {
public:
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

#endif
