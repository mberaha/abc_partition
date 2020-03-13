#ifndef DISTANCE_HPP
#define DISTANCE_HPP

#include "include_arma.hpp"
#include <string>
#include "wasserstein.hpp"
#include "ot/sinkhorn.hpp"
#include "graph.hpp"

class Distance {
public:
    virtual ~Distance() {}

    virtual std::tuple<arma::uvec, double> compute(
        const arma::mat &atoms_x, const arma::mat &atoms_y) = 0;
};


class UniformDiscreteWassersteinDistance: public Distance {
protected:
    int max_iter;
    int p = 1;

public:
    UniformDiscreteWassersteinDistance(int max_iter=100): max_iter(max_iter) {}

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
    double entropic_eps;
    double threshold;
    int max_iter;
    int p = 1;

    bool init_done = false;
    arma::vec w_in, w_out;
    int n_in, n_out;
    arma::mat cost_mat;
    arma::mat transport;
    arma::uvec perm;
    double dist;

    bool greedy = false;

public:

UniformSinkhorn(
    double entropic_eps=0.1, double threshold=1e-4,
    int max_iter=100, bool greedy = false):
        entropic_eps(entropic_eps), threshold(threshold),
        max_iter(max_iter), greedy(greedy) {}

std::tuple<arma::uvec, double> compute(
    const arma::mat &real_data, const arma::mat &synth_data);

void call() {
    if (greedy)
        greenkhorn(w_in, w_out, cost_mat, entropic_eps, threshold, max_iter,
             p, &transport, &dist, true);
    else
        sinkhorn(w_in, w_out, cost_mat, entropic_eps, threshold, max_iter,
             p, &transport, &dist);
}

void compute_cost(const arma::mat &real_data, const arma::mat &synth_data);

void init(const arma::mat &real_data, const arma::mat &synth_data);
};


class GraphSinkhorn {
protected:
    double entropic_eps;
    double threshold;
    int max_iter;
    int p = 1;

    bool init_done = false;
    arma::vec w_in, w_out;
    int n_in, n_out;
    arma::mat cost_mat;
    arma::mat transport;
    arma::uvec perm;
    double dist;

    bool greedy = false;

public:
    GraphSinkhorn(
        double entropic_eps = 0.1, double threshold = 1e-4,
        int max_iter = 100, bool greedy = false):
            entropic_eps(entropic_eps), threshold(threshold),
            max_iter(max_iter), greedy(greedy) {}

    std::tuple<arma::uvec, double> compute(
        const std::vector<Graph> &real_data, 
        const std::vector<Graph> &synth_data);

    void call()
    {
        if (greedy)
            greenkhorn(w_in, w_out, cost_mat, entropic_eps, threshold, max_iter,
                       p, &transport, &dist, true);
        else
            sinkhorn(w_in, w_out, cost_mat, entropic_eps, threshold, max_iter,
                     p, &transport, &dist);
    }

    void compute_cost(
        const std::vector<Graph> &real_data,
        const std::vector<Graph> &synth_data);

    void init(const std::vector<Graph> &real_data,
              const std::vector<Graph> &synth_data);
};

#endif
