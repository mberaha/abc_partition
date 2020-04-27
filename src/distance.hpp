#ifndef DISTANCE_HPP
#define DISTANCE_HPP

#include "include_arma.hpp"
#include <string>
#include "time_series.hpp"
#include "wasserstein.hpp"
#include "ot/sinkhorn.hpp"
#include "utils.hpp"
#include "./ot/emd.hpp"



template <typename data_t>
class Distance
{
public:
    virtual ~Distance() {}

    virtual std::tuple<arma::uvec, double> compute(
        const std::vector<data_t> &atoms_x,
        const std::vector<data_t> &atoms_y) = 0;
};

template<typename data_t>
class UniformWasserstein : public Distance<data_t>
{
protected:
    int max_iter;
    int p = 1;

public:
    UniformWasserstein(int max_iter = 100) : max_iter(max_iter) {}

    std::tuple<arma::uvec, double> compute(
        const std::vector<data_t> &atoms_x,
        const std::vector<data_t> &atoms_y);
};

// NB!! Assumes atoms_x already sorted
// class SortingDistance1d: public Distance<double> {
// protected:
//     double p = 1.0;

// public:
//     std::tuple<arma::uvec, double> compute(
//         const std::vector<double> &real_data,
//         const std::vector<double> &synth_data);
// };


template <typename data_t>
class UniformSinkhorn: public Distance<data_t> {
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
        const std::vector<data_t> &real_data,
        const std::vector<data_t> &synth_data);

    void call() {
        if (greedy)
            greenkhorn(w_in, w_out, cost_mat, entropic_eps, threshold, max_iter,
                p, &transport, &dist, true);
        else
            sinkhorn(w_in, w_out, cost_mat, entropic_eps, threshold, max_iter,
                p, &transport, &dist);
    }

    void init(const std::vector<data_t> &real_data, 
              const std::vector<data_t> &synth_data);
};

template <typename data_t>
std::tuple<arma::uvec, double> UniformWasserstein<data_t>::compute(
    const std::vector<data_t> &real_data,
    const std::vector<data_t> &synth_data)
{
    int n1 = real_data.size();
    int n2 = synth_data.size();
    arma::vec weights_x = arma::ones<arma::vec>(n1) / n1;
    arma::vec weights_s = arma::ones<arma::vec>(n2) / n2;
    arma::vec alpha = arma::zeros<arma::vec>(n1);
    arma::vec beta = arma::zeros<arma::vec>(n2);
    arma::mat t_mat(n1, n2);
    int status = 0;

    double cost;

    // Compute pairwise distance (cost) matrix
    arma::mat cost_mat = pairwise_dist(real_data, synth_data);

    status = EMD_wrap(
        n1, n1, weights_x.memptr(), weights_s.memptr(), cost_mat.memptr(),
        t_mat.memptr(), alpha.memptr(), beta.memptr(), &cost, max_iter);

    if (p == 2)
        cost = std::sqrt(cost);
    else
        cost = std::pow(cost, 1.0 / p);

    arma::umat perm_mat = arma::conv_to<arma::umat>::from(t_mat * n1);

    arma::uvec perm =  perm_mat * arma::regspace<arma::uvec>(
            0, real_data.size() - 1);
    return std::make_tuple(perm, cost);
}

template <typename data_t>
std::tuple<arma::uvec, double> UniformSinkhorn<data_t>::compute(
    const std::vector<data_t> &real_data,
    const std::vector<data_t> &synth_data)
{
    if (!init_done)
        init(real_data, synth_data);

    cost_mat = pairwise_dist(real_data, synth_data);

    dist = -10;
    call();

    #pragma omp parallel for
    for (int i = 0; i < n_out; i++)
    {
        perm(i) = transport.row(i).index_max();
    }
    return std::make_tuple(perm, dist);
}

template <typename data_t>
void UniformSinkhorn<data_t>::init(const std::vector<data_t> &real_data,
                                   const std::vector<data_t> &synth_data)
{
    n_in = real_data.size();
    n_out = synth_data.size();

    w_in.resize(n_in);
    w_in.fill(1.0 / n_in);
    w_out.resize(n_in);
    w_out.fill(1.0 / n_out);

    cost_mat.resize(n_in, n_out);
    transport.resize(n_in, n_out);
    perm.resize(n_out);
    init_done = true;
}

#endif
