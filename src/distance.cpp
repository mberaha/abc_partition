#include "distance.hpp"

std::tuple<arma::uvec, double>
UniformDiscreteWassersteinDistance::compute(
        const arma::mat &real_data, const arma::mat &synth_data) {

    std::tuple<arma::umat, double, int> out = d_wasserstein(
        real_data, synth_data, p, max_iter);
    arma::umat perm_mat = std::get<0>(out);
    arma::uvec perm =  perm_mat * arma::regspace<arma::uvec>(
        0, real_data.n_rows - 1);
    return std::make_tuple(perm, std::get<1>(out));
}


std::tuple<arma::uvec, double>
SortingDistance1d::compute(
        const arma::mat &real_data, const arma::mat &synth_data) {

    arma::uvec perm = arma::sort_index(synth_data);
    double dist = arma::accu(pow(abs(real_data - synth_data(perm)), p));
    return std::make_tuple(perm, dist);
}


std::tuple<arma::uvec, double> UniformSinkhorn::compute(
        const arma::mat &real_data, const arma::mat &synth_data) {
    if (! init_done)
        init(real_data, synth_data);

    compute_cost(real_data, synth_data);

    dist=-10;
    call();

    #pragma omp parallel for
    for (int i=0; i < n_out; i++) {
        perm(i) = transport.row(i).index_max();
    }
    return std::make_tuple(perm, dist);
}

void UniformSinkhorn::init(const arma::mat &real_data,
                           const arma::mat &synth_data) {
   n_in = real_data.n_rows;
   n_out = synth_data.n_rows;

   w_in.resize(n_in);
   w_in.fill(1.0 / n_in);
   w_out.resize(n_in);
   w_out.fill(1.0 / n_out);

   cost_mat.resize(n_in, n_out);
   transport.resize(n_in, n_out);
   perm.resize(n_out);
   init_done = true;
}

void UniformSinkhorn::compute_cost(const arma::mat &real_data,
                                   const arma::mat &synth_data) {

   #pragma omp parallel for
   for (int i = 0; i < n_in;  ++i) {
       arma::rowvec curr = real_data.row(i);
       cost_mat.row(i) = arma::sum(
           arma::abs(synth_data.each_row() - curr), 1).t();
   }
}

std::tuple<arma::uvec, double> GraphSinkhorn::compute(
    const std::vector<Graph> &real_data,
    const std::vector<Graph> &synth_data)
{
    if (!init_done)
        init(real_data, synth_data);

    compute_cost(real_data, synth_data);

    dist = -10;
    call();

    #pragma omp parallel for
    for (int i = 0; i < n_out; i++)
    {
        perm(i) = transport.row(i).index_max();
    }
    return std::make_tuple(perm, dist);
}

void GraphSinkhorn::init(const std::vector<Graph> &real_data,
                         const std::vector<Graph> &synth_data)
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

void GraphSinkhorn::compute_cost(const std::vector<Graph> &real_data,
                                 const std::vector<Graph> &synth_data)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n_in; ++i) {
        for (int j = 0; j < n_out; ++j) {
            cost_mat(i, j) = graph_dist(real_data[i], synth_data[j]);
        }
    }
}