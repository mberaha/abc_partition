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

    double dist=-10;
    sinkhorn(w_in, w_out, cost_mat, eps, threshold, max_iter,
             p, &transport, &dist);

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
   arma::rowvec curr;
   for (int i = 0; i < n_in;  ++i) {
       curr = real_data.row(i);
       cost_mat.row(i) = arma::sum(
           arma::abs(synth_data.each_row() - curr), 1).t();
   }
}
