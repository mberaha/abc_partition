#include "distance.hpp"

std::tuple<arma::uvec, double>
UniformDiscreteWassersteinDistance::compute(
        const arma::mat &real_data, const arma::mat &synth_data) {

    std::tuple<arma::umat, double, int> out = d_wasserstein(
        real_data, synth_data, p, max_iter);
    arma::umat perm_mat = std::get<0>(out);
    arma::uvec perm =  perm_mat * arma::regspace<arma::uvec>(
        0, real_data.n_elem-1);
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

    int n_in = real_data.n_rows;
    int n_out = synth_data.n_rows;

    arma::vec w_in(n_in, arma::fill::ones);
    arma::vec w_out(n_out, arma::fill::ones);
    w_in /= n_in;
    w_out /= n_out;

    arma::mat cost_mat(n_in, n_out);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n_in;  ++i) {
        for (int j=0; j < n_out; ++j)
           cost_mat(i, j) = lp_dist(real_data.row(i), synth_data.row(j), p);
    }

    // std::cout << "cost_mat \n" << cost_mat << std::endl;

    arma::mat transport(n_in, n_out);
    double dist=-10;
    sinkhorn(w_in, w_out, cost_mat, eps, threshold, max_iter,
             p, &transport, &dist);

    // std::cout << "transport \n" << transport << std::endl;
    arma::uvec perm(n_out);
    for (int i=0; i < n_out; i++) {
        perm(i) = transport.row(i).index_max();
    }

    return std::make_tuple(perm, dist);
}
