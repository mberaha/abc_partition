#include "distance.hpp"

// std::tuple<arma::uvec, double>
// UniformDiscreteWassersteinDistance::compute(
//         const arma::mat &real_data, const arma::mat &synth_data) {

//     std::tuple<arma::umat, double, int> out = d_wasserstein(
//         real_data, synth_data, p, max_iter);
//     arma::umat perm_mat = std::get<0>(out);
//     arma::uvec perm =  perm_mat * arma::regspace<arma::uvec>(
//         0, real_data.n_rows - 1);
//     return std::make_tuple(perm, std::get<1>(out));
// }


// std::tuple<arma::uvec, double>
// SortingDistance1d::compute(
//         const arma::mat &real_data, const arma::mat &synth_data) {
//     arma::uvec perm = arma::sort_index(synth_data);
//     double dist = arma::accu(pow(abs(real_data - synth_data(perm)), p));
//     return std::make_tuple(perm, dist);
// }