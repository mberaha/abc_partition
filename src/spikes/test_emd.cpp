
#include <armadillo>
#include <iostream>
#include <tuple>
#include "../wasserstein.hpp"
#include "../distance.hpp"
#include "../ot/sinkhorn.hpp"


int main() {

    Distance* d =  new UniformSinkhorn();

    int n = 10;
    arma::mat first_clus(n, 2, arma::fill::randn);
    arma::mat second_clus(n, 2, arma::fill::randn);
    second_clus += 10;

    arma::mat synth_data = arma::join_cols(first_clus, second_clus);
    arma::mat real_data = arma::shuffle(synth_data, 0);
    //
    // std::tuple<arma::uvec, double> out = d->compute(real_data, synth_data);
    // std::cout << "Distance: " << std::get<1>(out) << std::endl;

    // arma::uvec partition = temp_part(std::get<0>(out));
    // arma::uvec perm = std::get<0>(out);

    // for (int i=0; i < partition.n_elem; i++) {
    //     std::cout << "r: " << real_data(i) << ", s: " << synth_data(perm[i])
    //               << ", cluster x: " << partition[i] << std::endl;
    // }

    //

    arma::mat transport(n * 2, n * 2);
    double dist = -10;

    arma::vec weights_in(n * 2, arma::fill::ones);
    weights_in /= (2 * n);

    arma::vec weights_out(n * 2, arma::fill::ones);
    weights_out /= (2 * n);

    arma::mat cost_mat(n * 2, n * 2);
    for (int i = 0; i < n * 2; ++i) {
        for (int j=0; j < n * 2; j++)
           cost_mat(i, j) = lp_dist(real_data.row(i).t(), synth_data.row(j).t(), 1);
    }

    double eps = 1e-3;
    double threshold = 1.0;

    sinkhorn(weights_in, weights_out, cost_mat, eps, threshold, 100,
             1, &transport, &dist);

    std::cout << "distance: " << dist << std::endl;
    //
    // std::cout << "transport\n"; transport.print();
    std::cout << "Mappings: " << std::endl;
    for (int i=0; i < 10; i++) {
        int ind1 = transport.col(i).index_max();
        std::cout << "y: " << synth_data.row(i).t() << ", Tx: " << real_data.row(ind1).t() << std::endl;
    }



    //
    //
    // out = d_wasserstein(atoms_x, atoms_y, 2.0);
    // arma::umat perm_mat = std::get<0>(out);
    // std::cout << "distance: " << std::get<1>(out) << ", status:" << std::get<2>(out) << std::endl;
    // std::cout << "Mappings: " << std::endl;
    //
    // arma::uvec perm =  perm_mat.t() * arma::regspace<arma::uvec>(0, atoms_x.n_elem-1);
    // for (int i=0; i < 100; i++) {
    //     arma::vec tx = perm_mat.col(i).t() * atoms_x;
    //     std::cout << "y: " << atoms_y(i) << ", Tx: " << tx(0) <<
    //                  ", with perm: " << atoms_x(perm(i)) << std::endl;
    // }

    // atoms_y.randu();
    // out = d_wasserstein(atoms_x, atoms_y, 2.0);
    // std::cout << "distance: " << std::get<1>(out) << ", status:" << std::get<2>(out) << std::endl;
    // std::cout << "Mappings: " << std::endl;
    // for (int i=0; i < 10; i++) {
    //     arma::uvec ind1 = arma::find(t_mat.col(i) > 0);
    //     std::cout << "y: " << atoms_y(i) << ", Tx: " << atoms_x(ind1(0)) << std::endl;
    // }
}
