
#include <armadillo>
#include <iostream>
#include <tuple>
#include "../wasserstein.hpp"
#include "../distance.hpp"


int main() {

    Distance* d =  new UniformDiscreteWassersteinDistance();

    int natoms = 50;
    arma::vec first_clus(natoms, arma::fill::randn);
    arma::vec second_clus(natoms, arma::fill::randn);
    second_clus += 10;

    arma::vec synth_data = arma::join_cols(first_clus, second_clus);
    arma::uvec temp_part(2 * natoms, arma::fill::zeros);
    temp_part.head(natoms) += 1;

    // arma::vec atoms_x(natoms);
    // arma::vec atoms_x(2 * natoms);


    arma::vec real_data = arma::shuffle(synth_data);
    std::tuple<arma::uvec, double> out = d->compute(real_data, synth_data);
    std::cout << "Distance: " << std::get<1>(out) << std::endl;

    arma::uvec partition = temp_part(std::get<0>(out));
    arma::uvec perm = std::get<0>(out);

    for (int i=0; i < 2*natoms; i++) {
        std::cout << "x: " << real_data(i) << ", Ty: " << synth_data(perm[i])
                  << ", cluster x: " << partition[i] << std::endl;
    }

    std::cout << std::endl << std::endl;

    arma::vec cluster1 = real_data(arma::find(partition == 1));
    std::cout << "cluster 1: "; cluster1.t().print();

    arma::vec cluster0 = real_data(arma::find(partition == 0));
    std::cout << "cluster 0: "; cluster0.t().print();


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
