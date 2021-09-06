#include "../include_arma.hpp"
#include <iostream>
#include <tuple>
#include <vector>
#include "../wasserstein.hpp"
#include "../distance.hpp"
#include "../ot/sinkhorn.hpp"


int main() {

    Distance<double>* d =  new SortingDistance1d();

    int n = 5000;
    arma::vec first_clus(n, arma::fill::randn);
    arma::vec second_clus(n, arma::fill::randn);
    second_clus += 10;

    arma::vec synth_data = arma::join_cols(first_clus, second_clus);
    arma::vec real_data = arma::sort(synth_data);

    auto out = d->compute(
        arma::conv_to<std::vector<double>>::from(real_data),
        arma::conv_to<std::vector<double>>::from(synth_data));

    std::cout << "Sorting Dist: " << std::get<1>(out) << std::endl;

}
