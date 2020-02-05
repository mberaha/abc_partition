#include "wasserstein.hpp"

std::tuple<arma::umat, double, int> d_wasserstein(
        arma::mat atoms_x, arma::mat atoms_s, double p, int max_iter) {
    // TODO Check if univariate!

    int n1 = atoms_x.n_rows;
    int n2 = atoms_s.n_rows;
    arma::vec weights_x = arma::ones<arma::vec>(n1) / n1;
    arma::vec weights_s = arma::ones<arma::vec>(n2) / n2;
    arma::vec alpha = arma::zeros<arma::vec>(n1);
    arma::vec beta = arma::zeros<arma::vec>(n2);
    arma::mat t_mat(n1, n2);
    int status = 0;

    double cost;

    // Compute pairwise distance (cost) matrix
    arma::mat cost_mat(n1, n2);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n1; ++i) {
        for (int j=0; j < n2; j++)
           cost_mat(i, j) = lp_dist(atoms_x.row(i).t(), atoms_s.row(j).t(), p);
    }

    status = EMD_wrap(
        n1, n1, weights_x.memptr(), weights_s.memptr(), cost_mat.memptr(),
        t_mat.memptr(), alpha.memptr(), beta.memptr(), &cost, max_iter);

    if (p==2) {
        cost = std::sqrt(cost);
    } else {
        cost = std::pow(cost, 1.0/p);
    }

    arma::umat perm_mat = arma::conv_to<arma::umat>::from(t_mat * n1);
    return std::make_tuple(perm_mat, cost, status);
}
