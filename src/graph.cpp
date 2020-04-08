#include "graph.hpp"

using namespace arma;

void Graph::compute_laplacian()
{
    vec degree = sum(adj_mat, 1) + 0.01;
    vec inv_degree = pow(degree, -1);
    mat A = arma::sqrt(diagmat(inv_degree));

    laplacian = mat(n_nodes, n_nodes, fill::eye) - A * adj_mat * A;
}

void Graph::compute_eigenvalues()
{
    eig_sym(eigenvalues, eigenvectors, laplacian);
}

double graph_dist(const Graph &g1, const Graph &g2)
{
    int n1 = g1.get_n_edges();
    int n2 = g2.get_n_edges();
    const arma::vec eig1 = g1.get_eigenvalues();
    const arma::vec eig2 = g2.get_eigenvalues();
    double out = arma::sum(arma::pow(eig1 - eig2, 2));
    out *= 1.0 * (std::abs(n1 - n2) + 1.0) / (1.0 * n1 + n2 + 1.0);
    return out;
}

arma::mat GraphSimulator::simulate_graph(arma::vec param, int n_nodes)
{
    Rcpp::NumericVector out = simulate_ergm_(param, n_nodes);
    return arma::mat(out.begin(), n_nodes, n_nodes);
}