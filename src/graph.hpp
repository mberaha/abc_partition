#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

class Graph {
 protected:
    arma::mat adj_mat;
    arma::mat laplacian;
    arma::vec eigenvalues;
    arma::mat eigenvectors;
    int n_nodes;

 public:
    Graph(const arma::mat adj_mat): adj_mat(adj_mat), n_nodes(adj_mat.n_rows) {
        compute_laplacian();
        compute_eigenvalues();
    }

    void compute_laplacian();

    void compute_eigenvalues();

    arma::vec get_eigenvalues() const { return eigenvalues; }
};


double graph_dist(const Graph& g1, const Graph& g2);

class GraphSimulator {
 protected:
    Rcpp::function simulate_ergm_;

 public:
    GraphSimulator();

    arma::mat simulate_graph(arma::vec param);
}

#endif  // GRAPH_HPP