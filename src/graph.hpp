#ifndef GRAPH_HPP
#define GRAPH_HPP

#include "include_arma.hpp"
#include "Rcpp.h"


class Graph {
 protected:
    arma::mat adj_mat;
    arma::mat laplacian;
    arma::vec eigenvalues;
    arma::mat eigenvectors;
    int n_nodes;
    int n_edges;

 public:
    Graph() {}

    Graph(const arma::mat adj_mat): adj_mat(adj_mat), n_nodes(adj_mat.n_rows) {
        compute_laplacian();
        compute_eigenvalues();
        n_edges = (int) arma::accu(adj_mat);
    }

    void compute_laplacian();

    void compute_eigenvalues();

    arma::vec get_eigenvalues() const { return eigenvalues; }

    int get_n_nodes() const {return n_nodes;}

    int get_n_edges() const {return n_edges;}
};

class GraphSimulator
{
protected:
    Rcpp::Function simulate_ergm_ = Rcpp::Environment::global_env()["simulate_ergm"];

public:
    GraphSimulator() {}

    arma::mat simulate_graph(arma::vec param, int n_nodes);
};


double graph_dist(const Graph& g1, const Graph& g2);

arma::mat pairwise_dist(const std::vector<Graph> &x,
                        const std::vector<Graph> &y);


#endif  // GRAPH_HPP