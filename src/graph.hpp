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

// class GraphSinkhorn
// {
// protected:
//     double entropic_eps;
//     double threshold;
//     int max_iter;
//     int p = 1;

//     bool init_done = false;
//     arma::vec w_in, w_out;
//     int n_in, n_out;
//     arma::mat cost_mat;
//     arma::mat transport;
//     arma::uvec perm;
//     double dist;

//     bool greedy = false;

// public:
//     GraphSinkhorn(
//         double entropic_eps = 0.1, double threshold = 1e-4,
//         int max_iter = 100, bool greedy = false) : entropic_eps(entropic_eps), threshold(threshold),
//                                                    max_iter(max_iter), greedy(greedy) {}

//     std::tuple<arma::uvec, double> compute(
//         const std::vector<Graph> &real_data,
//         const std::vector<Graph> &synth_data);

//     void call()
//     {
//         if (greedy)
//             greenkhorn(w_in, w_out, cost_mat, entropic_eps, threshold, max_iter,
//                        p, &transport, &dist, true);
//         else
//             sinkhorn(w_in, w_out, cost_mat, entropic_eps, threshold, max_iter,
//                      p, &transport, &dist);
//     }

//     void compute_cost(
//         const std::vector<Graph> &real_data,
//         const std::vector<Graph> &synth_data);

//     void init(const std::vector<Graph> &real_data,
//               const std::vector<Graph> &synth_data);
// };

// // This is UGLY, should inherit from AbcPy
// class AbcPyGraph
// {
// protected:
//     std::vector<Graph> data;
//     std::vector<Graph> data_synt;
//     int n_nodes;

//     GraphSimulator simulator;

//     std::vector<arma::vec> param;
//     std::vector<arma::vec> tparam;

//     GraphSinkhorn *d;
//     arma::imat part_results;
//     arma::vec dist_results;
//     double curr_dist;
//     arma::ivec part;
//     arma::ivec temp_part;
//     arma::ivec tvec;
//     arma::ivec uniq_temp;

//     std::vector<arma::mat> param_results;

//     // params
//     double theta, sigma;
//     int n_data;
//     double eps0;

//     int n_unique;
//     arma::uvec sort_indices;
//     arma::vec t_diff;
//     double lEsum = 0;
//     double lEsum2 = 0;
//     double eps;
//     double meanEps, meanEps2;

//     arma::vec m0;
//     arma::mat prior_var_chol;

// public:
//     AbcPyGraph() {}
//     ~AbcPyGraph() {}

//     AbcPyGraph(
//         const std::vector<arma::mat> &data_, double theta, double sigma,
//         double eps0, const arma::mat &prior_var_chol,
//         const arma::vec &m0, std::string distance,
//         const std::vector<arma::vec> &inits,
//         int max_iter = 100, double entropic_eps = 0.1, double threshold = 1e-4);

//     void updateUrn();

//     void updateParams();

//     void generateSyntData();

//     inline void saveCurrParam()
//     {
//         param = tparam;
//     }

//     void step();

//     std::tuple<
//         arma::vec, arma::imat, double,
//         std::vector<arma::mat>>
//     run(int nrep);
// };

#endif  // GRAPH_HPP