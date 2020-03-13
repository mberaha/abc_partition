#include "../graph.hpp"
#include "../include_arma.hpp"


arma::mat generate_graph(int n_nodes, double thr) {
    arma::mat adj1(n_nodes, n_nodes, arma::fill::zeros);
    for (int i = 0; i < n_nodes; i++)
    {
        for (int j = 0; j < i; j++)
        {
            double u = arma::randu();
            if (arma::randu() < thr)
            {
                adj1(i, j) = 1;
                adj1(j, i) = 1;
            }
        }
    }
    return adj1;
}

int main() {
    int n_nodes = 100;
    double thr = 0.3;

    std::vector<Graph> graphs;
    graphs.push_back(Graph(generate_graph(100, 0.1)));
    graphs.push_back(Graph(generate_graph(100, 0.15)));
    graphs.push_back(Graph(generate_graph(100, 0.6)));
    graphs.push_back(Graph(generate_graph(100, 0.65)));

    for (int i=0; i < graphs.size(); i++) {
        for (int j=0; j < graphs.size(); j++) {
            std::cout << "d(g" << i << ", g" << j << ") = "
                      << graph_dist(graphs[i], graphs[j]) << std::endl;
        }
    }
}