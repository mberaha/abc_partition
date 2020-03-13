#include "../graph.hpp"
#include "../include_arma.hpp"
#include "../distance.hpp"
#include "../abc_py_class.hpp"

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

    int ngraphs = 50;
    std::vector<Graph> graphs1(ngraphs);
    std::vector<Graph> graphs2(ngraphs);
    std::vector<arma::mat> data(ngraphs);
    for (int i=0; i < ngraphs; i++) {
        arma::mat g = generate_graph(n_nodes, arma::randu() * thr);
        data[i] = g;
        graphs1[i] = Graph(g);
        graphs2[ngraphs -1 - i] = Graph(g);
    }

    GraphSinkhorn distance;
    std::tuple<arma::uvec, double> dist = distance.compute(graphs1, graphs2);

    std::string a = "afaa";
    AbcPyGraph abcpy(
        data, 1.0, 0.2, 1000, arma::mat(6, 6, arma::fill::eye),
        arma::vec(6, arma::fill::zeros), a);

    std::tuple<arma::vec, arma::mat, double> out = abcpy.run(100);

    std::cout << "graph sinkhorn distance: \n"
        << std::get<0>(dist).t() << "\n"
        << std::get<1>(dist) << std::endl;
}