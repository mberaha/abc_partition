#include <armadillo>
#include <tuple>
#include "src/abc_py.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


std::tuple<std::vector<double>, std::vector<std::vector<double>>, double>
runAbcMCMC(std::vector<double> data, int nrep, double theta,
          double sigma, double m0, double k0, double a0, double b0,
          double eps, int p, std::string dist="wasserstein") {

    arma::vec datavec = arma::conv_to<arma::vec>::from(data);

    std::tuple<arma::vec, arma::mat, double> out = ABC_MCMC(
        data, nrep, theta, sigma, m0, k0, a0, b0, eps, p, dist);

    int nrows = std::get<1>(out).n_rows;
    std::vector<std::vector<double>> parts(nrows);
    for (int i=0; i < nrows; i++) {
        parts[i] = arma::conv_to<std::vector<double>>::from(
            std::get<1>(out).row(i));
    }

    return std::make_tuple(
        arma::conv_to<std::vector<double>>::from(std::get<0>(out)),
        parts,
        std::get<2>(out));
}

PYBIND11_MODULE(abcpp, m) {
    m.doc() = "aaa"; // optional module docstring

    m.def("runAbcMCMC", &runAbcMCMC,
          "...");

}
