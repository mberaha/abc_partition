#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "include_arma.hpp"
#include <tuple>
#include "src/abc_py_class.hpp"

namespace py = pybind11;
typedef py::array_t<double, py::array::f_style | py::array::forcecast> pyarr_d;
typedef py::array_t<arma::uword, py::array::forcecast> pyarr_u;


inline arma::mat py_to_mat(pyarr_d& pmat) {
	py::buffer_info info = pmat.request();
	arma::mat amat;
	if(info.ndim == 1) {
		amat = arma::mat(reinterpret_cast<double*>(info.ptr),info.shape[0],1);
	} else {
		amat = arma::mat(reinterpret_cast<double*>(info.ptr),info.shape[0],info.shape[1]);
	}
	return amat;
}

std::tuple<std::vector<double>, std::vector<std::vector<double>>, double>
runAbcMCMC_univ(std::vector<double> data, int nrep, double theta,
          double sigma, double m0, double k0, double a0, double b0,
          double eps, int p, std::string dist="wasserstein") {

    arma::vec datavec = arma::conv_to<arma::vec>::from(data);

    AbcPyUniv abc_mcm(datavec, theta, sigma, eps, a0, b0, k0, m0, dist);
    std::tuple<arma::vec, arma::mat, double> out = abc_mcm.run(nrep);

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


std::tuple<std::vector<double>, std::vector<std::vector<double>>, double>
runAbcMCMC_multi(pyarr_d data, int nrep, double theta,
          double sigma, pyarr_d m0, double k0, double df,
          pyarr_d prec_chol, double eps, int p,
          std::string dist="wasserstein") {

    arma::mat datamat = py_to_mat(data);
    arma::vec m0p = py_to_mat(m0);
    arma::mat prior_prec_chol = py_to_mat(prec_chol);

    AbcPyMultiv abc_mcm(
        datamat, theta, sigma, eps, df, prior_prec_chol, k0, m0p, dist);
    std::tuple<arma::vec, arma::mat, double> out = abc_mcm.run(nrep);

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

    m.def("runAbcMCMC_univ", &runAbcMCMC_univ,
          "...");

    m.def("runAbcMCMC_multi", &runAbcMCMC_multi,
        "...");

}
