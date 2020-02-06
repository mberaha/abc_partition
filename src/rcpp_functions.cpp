#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

// #include <tuple>
// #include "abc_py_class.hpp"

// [[Rcpp::export]]
Rcpp::List test(arma::mat data) {
    arma::mat newdata(100, 1);
    newdata.fill(0.1);
    Rcpp::List res;
    return res;
}

// // [[Rcpp::export]]
// Rcpp::List runAbcMCMC_univ_R(
//           arma::mat data, int nrep=1, double theta=1.0,
//           double sigma=0.2, double m0=0.0, double k0=0.5, double a0=2.0,
//           double b0=2.0, double eps=10.0, int p=1,
//           std::string dist="sorting") {
//
//   arma::mat newdata(100, 1);
//   newdata.fill(0.1);
//
//   // Rcpp::Rcout << "HERE" << std::endl;
//   // Rcpp::Rcout << "data: " << data.n_rows << " x " << data.n_cols << std::endl;
//   // arma::mat newdata(100, 1);
//   // newdata.fill(0.0);
//   // Rcpp::Rcout << "newdata: " << newdata.n_rows << " x " << newdata.n_cols << std::endl;
//   //
//   // AbcPyUniv abc_mcm(
//   //     data, theta, sigma, eps, a0, b0, k0, m0, dist);
//   // std::tuple<arma::vec, arma::mat, double> out = abc_mcm.run(nrep);
//   //
//   // Rcpp::Rcout << "done" << std::endl;
//   Rcpp::List res;
//   // res["dist"] = std::get<0>(out);
//   // res["part_results"] = std::get<1>(out);
//   // res["time"] = std::get<2>(out);
//   return res;
// }
//
// // [[Rcpp::export]]
// Rcpp::List runAbcMCMC_multi_R(
//           arma::mat data, int nrep, double theta,
//           double sigma, arma::vec m0, double k0, double df,
//           arma::mat prec_chol, double eps, int p,
//           std::string dist="sinkhorn") {
//
//   AbcPyMultiv abc_mcm(
//       data, theta, sigma, eps, df, prec_chol, k0, m0, dist);
//   std::tuple<arma::vec, arma::mat, double> out = abc_mcm.run(nrep);
//
//   Rcpp::List res;
//   res["dist"] = std::get<0>(out);
//   res["part_results"] = std::get<1>(out);
//   res["time"] = std::get<2>(out);
//   return res;
// }
