#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

#include <tuple>
#include "abc_py_class.hpp"

// [[Rcpp::export]]
Rcpp::List runAbcMCMC_univ_R(
          arma::mat data, int nrep, double theta,
          double sigma, double m0, double k0, double a0,
          double b0, double eps, int p,
          std::string dist="sorting") {

  Rcpp::Rcout << "HERE" << std::endl;
  Rcpp::Rcout << "data: " << data.n_elem << std::endl;
  AbcPyUniv abc_mcm(
      data, theta, sigma, eps, a0, b0, k0, m0, dist);
  std::tuple<arma::vec, arma::mat, double> out = abc_mcm.run(nrep);
  Rcpp::Rcout << "done" << std::endl;
  Rcpp::List res;
  res["dist"] = std::get<0>(out);
  Rcpp::Rcout << "dist done" << std::endl;
  res["part_results"] = std::get<1>(out);
  Rcpp::Rcout << "part done" << std::endl;
  res["time"] = std::get<2>(out);
  Rcpp::Rcout << "time" << std::endl;
  return res;
}

// [[Rcpp::export]]
Rcpp::List runAbcMCMC_multi_R(
          arma::mat data, int nrep, double theta,
          double sigma, arma::vec m0, double k0, double df,
          arma::mat prec_chol, double eps, int p,
          std::string dist="sinkhorn") {

  AbcPyMultiv abc_mcm(
      data, theta, sigma, eps, df, prec_chol, k0, m0, dist);
  std::tuple<arma::vec, arma::mat, double> out = abc_mcm.run(nrep);

  Rcpp::List res;
  res["dist"] = std::get<0>(out);
  res["part_results"] = std::get<1>(out);
  res["time"] = std::get<2>(out);
  return res;
}
