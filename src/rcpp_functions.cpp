#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

#include <omp.h>
#include <tuple>
#include "abc_py_class.hpp"
#include "graph.hpp"

// [[Rcpp::export]]
Rcpp::List test(arma::mat data) {
    arma::mat newdata(100, 1);
    newdata.fill(0.1);
    Rcpp::List res;
    return res;
}

// [[Rcpp::export]]
Rcpp::List runAbcMCMC_univ_R(
          Rcpp::NumericVector data, int nrep=1, double theta=1.0,
          double sigma=0.2, double m0=0.0, double k0=0.5, double a0=2.0,
          double b0=2.0, double eps=10.0, int p=1,
          std::string dist="sorting") {
  Rcpp::Rcout << "run" << std::endl;
  arma::vec datavec(data.begin(), data.size(), false);
  Rcpp::Rcout << "HERE" << std::endl;
  AbcPyUniv abc_mcm(
      datavec, theta, sigma, eps, a0, b0, k0, m0, dist);
  std::tuple<arma::vec, arma::mat, double> out = abc_mcm.run(nrep);

  Rcpp::List res;
  res["dist"] = std::get<0>(out);
  res["part_results"] = std::get<1>(out);
  res["time"] = std::get<2>(out);
  return res;
}
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

// [[Rcpp::export]]
Rcpp::List runAbcMCMC_graph(
    std::vector<arma::mat> data, int nrep, double theta,
    double sigma, arma::vec m0, arma::mat var_chol,
    const std::vector<arma::vec> &inits, double eps, std::string dist)
{
    Rcpp::Rcout << "run" << std::endl;
    Rcpp::Rcout << "HERE" << std::endl;
    AbcPyGraph abc_mcm(
        data, theta, sigma, eps, var_chol, m0, dist, inits);

    std::tuple<
        arma::vec, arma::imat, double,
        std::vector<arma::mat>> out = abc_mcm.run(nrep);

    Rcpp::List res;
    res["dist"] = std::get<0>(out);
    res["part_results"] = std::get<1>(out);
    res["time"] = std::get<2>(out);
    res["param_results"] = std::get<3>(out);
    return res;
}

// [[Rcpp::export]]
double graph_dist_R(const arma::mat &g1, const arma::mat &g2) {
    Graph gg1(g1);
    Graph gg2(g2);

    return graph_dist(gg1, gg2);
}


// [[Rcpp::export]]
arma::mat simulate_graph(const arma::vec& theta) {
    GraphSimulator simulator;
    return simulator.simulate_graph(theta, 100);
}