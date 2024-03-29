// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// run_graph
Rcpp::List run_graph(std::vector<arma::mat> data, arma::vec m0, arma::mat var_chol, int nrep, int nburn, double theta, double sigma, double eps0, double eps_star, std::string dist, const std::vector<arma::vec>& inits, bool log);
RcppExport SEXP _abcpp_run_graph(SEXP dataSEXP, SEXP m0SEXP, SEXP var_cholSEXP, SEXP nrepSEXP, SEXP nburnSEXP, SEXP thetaSEXP, SEXP sigmaSEXP, SEXP eps0SEXP, SEXP eps_starSEXP, SEXP distSEXP, SEXP initsSEXP, SEXP logSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< std::vector<arma::mat> >::type data(dataSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type m0(m0SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type var_chol(var_cholSEXP);
    Rcpp::traits::input_parameter< int >::type nrep(nrepSEXP);
    Rcpp::traits::input_parameter< int >::type nburn(nburnSEXP);
    Rcpp::traits::input_parameter< double >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< double >::type eps0(eps0SEXP);
    Rcpp::traits::input_parameter< double >::type eps_star(eps_starSEXP);
    Rcpp::traits::input_parameter< std::string >::type dist(distSEXP);
    Rcpp::traits::input_parameter< const std::vector<arma::vec>& >::type inits(initsSEXP);
    Rcpp::traits::input_parameter< bool >::type log(logSEXP);
    rcpp_result_gen = Rcpp::wrap(run_graph(data, m0, var_chol, nrep, nburn, theta, sigma, eps0, eps_star, dist, inits, log));
    return rcpp_result_gen;
END_RCPP
}
// graph_dist_R
double graph_dist_R(const arma::mat& g1, const arma::mat& g2);
RcppExport SEXP _abcpp_graph_dist_R(SEXP g1SEXP, SEXP g2SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type g1(g1SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type g2(g2SEXP);
    rcpp_result_gen = Rcpp::wrap(graph_dist_R(g1, g2));
    return rcpp_result_gen;
END_RCPP
}
// simulate_graph
arma::mat simulate_graph(const arma::vec& theta);
RcppExport SEXP _abcpp_simulate_graph(SEXP thetaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type theta(thetaSEXP);
    rcpp_result_gen = Rcpp::wrap(simulate_graph(theta));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_abcpp_run_graph", (DL_FUNC) &_abcpp_run_graph, 12},
    {"_abcpp_graph_dist_R", (DL_FUNC) &_abcpp_graph_dist_R, 2},
    {"_abcpp_simulate_graph", (DL_FUNC) &_abcpp_simulate_graph, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_abcpp(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
