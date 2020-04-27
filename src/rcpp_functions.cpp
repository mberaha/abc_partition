#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

#include <omp.h>
#include <tuple>
#include "abc_py_class.hpp"
#include "graph.hpp"
#include "time_series.hpp"
#include "kernels.hpp"

// [[Rcpp::export]]
Rcpp::List run_univariate(
    arma::vec data, int nrep, double theta, double sigma, double m0,
    double k0, double a0, double b0, double eps, int p,
    std::string dist, const std::vector<arma::vec> &inits,
    bool log=false)
{

    UnivGaussianKernel kernel(m0, a0, b0, k0);
    std::vector<double> datavec = arma::conv_to<std::vector<double>>::from(data);
    UnivAbcPy abc_mcmc(datavec, inits, theta, sigma, eps, dist, kernel);

    if (log)
        abc_mcmc.set_log();

    double time = abc_mcmc.run(nrep);

    Rcpp::List res;
    res["dist"] = abc_mcmc.get_dists();
    res["part_results"] = abc_mcmc.get_parts();
    res["time"] = time;
    // res["param_log"] = abc_mcmc.get_params_log();
    return res;
}

// [[Rcpp::export]]
Rcpp::List run_multivariate(
    arma::mat data, int nrep, double theta, double sigma, arma::vec m0, 
    double k0, double df, arma::mat prec_chol, double eps, int p,
    std::string dist, const Rcpp::List &inits_,
    bool log=true)
{
    MultiGaussianKernel kernel(m0, prec_chol, df, k0);

    std::vector<mvnorm_param> inits(inits_.size());
    for (int i=0; i < inits_.size(); i++) {
        Rcpp::List curr = Rcpp::as<Rcpp::List>(inits_[i]);
        arma::vec mu = Rcpp::as<arma::vec>(curr[0]);
        arma::mat prec_chol = Rcpp::as<arma::mat>(curr[1]);
        inits[i] = std::make_tuple(mu, prec_chol);
    }

    MultiAbcPy abc_mcmc(to_vectors(data), inits, theta, sigma,
                        eps, dist, kernel);

    if (log)
        abc_mcmc.set_log();

    double time = abc_mcmc.run(nrep);

    Rcpp::List res;
    res["dist"] = abc_mcmc.get_dists();
    res["part_results"] = abc_mcmc.get_parts();
    res["time"] = time;
    // res["param_log"] = vstack(abc_mcmc.get_params_log());
    return res;
}

// [[Rcpp::export]]
Rcpp::List run_timeseries(
    arma::mat data, double mu_mean, double mu_sd, double beta_mean,
    double beta_sd, double xi_rate, double omega_sq_rate, double lambda_rate,
    int nrep, double theta, double sigma, double eps,
    std::string dist, std::vector<arma::vec> inits,
    bool log = false) 
{
    int nsteps = data.n_cols;

    TimeSeriesKernel kernel(
        mu_mean, mu_sd, beta_mean, beta_sd, xi_rate, omega_sq_rate,
        lambda_rate, nsteps);

    std::vector<TimeSeries> datavec(data.n_rows);
    for (int i = 0; i < data.n_rows; i++)
        datavec[i] = TimeSeries(data.row(i).t());

    TimeSeriesAbcPy abc_mcmc(datavec, inits, theta, sigma,
                             eps, dist, kernel);

    if (log)
        abc_mcmc.set_log();

    double time = abc_mcmc.run(nrep);
    Rcpp::List res;
    res["dist"] = abc_mcmc.get_dists();
    res["part_results"] = abc_mcmc.get_parts();
    res["time"] = time;
    // res["param_log"] = abc_mcmc.get_params_log();
    return res;
}

// [[Rcpp::export]]
Rcpp::List run_graph(
    std::vector<arma::mat> data, arma::vec m0, arma::mat var_chol,
    int nrep, double theta, double sigma, double eps,
    std::string dist, const std::vector<arma::vec> &inits,
    bool log = false)
{
    Rcpp::Rcout << "1" << std::endl;
    int n_nodes = data[0].n_rows;
    Rcpp::Rcout << "n_nodes: " << n_nodes << std::endl;
    GraphKernel kernel(n_nodes, m0, var_chol);
    Rcpp::Rcout << "2" << std::endl;

    std::vector<Graph> graphs(data.size());
    for (int i=0; i < data.size(); i++)
        graphs[i] = Graph(data[i]);

    Rcpp::Rcout << "3" << std::endl;

    GraphAbcPy abc_mcmc(
        graphs, inits, theta, sigma, eps, dist, kernel);

    Rcpp::Rcout << "4" << std::endl;

    if (log)
        abc_mcmc.set_log();

    double time = abc_mcmc.run(nrep);

    Rcpp::Rcout << "5" << std::endl;

    Rcpp::List res;
    res["dist"] = abc_mcmc.get_dists();
    res["part_results"] = abc_mcmc.get_parts();
    res["time"] = time;
    // res["param_log"] = abc_mcmc.get_params_log();

    return res;
}

// [[Rcpp::export]]
arma::vec simulate_ts(
    int num_steps, double mu, double beta, double xi, double omega_sq,
    double lambda)
{
    TimeSeriesKernel kernel(num_steps);
    TimeSeries out = kernel.generate_single(mu, beta, xi, omega_sq, lambda);
    return out.get_ts();
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