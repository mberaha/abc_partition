#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "src/include_arma.hpp"
#include <tuple>
#include "src/abc_py_class.hpp"
#include "src/time_series.hpp"

#include "carma/carma.h"

namespace py = pybind11;
typedef py::array_t<double, py::array::f_style | py::array::forcecast> pyarr_d;
typedef py::array_t<arma::uword, py::array::forcecast> pyarr_u;
typedef py::array_t<int, py::array::forcecast> pyarr_i;

using py_return_t_univ = std::tuple<
    pyarr_d, pyarr_i, double, std::vector<std::vector<pyarr_d>>>;

using py_return_t_multi = std::tuple<
    pyarr_d, pyarr_i, double, std::vector<std::vector<std::tuple<pyarr_d, pyarr_d>>>>;

using py_return_t_ts = std::tuple<
    pyarr_d, pyarr_i, double, std::vector<std::vector<pyarr_d>>>;


py_return_t_univ run_univariate(
    std::vector<double> data, double m0, double a0, double b0, double k0,
    int nrep, double theta, double sigma, double eps, int p = 1,
    std::string dist = "sorting", std::vector<std::vector<double>> inits_ = {},
    bool log=false)
{
    UnivGaussianKernel kernel(m0, a0, b0, k0);

    std::vector<arma::vec> inits;
    if (inits_.size() == 0)
        inits = kernel.make_default_init();
    else
        for (int i = 0; i < inits_.size(); i++)
            inits.push_back(arma::conv_to<arma::vec>::from(inits_[i]));

    UnivAbcPy abc_mcmc(data, inits, theta, sigma, eps, eps, dist, kernel);

    std::cout << "Initialized object" << std::endl;
    
    if (log)
        abc_mcmc.set_log();

    double time = abc_mcmc.run(nrep, int(nrep / 2));

    std::vector<std::vector<arma::vec>> params_log = abc_mcmc.get_params_log();
    std::vector<std::vector<pyarr_d>> params_log_py(params_log.size());
    for (int i=0; i < params_log.size(); i++) {
        std::vector<pyarr_d> curr(params_log[i].size());
        for (int k=0; k < params_log[i].size(); k++)
            curr[k] = carma::mat_to_arr(params_log[i][k]);

        params_log_py[i] = curr;
    }

    arma::vec dists = abc_mcmc.get_dists();
    arma::imat parts = abc_mcmc.get_parts();

    return std::make_tuple(
        carma::mat_to_arr(dists), carma::mat_to_arr(parts), time, params_log_py);
}

py_return_t_multi run_multivariate(
    pyarr_d data, pyarr_d m0, double df, pyarr_d prec_chol, double k0,
    int nrep, double theta, double sigma, double eps, int p = 1,
    std::string dist = "wasserstein",
    std::vector<std::tuple<pyarr_d, pyarr_d>> inits_ = {}, bool log = false) {
    
    MultiGaussianKernel kernel(
        carma::arr_to_mat<double>(m0), 
        carma::arr_to_mat<double>(prec_chol), df, k0);
    std::vector<mvnorm_param> inits;
    if (inits_.size() == 0)
        inits = kernel.make_default_init();
    else {
        for (int i=0; i < inits_.size(); i++) {
            arma::vec mu = carma::arr_to_mat<double>(std::get<0>(inits_[i]));
            arma::mat sigma = carma::arr_to_mat<double>(std::get<1>(inits_[i]));
            inits.push_back(std::make_tuple(mu, sigma));
        }
    }

    MultiAbcPy abc_mcmc(
        to_vectors(carma::arr_to_mat<double>(data)), 
        inits, theta, sigma, eps, eps, dist, kernel);
    if (log)
        abc_mcmc.set_log();

    double time = abc_mcmc.run(nrep, int(nrep / 2));

    std::vector<std::vector<mvnorm_param>> params_log;

    std::vector<std::vector<std::tuple<pyarr_d, pyarr_d>>> params_log_py(params_log.size());
    for (int i = 0; i < params_log.size(); i++) {
        std::vector<std::tuple<pyarr_d, pyarr_d>> curr(params_log[i].size());
        for (int k=0; k < params_log[i].size(); k++) {
            pyarr_d mean = carma::mat_to_arr<double>(std::get<0>(params_log[i][k]));
            pyarr_d sigma_chol = carma::mat_to_arr<double>(std::get<1>(params_log[i][k]));
            curr[k] = std::make_tuple(mean, sigma_chol);
        }
        params_log_py[i] = curr;
    }

    arma::vec dists = abc_mcmc.get_dists();
    arma::imat parts = abc_mcmc.get_parts();

    return std::make_tuple(
        carma::mat_to_arr(dists), carma::mat_to_arr(parts), time, params_log_py);
}

py_return_t_multi run_gandk(
    pyarr_d data, double rho, int nrep, int nburn, 
    double theta, double sigma, 
    double eps0, double eps_star, 
    int p = 1, std::string dist = "wasserstein",
    std::vector<std::tuple<pyarr_d, pyarr_d>> inits_ = {}, bool log = false) {
    
    MultiGandKKernel kernel(rho);
    std::vector<gandk_param> inits;
    inits = kernel.make_default_init();
    // if (inits_.size() == 0)
    //     inits = kernel.make_default_init();
    // else {
    //     for (int i=0; i < inits_.size(); i++) {
    //         arma::vec mu = carma::arr_to_mat<double>(std::get<0>(inits_[i]));
    //         arma::mat sigma = carma::arr_to_mat<double>(std::get<1>(inits_[i]));
    //         inits.push_back(std::make_tuple(mu, sigma));
    //     }
    // }

    MultiGnKAbcPy abc_mcmc(
        to_vectors(carma::arr_to_mat<double>(data)), 
        inits, theta, sigma, eps0, eps_star, dist, kernel);
    if (log)
        abc_mcmc.set_log();

    double time = abc_mcmc.run(nrep, nburn);

    std::vector<std::vector<gandk_param>> params_log;
    std::vector<std::vector<std::tuple<pyarr_d, pyarr_d>>> params_log_py(params_log.size());
    
    // for (int i = 0; i < params_log.size(); i++) {
    //     std::vector<std::tuple<pyarr_d, pyarr_d>> curr(params_log[i].size());
    //     for (int k=0; k < params_log[i].size(); k++) {
    //         pyarr_d mean = carma::mat_to_arr<double>(std::get<0>(params_log[i][k]));
    //         pyarr_d sigma_chol = carma::mat_to_arr<double>(std::get<1>(params_log[i][k]));
    //         curr[k] = std::make_tuple(mean, sigma_chol);
    //     }
    //     params_log_py[i] = curr;
    // }

    arma::vec dists = abc_mcmc.get_dists();
    arma::imat parts = abc_mcmc.get_parts();

    return std::make_tuple(
        carma::mat_to_arr(dists), carma::mat_to_arr(parts), time, params_log_py);

}

py_return_t_ts run_timeseries(
    pyarr_d data, double mu_mean, double mu_sd, double beta_mean,
    double beta_sd, double xi_rate, double omega_sq_rate, double lambda_rate,
    int nrep, double theta, double sigma, double eps, int p = 1,
    std::string dist = "sorting", std::vector<std::vector<double>> inits_ = {},
    bool log = false)
{
    arma::mat datamat = carma::arr_to_mat<double>(data);
    int nsteps = datamat.n_cols;

    TimeSeriesKernel kernel(
        mu_mean, mu_sd, beta_mean, beta_sd, xi_rate, omega_sq_rate,
        lambda_rate, nsteps);

    std::vector<arma::vec> inits;
    if (inits_.size() == 0)
        inits = kernel.make_default_init();
    else {
        for (int i=0; i < inits_.size(); i++)
            inits.push_back(arma::conv_to<arma::vec>::from(inits_[i]));
    }

    std::vector<TimeSeries> datavec(datamat.n_rows);
    for (int i = 0; i < datamat.n_rows; i++)
        datavec[i] = TimeSeries(datamat.row(i).t());

    TimeSeriesAbcPy abc_mcmc(datavec, inits, theta, sigma,
                             eps, eps, dist, kernel);

    if (log)
        abc_mcmc.set_log();

    double time = abc_mcmc.run(nrep, int(nrep / 2));

    std::vector<std::vector<arma::vec>> params_log = abc_mcmc.get_params_log();
    std::vector<std::vector<pyarr_d>> params_log_py(params_log.size());
    for (int i = 0; i < params_log.size(); i++)
    {
        std::vector<pyarr_d> curr(params_log[i].size());
        for (int k = 0; k < params_log[i].size(); k++)
            curr[k] = carma::mat_to_arr(params_log[i][k]);

        params_log_py[i] = curr;
    }

    arma::vec dists = abc_mcmc.get_dists();
    arma::imat parts = abc_mcmc.get_parts();
    std::cout << "parts \n" << parts << std::endl;

    return std::make_tuple(
        carma::mat_to_arr(dists), carma::mat_to_arr(parts), time, params_log_py);
}

std::vector<double> rand_gandk(double rho, pyarr_d a, pyarr_d b, pyarr_d c, pyarr_d k) {
    MultiGandKKernel kern(rho);
    gandk_param p {
        carma::arr_to_mat<double>(a).col(0),
        carma::arr_to_mat<double>(b).col(0),
        carma::arr_to_mat<double>(c).col(0),
        carma::arr_to_mat<double>(k).col(0),
    };
    arma::vec out = kern.rand_from_param(p);
    return arma::conv_to<std::vector<double>>::from(out);
}

std::vector<double> simulate_ts(
    int num_steps, double mu, double beta, double xi, double omega_sq, 
    double lambda) 
{
    TimeSeriesKernel kernel(num_steps);
    TimeSeries out = kernel.generate_single(mu, beta, xi, omega_sq, lambda);
    return arma::conv_to<std::vector<double>>::from(out.get_ts());
}

PYBIND11_MODULE(abcpp, m)
{
    m.doc() = "aaa"; // optional module docstring

    py::add_ostream_redirect(m, "ostream_redirect");

    m.def("run_univariate", &run_univariate,
          "...");

    m.def("run_multivariate", &run_multivariate,
          "...");

    m.def("run_gandk", &run_gandk,
          "...");

    m.def("run_timeseries", &run_timeseries,
          "...");

    m.def("rand_gandk", &rand_gandk, "...");

    m.def("simulate_ts", &simulate_ts, "...");
}
