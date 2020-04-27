#include "kernels.hpp"

//
// UNIVARIATE GAUSSIAN
//

UnivGaussianKernel::UnivGaussianKernel(
    double m0, double a0, double b0, double k0):
        m0(m0), a0(a0), b0(b0), k0(k0) {}

arma::vec UnivGaussianKernel::sample_prior() 
{
    arma::vec out(2);
    out(1) = 1.0 / arma::randg(arma::distr_param(a0, 1.0 / b0));
    out(0) = arma::randn() * sqrt(out(1) / k0) + m0;
    return out;
}

std::vector<double> UnivGaussianKernel::generate_dataset(
    arma::ivec temp_part, std::vector<arma::vec> tparam)
{
    std::vector<double> data_synt(temp_part.n_elem);
    for (arma::uword j = 0; j < temp_part.n_elem; j++)
    {
        data_synt[j] = arma::randn() * sqrt(tparam[temp_part(j)](1)) + \
                       tparam[temp_part(j)](0);
    }

    return data_synt;
}

std::vector<arma::vec> UnivGaussianKernel::make_default_init() 
{
    std::vector<arma::vec> out(1);
    arma::vec param(2);
    param(0) = m0;
    param(1) = b0 / (a0 - 1);
    out[0] = param;
    return out;
}

//
// MULTIVARIATE GAUSSIAN
//

MultiGaussianKernel::MultiGaussianKernel(
    arma::vec m0, arma::mat prior_prec_chol, double df, double k0) : m0(m0), prior_prec_chol(prior_prec_chol), df(df), k0(k0)
{
}

std::tuple<arma::vec, arma::mat> MultiGaussianKernel::sample_prior() 
{
    arma::mat chol_prec = rwishart_chol(df, prior_prec_chol);
    arma::vec mean = rnorm_prec_chol(m0, sqrt(k0) * chol_prec);
    return std::make_tuple(mean, chol_prec);
}

std::vector<arma::vec> MultiGaussianKernel::generate_dataset(
    arma::ivec temp_part, std::vector<mvnorm_param> tparam)
{
    std::vector<arma::vec> data_synt(temp_part.n_elem);
    for (arma::uword j = 0; j < temp_part.n_elem; j++)
    {
        arma::vec currmean = std::get<0>(tparam[temp_part(j)]);
        arma::mat currprec = std::get<1>(tparam[temp_part(j)]);
        data_synt[j] = rnorm_prec_chol(currmean, currprec);
    }

    return data_synt;
}

std::vector<mvnorm_param> MultiGaussianKernel::make_default_init()
{
    int dim = prior_prec_chol.n_rows;
    mvnorm_param param = std::make_tuple(
        m0, arma::mat(dim, dim, arma::fill::eye));

    return std::vector<mvnorm_param>{param};
}

//
// TIME SERIES
//

TimeSeriesKernel::TimeSeriesKernel(
    double mu_mean, double mu_sd, double beta_mean, double beta_sd, 
    double xi_rate, double omega_sq_rate, double lambda_rate, int num_steps) :
         mu_mean(mu_mean), mu_sd(mu_sd), beta_mean(beta_mean),
         beta_sd(beta_sd), xi_rate(xi_rate), omega_sq_rate(omega_sq_rate),
         lambda_rate(lambda_rate), num_steps(num_steps) {}

arma::vec TimeSeriesKernel::sample_prior() 
{
    arma::vec out(5);
    out(0) = arma::randn() * mu_sd + mu_mean;
    out(1) = arma::randn() * beta_sd + beta_mean;
    out(2) = arma::randg(arma::distr_param(1.0, 1.0 / xi_rate));
    out(3) = arma::randg(arma::distr_param(1.0, 1.0 / omega_sq_rate));
    out(4) = arma::randg(arma::distr_param(1.0, 1.0 / lambda_rate));
    return out;
}

std::vector<TimeSeries> TimeSeriesKernel::generate_dataset(
    arma::ivec temp_part, std::vector<arma::vec> tparam)
{
    std::vector<TimeSeries> data_synt(temp_part.n_elem);
    double mu, beta, xi, omega_sq, lambda;

    #pragma omp parallel for
    for (arma::uword j = 0; j < temp_part.n_elem; j++)
    {
        arma::vec curr_param = tparam[temp_part(j)];
        mu = curr_param(0); beta = curr_param(1); xi = curr_param(2);
        omega_sq = curr_param(3); lambda = curr_param(4);
        data_synt[j] = generate_single(mu, beta, xi, omega_sq, lambda);
    }
    return data_synt;
}

TimeSeries TimeSeriesKernel::generate_single(
        double mu, double beta, double xi, double omega_sq, double lambda) {
    arma::vec ts(num_steps);
    double curr_z = arma::randg(arma::distr_param(
        xi * xi / omega_sq, omega_sq / xi));
    double prev_z = curr_z;
    double v_t;
    // std::cout << "generate single" << std::endl;
    TimeSeries out;
    for (int t = 0; t < num_steps; t++)
    {
        int k = stats::rpois(lambda * xi * xi / omega_sq, rng);
        if (k == 0) {
            curr_z = std::exp(-lambda) * curr_z;
            v_t = (prev_z - curr_z) / lambda;
        } else {
            // std::cout << "k: " << k << std::endl;
            arma::vec cs(k);
            arma::vec es(k);
            for (int i = 0; i < k; i++)
            {
                cs(i) = stats::runif(1.0 * t, 1.0 * t + 1, rng);
                es(i) = arma::randg(arma::distr_param(1.0, omega_sq / xi));
            }
            curr_z = std::exp(-lambda) * curr_z +
                    arma::accu((arma::exp((-cs + t + 1) * (-lambda))) % es);
            v_t = (prev_z - curr_z + arma::accu(es)) / lambda;
        }
        prev_z = curr_z;
        ts(t) = arma::randn() * std::sqrt(v_t) + mu + beta * v_t;
    }

    out = TimeSeries(ts, 1);
    return out;
}

std::vector<arma::vec> TimeSeriesKernel::make_default_init()
{
    arma::vec param(5);
    param(0) = mu_mean;
    param(1) = beta_mean;
    param(2) = 1.0 / xi_rate;
    param(3) = 1.0 / omega_sq_rate;
    param(4) = 1.0 / lambda_rate;
    return std::vector<arma::vec>{param};
}

//
// GRAPH
//

#ifdef USE_R

GraphKernel::GraphKernel(int n_nodes, arma::vec m0, arma::mat prior_prec_chol) {
    this->n_nodes = n_nodes;
    this->m0 = m0;
    this->prior_prec_chol = prior_prec_chol;
}

arma::vec GraphKernel::sample_prior() {
    return rnorm_prec_chol(m0, prior_prec_chol);
}

std::vector<Graph> GraphKernel::generate_dataset(
    arma::ivec temp_part, std::vector<arma::vec> tparam)
{
    std::vector<Graph> data_synt(temp_part.n_elem);
    for (arma::uword j = 0; j < temp_part.n_elem; j++)
    {
        arma::vec currparam = tparam[temp_part(j)];
        arma::mat g = simulator.simulate_graph(currparam, n_nodes);
        data_synt[j] = Graph(g);
    }
    return data_synt;
}

std::vector<arma::vec> GraphKernel::make_default_init() {
    return std::vector<arma::vec>{m0};
}

#endif