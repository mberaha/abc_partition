#include "src/kernels.hpp"
#include "src/distance.hpp"
#include "src/abc_py_class.hpp"
#include "src/time_series.hpp"
#include <iostream>

struct ts_param {
    double mu_mean;
    double mu_sd;
    double beta_mean;
    double beta_sd;
    double xi_rate;
    double omega_rate;
    double lambda_rate;
};


std::vector<TimeSeries> load_data(std::string file_name) {
    arma::mat data;
    data.load(file_name, arma::csv_ascii);
    std::vector<TimeSeries> ts_vec;
    for (int i = 1; i < data.n_cols; i++) {
        TimeSeries ts(data.col(i));
        ts_vec.push_back(ts);
    }
    return ts_vec;
}

std::vector<TimeSeries> generate_data(int data_per_clus) {
    std::vector<TimeSeries> ts_vec(data_per_clus * 2); 
    TimeSeriesKernel kernel(50);
    for (int i = 0; i < data_per_clus; i++) {
        ts_vec[i] = kernel.generate_single(1.5, 2.75, 1.0, 2.5, 1.0);
        ts_vec[i + data_per_clus] = kernel.generate_single(1.0, 2.0, 0.6, 1, 0.4);
    }
    return ts_vec;
}

double run_one(ts_param params, int i, std::string data_dir) {
    std::vector<TimeSeries> ts_vec = generate_data(25);

    TimeSeriesKernel kernel(
        params.mu_mean, params.mu_sd, params.beta_mean, params.beta_sd, 
        params.xi_rate, params.omega_rate,
        params.lambda_rate, ts_vec[0].get_ts().n_elem);

    std::vector<arma::vec> inits = kernel.make_default_init();

    TimeSeriesAbcPy abc(ts_vec, inits, 1.0, 0.1,
                        100, 0.5,  "wasserstein", kernel);

    double time = abc.run(20000, 10000, false);

    std::string outfile = data_dir + "abc_out_" + std::to_string(i) + ".csv";
    arma::imat parts = abc.get_parts();
    parts.save(outfile, arma::csv_ascii);
    return time;
}

int main() {
    std::string data_dir = "data/";
    std::vector<ts_param> params;

    params.push_back(ts_param{0.0, 2.0, 0.0, 2.0, 1.0, 1.0, 1.0});
    params.push_back(ts_param{0.0, 2.0, 0.0, 2.0, 2.0, 2.0, 2.0});
    params.push_back(ts_param{0.0, 2.0, 0.0, 2.0, 5.0, 5.0, 5.0});
    params.push_back(ts_param{0.0, 2.0, 0.0, 2.0, 5.0, 1.0, 1.0});
    params.push_back(ts_param{0.0, 2.0, 0.0, 2.0, 5.0, 2.0, 2.0});
    params.push_back(ts_param{0.0, 2.0, 0.0, 2.0, 1.0, 5.0, 1.0});
    params.push_back(ts_param{0.0, 2.0, 0.0, 2.0, 2.0, 5.0, 2.0});
    params.push_back(ts_param{0.0, 2.0, 0.0, 2.0, 1.0, 1.0, 5.0});
    params.push_back(ts_param{0.0, 2.0, 0.0, 2.0, 2.0, 2.0, 5.0});

    params.push_back(ts_param{0.0, 5.0, 0.0, 2.0, 1.0, 1.0, 1.0});
    params.push_back(ts_param{0.0, 5.0, 0.0, 2.0, 2.0, 2.0, 2.0});
    params.push_back(ts_param{0.0, 5.0, 0.0, 2.0, 5.0, 5.0, 5.0});
    params.push_back(ts_param{0.0, 5.0, 0.0, 2.0, 5.0, 1.0, 1.0});
    params.push_back(ts_param{0.0, 5.0, 0.0, 2.0, 5.0, 2.0, 2.0});
    params.push_back(ts_param{0.0, 5.0, 0.0, 2.0, 1.0, 5.0, 1.0});
    params.push_back(ts_param{0.0, 5.0, 0.0, 2.0, 2.0, 5.0, 2.0});
    params.push_back(ts_param{0.0, 5.0, 0.0, 2.0, 1.0, 1.0, 5.0});
    params.push_back(ts_param{0.0, 5.0, 0.0, 2.0, 2.0, 2.0, 5.0});

    params.push_back(ts_param{0.0, 2.0, 0.0, 5.0, 1.0, 1.0, 1.0});
    params.push_back(ts_param{0.0, 2.0, 0.0, 5.0, 2.0, 2.0, 2.0});
    params.push_back(ts_param{0.0, 2.0, 0.0, 5.0, 5.0, 5.0, 5.0});
    params.push_back(ts_param{0.0, 2.0, 0.0, 5.0, 5.0, 1.0, 1.0});
    params.push_back(ts_param{0.0, 2.0, 0.0, 5.0, 5.0, 2.0, 2.0});
    params.push_back(ts_param{0.0, 2.0, 0.0, 5.0, 1.0, 5.0, 1.0});
    params.push_back(ts_param{0.0, 2.0, 0.0, 5.0, 2.0, 5.0, 2.0});
    params.push_back(ts_param{0.0, 2.0, 0.0, 5.0, 1.0, 1.0, 5.0});
    params.push_back(ts_param{0.0, 2.0, 0.0, 5.0, 2.0, 2.0, 5.0});

    
    params.push_back(ts_param{0.0, 5.0, 0.0, 5.0, 1.0, 1.0, 1.0});
    params.push_back(ts_param{0.0, 5.0, 0.0, 5.0, 2.0, 2.0, 2.0});
    params.push_back(ts_param{0.0, 5.0, 0.0, 5.0, 5.0, 5.0, 5.0});
    params.push_back(ts_param{0.0, 5.0, 0.0, 5.0, 5.0, 1.0, 1.0});
    params.push_back(ts_param{0.0, 5.0, 0.0, 5.0, 5.0, 2.0, 2.0});
    params.push_back(ts_param{0.0, 5.0, 0.0, 5.0, 1.0, 5.0, 1.0});
    params.push_back(ts_param{0.0, 5.0, 0.0, 5.0, 2.0, 5.0, 2.0});
    params.push_back(ts_param{0.0, 5.0, 0.0, 5.0, 1.0, 1.0, 5.0});
    params.push_back(ts_param{0.0, 5.0, 0.0, 5.0, 2.0, 2.0, 5.0});

    #pragma omp parallel for
    for (int i=0; i < params.size(); i++) {
        run_one(params[i], i, data_dir);
    }
}