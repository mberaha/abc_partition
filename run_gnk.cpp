#include "../kernels.hpp"
#include "../distance.hpp"
#include "../abc_py_class.hpp"
#include <iostream>


std::vector<arma::vec> generate_data(
        int data_per_clus, gandk_param p1, gandk_param p2) {
    MultiGandKKernel kern(0.5);
    std::vector<arma::vec> data;
    for (int i = 0; i < data_per_clus; i++) {
        data.push_back(kern.rand_from_param(p1));
        data.push_back(kern.rand_from_param(p2));
    }
    return data;
}


int main() {    
    
   std::string dist = "sinkhorn";

   gandk_param param1 = {
        arma::vec({-3, -3}), 
        arma::vec({0.5, 0.5}),
        arma::vec({-0.9, -0.9}),
        arma::vec({0.1, 0.1}),
    };

     gandk_param param2 = {
        arma::vec({3, 3}), 
        arma::vec({0.5, 0.5}),
        arma::vec({0.9, 0.9}),
        arma::vec({0.1, 0.1}),
    };

    MultiGandKKernel kern(0.5);

    int nrep = 25;
    std::vector<int> data_per_clus = {50, 125};

    std::string out_dir = "./results/";
    std::vector<gandk_param> inits = kern.make_default_init();

    for (auto& dp : data_per_clus) {
	arma::vec times(nrep);

	#pragma omp parallel for
	for (int i = 0; i < nrep; i++) {
            std::vector<arma::vec> data = generate_data(dp, param1, param2);
                MultiGnKAbcPy abc(
                        data, 
                        inits, 
                        1.0, 0.1, 100, 0.5,
                        dist, kern);
                double time = abc.run(20000, 10000, false);
		times(i) = time;
                arma::imat parts = abc.get_parts();

                std::string outfile = out_dir + dist + "_" + \
                    std::to_string(i) + "_" + std::to_string(dp) + ".csv";
                
                parts.save(outfile, arma::csv_ascii);

        }
	std::string outfile_time = out_dir + "times_" + dist + "_" + std::to_string(dp) + ".csv";
	times.save(outfile_time, arma::csv_ascii);
    }
    return -1;
}
