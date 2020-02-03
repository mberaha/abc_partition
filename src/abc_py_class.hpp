#ifndef ABC_PY_CLASS
#define ABC_PY_CLASS

#include "distance.hpp"

class AbcPy {
 protected:
     Distance* d;
     arma::mat part_results;
     arma::vec dist_results;
     double curr_dist;
     arma::vec part;
     arma::vec temp_part;
     arma::vec tvec;
     arma::vec uniq_temp;

     arma::mat data;
     arma::mat data_synt;

     // params
     double theta, sigma;
     int n_data;
     double eps0;

     arma::mat param;
     arma::mat tparam;
     int n_unique;
     arma::uvec sort_indices;
     arma::vec t_diff;
     double lEsum = 0;
     double lEsum2 = 0;
     double eps;
     double meanEps, meanEps2;

 public:
     ~AbcPy() {delete d;}

     AbcPy(arma::mat data, double theta, double sigma, double eps0,
           std::string distance);

     void updateUrn();

     virtual void updateParams() {}

     virtual void generateSyntData() {}

     void updatePartition();

     void step();

     std::tuple<arma::vec, arma::mat, double> run(int nrep);
};


class AbcPyUniv: public AbcPy{
 protected:
     // base measure parameters
     double a0, b0, k0, m0;

 public:
    AbcPyUniv(
        arma::vec data, double theta, double sigma, double eps0,
        double a0, double b0, double k0, double m0, std::string distance):
            AbcPy(data, theta, sigma, eps0, distance),
            a0(a0), b0(b0), k0(k0), m0(m0) {
        param.resize(1, 2);
        param(0, 0) = m0;
        param(0, 1) = b0 / (a0 - 1);
        std::cout << "PARAM: "; param.t().print();
        tparam = param;
    }

    void print() {
        std::cout << "theta: " << theta << ", sigma: " << sigma <<
                     ", eps0: " << eps0 << ", a0: " << a0 << ", b0: " <<
                     b0 << ", k0: " << k0 << ", m0: " << m0 << std::endl;
    }

    void updateParams();
    void generateSyntData();
};


class AbcPyMultiv: public AbcPy{
 protected:
     // base measure parameters
     double a0, b0, k0, m0;

 public:
    AbcPyMultiv(
        arma::vec data, double theta, double sigma, double eps0,
        double a0, double b0, double k0, double m0, std::string distance):
            AbcPy(data, theta, sigma, eps0, distance),
            a0(a0), b0(b0), k0(k0), m0(m0) {
        param.resize(1, 2);
        param(0, 0) = m0;
        param(0, 1) = b0 / (a0 - 1);
        std::cout << "PARAM: "; param.t().print();
        tparam = param;
    }

    void print() {
        std::cout << "theta: " << theta << ", sigma: " << sigma <<
                     ", eps0: " << eps0 << ", a0: " << a0 << ", b0: " <<
                     b0 << ", k0: " << k0 << ", m0: " << m0 << std::endl;
    }

    void updateParams();
    void generateSyntData();
};

#endif
