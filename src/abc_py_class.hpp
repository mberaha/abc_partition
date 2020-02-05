#ifndef ABC_PY_CLASS
#define ABC_PY_CLASS

#include "distance.hpp"
#include "distributions.hpp"

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

     virtual void saveCurrParam() {}

     void updatePartition();

     void step();

     std::tuple<arma::vec, arma::mat, double> run(int nrep);
};


class AbcPyUniv: public AbcPy{
 protected:
     // base measure parameters
     double a0, b0, k0, m0;
     arma::mat param;
     arma::mat tparam;

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
    void saveCurrParam() {param = tparam;}
};


class AbcPyMultiv: public AbcPy{
 protected:
     // base measure parameters
     double df, k0;
     arma::vec m0;
     arma::mat prior_prec_chol;

     std::vector<arma::vec> mean;
     std::vector<arma::vec> tmean;
     std::vector<arma::mat> prec_chol;
     std::vector<arma::mat> tprec_chol;

 public:
    AbcPyMultiv(
        const arma::mat& data, double theta, double sigma, double eps0,
        double df, const arma::mat& prior_prec_chol,
        double k0, const arma::vec& m0, std::string distance):
            AbcPy(data, theta, sigma, eps0, distance),
            df(df), k0(k0), m0(m0), prior_prec_chol(prior_prec_chol) {

        mean.reserve(1000);
        tmean.reserve(1000);
        prec_chol.reserve(1000);
        tprec_chol.reserve(1000);

        mean.push_back(m0);
        tmean.push_back(m0);

        prec_chol.push_back(
            arma::mat(prior_prec_chol.n_rows, prior_prec_chol.n_rows,
                      arma::fill::eye));

        tprec_chol.push_back(
            arma::mat(prior_prec_chol.n_rows, prior_prec_chol.n_rows,
                      arma::fill::eye));
    }

    void print() {}

    void updateParams();
    void generateSyntData();
    void saveCurrParam() {
        mean = tmean;
        prec_chol = tprec_chol;
    }
};

#endif
