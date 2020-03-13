#ifndef ABC_PY_CLASS
#define ABC_PY_CLASS

#include <vector> 
#include "distance.hpp"
#include "distributions.hpp"
#include "graph.hpp"

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

     AbcPy() {}

     AbcPy(int n_data, double theta, double sigma,
           double eps0, std::string distance, int max_iter=100,
           double entropic_eps=0.1, double threshold=1e-4);

     void updateUrn();

     virtual void updateParams() = 0;

     virtual void generateSyntData() = 0;

     virtual void saveCurrParam() = 0;


     void step();

     std::tuple<arma::vec, arma::mat, double> run(int nrep);
};


class AbcPyUniv: public AbcPy{
 protected:
     arma::mat data;
     arma::mat data_synt;

     // base measure parameters
     double a0, b0, k0, m0;
     arma::mat param;
     arma::mat tparam;

 public:
    AbcPyUniv(
        const arma::mat &data_, double theta, double sigma, double eps0,
        double a0, double b0, double k0, double m0, std::string distance,
        int max_iter=100, double entropic_eps=0.1, double threshold=1e-4);

    void updateParams();
    void generateSyntData();
    void saveCurrParam() {param = tparam;}
};


class AbcPyMultiv: public AbcPy{
 protected:
     arma::mat data;
     arma::mat data_synt;

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
        const arma::mat &data_, double theta, double sigma, double eps0,
        double df, const arma::mat& prior_prec_chol,
        double k0, const arma::vec& m0, std::string distance,
        int max_iter=100, double entropic_eps=0.1, double threshold=1e-4);

    void print() {}

    void updateParams();
    void generateSyntData();
    void saveCurrParam() {
        mean = tmean;
        prec_chol = tprec_chol;
    }
};


// This is UGLY, should inherit from AbcPy
class AbcPyGraph {
 protected:
    std::vector<Graph> data;
    std::vector<Graph> data_synt;

    // GraphSimulator simulator;

    std::vector<arma::vec> param;
    std::vector<arma::vec> tparam;

    GraphSinkhorn* d;
    arma::mat part_results;
    arma::vec dist_results;
    double curr_dist;
    arma::vec part;
    arma::vec temp_part;
    arma::vec tvec;
    arma::vec uniq_temp;

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

    arma::vec m0;
    arma::mat prior_prec_chol;


public:
    AbcPyGraph() {}
    ~AbcPyGraph() {}

    AbcPyGraph(
        const std::vector<arma::mat> &data_, double theta, double sigma, 
        double eps0, const arma::mat &prior_prec_chol,
        const arma::vec &m0, std::string distance,
        int max_iter = 100, double entropic_eps = 0.1, double threshold = 1e-4);

    void updateUrn();

    void updateParams();
    
    void generateSyntData();

    void saveCurrParam() {
        param = tparam;
    }


    void step();

    std::tuple<arma::vec, arma::mat, double> run(int nrep);
};

#endif
