#ifndef SINKHORN_HPP
#define SINKHORN_HPP

#include "../include_arma.hpp"

void sinkhorn(const arma::vec &weights_in, const arma::vec &weights_out,
              const arma::mat &cost, double eps,
              double threshold, int max_iter, int norm_p, arma::mat* transport,
              double* dist);

// arma::vec softmin(arma::mat C);

// double log_sum_exp(arma::vec x);

void stable_sinkhorn(const arma::vec &weights_in, const arma::vec &weights_out,
              const arma::mat &cost, double eps,
              double threshold, int max_iter, int norm_p, arma::mat* transport,
              double* dist);


void greenkhorn(const arma::vec &weights_in, const arma::vec &weights_out,
                const arma::mat &cost, double eps,
                double threshold, int max_iter, int norm_p, arma::mat* transport,
                double* dist, bool uniform=false);

#endif  // SINKHORN_HPP
