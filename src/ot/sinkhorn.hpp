#ifndef SINKHORN_HPP
#define SINKHORN_HPP

#include <armadillo>

void sinkhorn(const arma::vec &weights_in, const arma::vec &weights_out,
              const arma::mat &cost, double eps,
              double threshold, int max_iter, int norm_p, arma::mat* transport,
              double* dist);


void greenkhorn(const arma::vec &weights_in, const arma::vec &weights_out,
                const arma::mat &cost, double eps,
                double threshold, int max_iter, int norm_p, arma::mat* transport,
                double* dist);

#endif  // SINKHORN_HPP
