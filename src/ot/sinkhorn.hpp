#ifndef SINKHORN_HPP
#define SINKHORN_HPP

#include <armadillo>

void sinkhorn(arma::vec weights_in, arma::vec weights_out,
              arma::mat cost, double eps,
              double threshold, int max_iter, int norm_p, arma::mat* transport,
              double* dist);

#endif  // SINKHORN_HPP
