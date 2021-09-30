#ifndef UTILS_HPP
#define UTILS_HPP

#include "include_arma.hpp"

arma::mat vstack(const std::vector<arma::vec> &rows);

std::vector<arma::vec> to_vectors(const arma::mat& mat); 

arma::mat pairwise_dist(const arma::mat &x, const arma::mat &y);

arma::mat pairwise_dist(const std::vector<double> &x, 
                        const std::vector<double> &y);

arma::mat pairwise_dist(const std::vector<arma::vec> &x,
                        const std::vector<arma::vec> &y);

void pairwise_dist_rowmajor(const std::vector<double> &x,
                            const std::vector<double> &y, double* out);

void pairwise_dist_rowmajor(const std::vector<arma::vec> &x,
                            const std::vector<arma::vec> &y, double* out);


#endif 