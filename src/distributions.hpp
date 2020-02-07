#ifndef DISTRIBUTIONS_HPP
#define DISTRIBUTIONS_HPP

#include "include_arma.hpp"

arma::mat rwishart(unsigned int df, const arma::mat& chol_S);

arma::mat rwishart_chol(unsigned int df, const arma::mat& chol_S);

arma::vec rnorm_prec_chol(const arma::vec& mean, const arma::mat& chol_prec);

#endif
