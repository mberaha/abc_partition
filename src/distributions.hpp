#ifndef DISTRIBUTIONS_HPP
#define DISTRIBUTIONS_HPP

#include "include_arma.hpp"
#include "kernels.hpp"

arma::mat rwishart(unsigned int df, const arma::mat& chol_S);

arma::mat rwishart_chol(unsigned int df, const arma::mat& chol_S);

arma::vec rnorm_prec_chol(const arma::vec& mean, const arma::mat& chol_prec);

arma::vec rnorm_chol(const arma::vec& mean, const arma::mat& cov_chol);

// arma::vec rgandk_biv(double rho, gandk_param param);

#endif
