#include "distributions.hpp"

arma::mat rwishart(unsigned int df, const arma::mat& chol_S){
  arma::mat C = rwishart_chol(df, chol_S);

  // Return random wishart
  return C.t()*C;
}

arma::mat rwishart_chol(unsigned int df, const arma::mat& chol_S){
  // Dimension of returned wishart
  unsigned int m = chol_S.n_rows;

  // Z composition:
  // sqrt chisqs on diagonal
  // random normals below diagonal
  // misc above diagonal
  arma::mat Z(m,m);

  // Fill the diagonal
  for(unsigned int i = 0; i < m; i++){
    Z(i,i) = sqrt(arma::chi2rnd(df-i));
  }

  // Fill the lower matrix with random guesses
  for(unsigned int j = 0; j < m; j++){
    for(unsigned int i = j+1; i < m; i++){
      Z(i,j) = arma::randu();
    }
  }

  // Lower triangle * chol decomp
  arma::mat C = arma::trimatl(Z).t() * chol_S;

  return C;
}

arma::vec rnorm_prec_chol(const arma::vec& mean, const arma::mat& chol_prec) {
    arma::vec z(mean.n_elem, arma::fill::randn);
    return mean + arma::solve(chol_prec, z);
}
