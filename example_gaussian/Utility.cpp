// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace arma;

/*
 * compute the distances among partitions
 */

// [[Rcpp::export]]
arma::mat compute_psm(arma::mat partition){
  int n = partition.n_cols;
  arma::mat PSM(n,n);
  
  for(arma::uword i = 0; i < n; i++){
    for(arma::uword j = 0; j <= i; j++){
      PSM(i,j) = ((double) arma::accu(partition.col(i) == partition.col(j)));
      PSM(j,i) = PSM(i,j);
    }
  }
  
  PSM /= ((double) partition.n_rows);
  return PSM;
}

// [[Rcpp::export]]
arma::vec compute_dist(arma::mat partition, arma::mat PSM){
  int n = partition.n_cols;
  arma::mat temp_mat = PSM;
  arma::vec dists(partition.n_rows);
  
  for(arma::uword k = 0; k < partition.n_rows; k++){
    temp_mat.fill(0.0);
    for(arma::uword i = 0; i < n; i++){
      for(arma::uword j = i; j < n; j++){
        temp_mat(i,j) = (double) (partition(k,i) == partition (k,j));
        temp_mat(j,i) = temp_mat(i,j);
      }
    }
    dists(k) = arma::accu(abs(PSM - temp_mat));
  }
  return dists;
}

//[[Rcpp::export]]
arma::vec VI_LB(arma::mat C_mat, arma::mat PSM){
  
  arma::vec result(C_mat.n_rows);
  double f = 0.0;
  int n = PSM.n_cols;
  arma::vec tvec(n);
  
  for(arma::uword j = 0; j < C_mat.n_rows; j++){
    f = 0.0;
    for(arma::uword i = 0; i < n; i++){
      tvec = PSM.col(i);
      f += (log2(arma::accu(C_mat.row(j) == C_mat(j,i))) +
        log2(arma::accu(tvec)) -
        2 * log2(arma::accu(tvec.elem(arma::find(C_mat.row(j).t() == C_mat(j,i))))))/n;
    }
    result(j) = f;
    Rcpp::checkUserInterrupt();
  }
  return(result);
}