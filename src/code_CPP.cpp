#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

//----------------- ABC-MCMC for random partitions

// int sample_PY(arma::vec values,
//               double theta,
//               double sigma){
//   
//   arma::vec uniq = unique(values);
//   double t_bound = arma::randu() * (values.n_elem + theta);
//   int k_max = uniq.n_elem;
//   int k = -1;
//   double accu_val = 0.0;
//   
//   // loop
//   while(t_bound >= accu_val){
//     k += 1;
//     if(k == k_max){
//       break;
//     }
//     accu_val += arma::accu(values == uniq[k]) - sigma;
//   }
//   
//   // return the value
//   return k;
// }

//[[Rcpp::export]]
void update_part_PY(arma::vec &temp_part,
                    arma::vec part,
                    arma::mat param,
                    arma::mat &tparam,
                    double theta,
                    double sigma,
                    double m0,
                    double k0,
                    double a0,
                    double b0){

  // initialize
  int n = part.n_elem;
  arma::vec tvec(2 * n);
  tvec.fill(0.0);
  tvec.head(n) = part;
  int k = param.n_rows;
  
  arma::vec uniq = unique(part);
  int k_max = uniq.n_elem;
  arma::vec tfreqs(k_max + n);
  tfreqs.fill(0.0);
  
  for(arma::uword j = 0; j < k_max; j++){
    tfreqs(j) = arma::accu(part == uniq(j)) - sigma;
  }
  
  for(arma::uword i = n; i < 2 * n; i++){
    double t_bound = arma::randu() * (i + theta);
    int k = -1;
    double accu_val = 0.0;

    // loop
    while(t_bound >= accu_val){
      k += 1;
      if(k == k_max){
        break;
      }
      accu_val += tfreqs(k);
    }

    if(k < k_max){
      tfreqs(k) += 1;
      tvec(i) = k;
    } else {
      tfreqs(k) = 1 - sigma;
      tvec(i) = k;
      k_max += 1;
    }
  }
  
  // take the tail of the vector tvec
  temp_part = tvec.tail(n);
  arma::vec uniq_temp = unique(temp_part);
  tparam.resize(uniq_temp.n_elem, 2);
  
  // clean
  int indx = 0;
  for(arma::uword j = 0; j < uniq_temp.n_elem; j++){
    if(uniq_temp(j) < k){
      temp_part(arma::find(temp_part == uniq_temp(j))).fill(j);
      tparam(j,0) = param(uniq_temp(j), 0);
      tparam(j,1) = param(uniq_temp(j), 1);
    } else {
      temp_part(arma::find(temp_part == uniq_temp(j))).fill(j);
      tparam(j,1) =  1.0 / arma::randg(arma::distr_param(a0, 1.0 / b0));
      tparam(j,0) = arma::randn() * sqrt(tparam(j,1) / k0) + m0;
    }
  }
}
 
// [[Rcpp::export]]
Rcpp::List ABC_MCMC(arma::vec data,
                    int nrep,
                    double theta,
                    double sigma,
                    double m0,
                    double k0,
                    double a0,
                    double b0,
                    double eps,
                    int p){

  // initialize results
  arma::mat part_results(nrep, data.n_elem, arma::fill::zeros);
  arma::vec dist_results(nrep);

  arma::vec part(data.n_elem);
  arma::vec temp_part(data.n_elem);
  arma::vec data_synth(data.n_elem);
  arma::mat param(1, 2);
  arma::mat tparam(1, 2);
  arma::uvec sort_indices;
  arma::vec t_diff;

  // fill
  part.fill(0);
  temp_part.fill(0);
  param(0, 0) = m0;
  param(0, 1) = b0 / (a0 - 1);
  tparam = param;

  // time
  int start_s = clock();
  int current_s;

  // main loop
  for(arma::uword iter = 0; iter < nrep; iter++){

    update_part_PY(temp_part, part, param, tparam,
                   theta, sigma, m0, k0, a0, b0);

    // sample the synthetic data
    for(arma::uword j = 0; j < temp_part.n_elem; j++){
      data_synth(j) = arma::randn() *
        sqrt(tparam(temp_part(j),1)) + tparam(temp_part(j),0);
    }
    //////////////////////////////
    // prova simulando gruppo
    //////////////////////////////

    // save the results
    sort_indices = sort_index(data_synth);
    dist_results(iter) = arma::accu(pow(abs(data - data_synth(sort_indices)), p));
    part_results.row(iter) = temp_part(sort_indices).t();

    if(dist_results(iter) < eps){
      part = temp_part;
      param = tparam;
    }

    // check for interruption
    Rcpp::checkUserInterrupt();
  }
  int end_s = clock();

  // return the results
  Rcpp::List resu;
  resu["dist"] = dist_results;
  resu["part_results"] = part_results;
  resu["time"] = double(end_s-start_s)/CLOCKS_PER_SEC;
  return resu;
}
