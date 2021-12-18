// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace arma;

//----------------- ABC-MCMC for random partitions

double my_quantile(vec V, double p){
  vec temp = sort(V);
  return(temp(V.n_elem * (p - 0.000000001)));
}

//[[Rcpp::export]]
void update_part_PY(vec &temp_part,
                    vec part,
                    mat param,
                    mat &tparam,
                    double theta,
                    double sigma,
                    vec hyperparam){
  
  // initialize
  int n = part.n_elem;
  vec tvec(2 * n);
  tvec.head(n) = part;
  int k = param.n_rows;
  
  vec uniq = unique(part);
  int k_max = uniq.n_elem;
  vec tfreqs(k_max + n);
  tfreqs.fill(0.0);
  
  for(uword j = 0; j < k_max; j++){
    tfreqs(j) = accu(part == uniq(j)) - sigma;
  }
  
  for(uword i = n; i < 2 * n; i++){
    double t_bound = randu() * (i + theta);
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
  vec uniq_temp = unique(temp_part);
  tparam.resize(uniq_temp.n_elem, 2);
  
  // clean
  int indx = 0;
  for(uword j = 0; j < uniq_temp.n_elem; j++){
    if(uniq_temp(j) < k){
      temp_part(find(temp_part == uniq_temp(j))).fill(j);
      tparam(j,0) = param(uniq_temp(j), 0);
      tparam(j,1) = param(uniq_temp(j), 1);
    } else {
      temp_part(find(temp_part == uniq_temp(j))).fill(j);
      tparam(j,1) =  1.0 / randg(distr_param(hyperparam(2), 1.0 / hyperparam(3)));
      tparam(j,0) = randn() * sqrt(tparam(j,1) / hyperparam(1)) + hyperparam(0);
    }
  }
}
 

// [[Rcpp::export]]
Rcpp::List ABC_MCMCw(vec data,
                     int niter,
                     int nburn,
                     double theta,
                     double sigma,
                     vec hyperparam,
                     double eps,
                     int p){
  
  // initialize results
  mat part_results(niter - nburn, data.n_elem, fill::zeros);
  vec dist_results(niter - nburn);
  
  vec part(data.n_elem);
  vec temp_part(data.n_elem);
  vec data_synth(data.n_elem);
  mat param(1, 2);
  mat tparam(1, 2);
  uvec sort_indices;
  vec t_diff;
  
  double dist_temp;
  int n = data.n_elem;
  
  // fill
  part.fill(0);
  temp_part.fill(0);
  param(0, 0) = hyperparam(0);
  param(0, 1) = hyperparam(3) / (hyperparam(2) - 1);
  tparam = param;
  
  // time
  int start_s = clock();
  int current_s;
  
  // for while loop
  int n_accepted = 0;
  
  // main loop
  Rcpp::Rcout << "\nStarting the ABC-MCMC algorithm ...";
  while(n_accepted  < niter){
    
    update_part_PY(temp_part, part, param, tparam,
                   theta, sigma, hyperparam);
    
    for(uword j = 0; j < temp_part.n_elem; j++){
      data_synth(j) = randn() *
        sqrt(tparam(temp_part(j),1)) + tparam(temp_part(j),0);
    }
    
    // save the results
    sort_indices = sort_index(data_synth);
    dist_temp = pow(accu(pow(abs(data - data_synth(sort_indices)), p)), 1.0 / p);
    
    if(dist_temp < eps){
      part = temp_part;
      param = tparam;
      
      if(n_accepted >= nburn){
        dist_results(n_accepted - nburn) = dist_temp;
        part_results.row(n_accepted - nburn) = temp_part(sort_indices).t();
      }
      
      n_accepted += 1;
    }
    
    // check for interruption
    Rcpp::checkUserInterrupt();
    //Rcpp::Rcout << "\n" << n_accepted;
  }
  int end_s = clock();
  
  // return the results
  Rcpp::List resu;
  resu["dist"] = dist_results;
  resu["part_results"] = part_results;
  resu["time"] = double(end_s-start_s)/CLOCKS_PER_SEC;
  return resu;
}

// [[Rcpp::export]]
Rcpp::List adapt_ABC_MCMCw(vec data,
                           int niter,
                           int nburn,
                           double theta,
                           double sigma,
                           vec hyperparam,
                           double eps0,
                           double eps_star,
                           int p, 
                           int m, 
                           double tau1, 
                           double tau2,
                           bool adapt_fix = true){
  
  double eps = eps0;
  double lEsum = 0;
  double lEsum2 = 0;
  
  // initialize results
  mat part_results(niter - nburn, data.n_elem, fill::zeros);
  vec dist_results(niter - nburn);
  // vec dist_results2(1);
  vec temp_dist(m);
  int t_idx_dist = 0;
  
  vec part(data.n_elem);
  vec temp_part(data.n_elem);
  vec data_synth(data.n_elem);
  mat param(1, 2);
  mat tparam(1, 2);
  uvec sort_indices;
  vec t_diff;
  
  double dist_temp, beta1, beta2, beta3;
  
  // fill
  part.fill(0);
  temp_part.fill(0);
  param(0, 0) = hyperparam(0);
  param(0, 1) = hyperparam(3) / (hyperparam(2) - 1);
  tparam = param;
  
  // time
  int start_s = clock();
  int current_s;
  int n = data.n_elem;
  
  // for while loop
  int n_accepted = 0;
  int n_loop = 0;
  int n_loop_old;
  
  // main loop
  Rcpp::Rcout << "\nStarting the adaptive ABC-MCMC algorithm ...";
  while(n_accepted  < niter){
    
    update_part_PY(temp_part, part, param, tparam,
                   theta, sigma, hyperparam);
    
    // sample the synthetic data
    for(uword j = 0; j < temp_part.n_elem; j++){
      data_synth(j) = randn() *
        sqrt(tparam(temp_part(j),1)) + tparam(temp_part(j),0);
    }
    
    sort_indices = sort_index(data_synth);
    dist_temp = pow(accu(pow(abs(data - data_synth(sort_indices)), p)), 1.0 / p);
    
    if(adapt_fix){
      if(n_accepted < nburn){
        n_loop += 1;
        lEsum = 0.0;
        if(dist_temp < eps_star){
          lEsum = 1.0;
        }
        eps_star *= exp(pow(n_loop, (-2 / 3)) * (0.05 - lEsum));
        eps = eps_star;
        n_loop_old = n_loop;
      } else {
        temp_dist(t_idx_dist) = dist_temp;
        if(t_idx_dist == m - 1){
          t_idx_dist = 0;
        } else {
          t_idx_dist += 1;
        }
        
        n_loop += 1;
        beta1 = exp(- (n_loop - n_loop_old + 1) / tau1);
        beta2 = exp(- (n_loop - n_loop_old + 1) / tau2) * (1 - beta1);
        beta3 = 1 - beta1 - beta2;
        if(n_loop < m){
          beta2 = 0;
        }
        eps = beta1 * eps0 + beta2 * my_quantile(temp_dist, 0.01) + beta3 * eps_star;
      }
    } else {
      n_loop += 1;
      lEsum = 0.0;
      if(dist_temp < eps_star){
        lEsum = 1.0;
      }
      eps_star *= exp(pow(n_loop, (-2 / 3)) * (0.05 - lEsum));
      eps = eps_star;
      n_loop_old = n_loop;
    }
    
    if(dist_temp < eps){
      part = temp_part;
      param = tparam;
      
      if(n_accepted >= nburn){
        dist_results(n_accepted - nburn) = dist_temp;
        part_results.row(n_accepted - nburn) = temp_part(sort_indices).t();
      }
      
      n_accepted += 1;
    }
    
    // check for interruption
    Rcpp::checkUserInterrupt();
  }
  Rcpp::Rcout << "final eps = " << eps << "\n";
  int end_s = clock();
  
  // return the results
  Rcpp::List resu;
  resu["dist"] = dist_results;
  resu["part_results"] = part_results;
  resu["time"] = double(end_s-start_s)/CLOCKS_PER_SEC;
  resu["final_eps"] = eps;
  return resu;
}

//----------------

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