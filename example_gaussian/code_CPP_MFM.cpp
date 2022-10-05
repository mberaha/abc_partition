// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace arma;

//----------------- ABC-MCMC for random partitions

double my_quantile(vec V, double p){
  vec temp = sort(V);
  return(temp(V.n_elem * (p - 0.000000001)));
}

vec initialize_part(int n){
  vec result(n);
  double r_mass;
  int out_temp;
  
  for(uword i = 0; i < n; i++){
    r_mass = randu() * n;
    out_temp = 0;
    while(out_temp < r_mass){
      out_temp += 1;
    }
    result(i) = out_temp - 1;
  }
  return(result);
}

//-----------------------------------------------------------------
// clean an univariate location-scale allocation,
// by destroying all the empty clusters

void part_clean_ABC(vec clust){
  int k = max(clust) + 1;
  
  // the loop starts from 1, 'cause the cluster 0 denotes the singletons
  for(arma::uword i = 0; i < k; i++){
    
    if((int) arma::sum(clust == i) == 0){
      for(arma::uword j = k; j > i; j--){
        if((int) arma::sum(clust == j) != 0){
          
          clust( arma::find(clust == j) ).fill(i);
          break;
        }
      }
    }
  }
}

void update_part_MFM(vec &temp_part,
                     vec part,
                     mat param,
                     mat &tparam,
                     double gamma,
                     mat Vnt_ABC,
                     vec hyperparam){
  
  // initialize
  int n = part.n_elem;
  vec tvec(2 * n);
  tvec.head(n) = part;
  int k_old = param.n_rows;
  int k;
  
  vec uniq = unique(part);
  int k_max = uniq.n_elem;
  vec tfreqs(k_max + n);
  tfreqs.fill(0.0);
  
  double accu_val;
  double t_bound;
  
  for(uword j = 0; j < k_max; j++){
    tfreqs(j) = accu(part == uniq(j)) + gamma;
  }
  
  
  for(uword i = n; i < (2 * n); i++){
    t_bound = randu() * (i + k_max * gamma + Vnt_ABC(i - n, k_max - 1) * gamma);
    k = -1;
    accu_val = 0.0;
    
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
      tfreqs(k) = 1 + gamma;
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
    if(uniq_temp(j) < k_old){
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
Rcpp::List ABC_MCMCw_MFM(vec data,
                     int niter,
                     int nburn,
                     double gamma,
                     mat Vnt_ABC,
                     vec hyperparam,
                     double eps,
                     int p){
  
  // initialize results
  mat part_results(niter - nburn, data.n_elem, fill::zeros);
  vec dist_results(niter - nburn);
  
  vec part(data.n_elem);
  vec temp_part(data.n_elem);
  vec data_synth(data.n_elem);
  mat param(data.n_elem, 2);
  mat tparam(data.n_elem, 2);
  uvec sort_indices;
  vec t_diff;
  
  double dist_temp;
  int n = data.n_elem;
  
  // fill
  part = initialize_part(data.n_elem);
  part_clean_ABC(part);
  temp_part = part;
  
  param.col(0).fill(hyperparam(0));
  param.col(1).fill(hyperparam(3) / (hyperparam(2) - 1));
  param.resize(max(part) + 1, 2);
  tparam = param;
  
  // time
  double start_s = clock();
  double end_s;
  int current_s;
  
  // for while loop
  int n_accepted = 0;
  start_s = clock(); 
  
  // main loop
  Rcpp::Rcout << "\nStarting the ABC-MCMC algorithm ...";
  while(n_accepted  < niter){
    
    update_part_MFM(temp_part, part, param, tparam,
                    gamma, Vnt_ABC, hyperparam);
    
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
  }
  end_s = clock();  
  
  // return the results
  Rcpp::List resu;
  resu["dist"] = dist_results;
  resu["part_results"] = part_results;
  resu["time"] = double(end_s-start_s)/CLOCKS_PER_SEC;
  return resu;
}

// [[Rcpp::export]]
Rcpp::List adapt_ABC_MCMCw_MFM(vec data,
                           int niter,
                           int nburn,
                           double gamma,
                           mat Vnt_ABC,
                           vec hyperparam,
                           double eps0,
                           double eps_star,
                           int p, 
                           bool adapt_fix = true){
  
  double eps = eps0;
  double lEsum = 0;
  double lEsum2 = 0;
  
  // initialize results
  mat part_results(niter - nburn, data.n_elem, fill::zeros);
  vec dist_results(niter - nburn);
  
  int t_idx_dist = 0;
  
  vec part(data.n_elem);
  vec temp_part(data.n_elem);
  vec data_synth(data.n_elem);
  mat param(data.n_elem, 2);
  mat tparam(data.n_elem, 2);
  uvec sort_indices;
  vec t_diff;
  
  double dist_temp;

  part.fill(0);
  temp_part = part;

  param.col(0).fill(hyperparam(0));
  param.col(1).fill(hyperparam(3) / (hyperparam(2) - 1));
  param.resize(1, 2);
  tparam = param;
  
  
  // time
  double start_s = clock();
  double end_s;
  int current_s;
  int n = data.n_elem;
  
  // for while loop
  int n_accepted = 0;
  int n_loop = 0;
  int n_loop_old;
  
  // main loop
  Rcpp::Rcout << "\nStarting the adaptive ABC-MCMC algorithm ...";
  while(n_accepted  < niter){
    
    update_part_MFM(temp_part, part, param, tparam,
                    gamma, Vnt_ABC, hyperparam);
    
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
  end_s = clock();
  
  // return the results
  Rcpp::List resu;
  resu["dist"] = dist_results;
  resu["part_results"] = part_results;
  resu["time"] = double(end_s-start_s)/CLOCKS_PER_SEC;
  resu["final_eps"] = eps;
  return resu;
}
