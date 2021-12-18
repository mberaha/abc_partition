// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace arma;

/*
 * sample an integer
 */

int rint(vec weights){
  
  double u = randu();
  vec probs;
  
  probs = weights;
  probs /= sum(probs);
  probs = cumsum(probs);
  
  for(uword k = 0; k < probs.n_elem; k++) {
    if(u <= probs[k]) {
      return k;
    }
  }
  return -1;
}

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

/*
 * quantile with correction
 */

double my_quantile(arma::vec V, double p){
  arma::vec temp = sort(V);
  return(temp(V.n_elem * (p - 0.000000001)));
}

/*
 * functions for the g&k distribution, in order
 *  - the quantile as function of a gaussian distribution
 *  - its derivative
 *  - the derivative of the target function (Q(z) - x)^2 w.r.t. z
 *  - Newton-Raphson optimization for the target function
 *  - density function
 */

//[[Rcpp::export]]
double Qnk(double z, double a, double b, double g, double k, double c){
  /*
  * Q(z) of the g-and-k distribution - Z distributed as a N(0,1)
  */
  double out = a + b * (1 + c * tanh(g * z / 2.0)) * z * pow(1 + z * z, k);
  return out; 
}

//[[Rcpp::export]]
double deriv_Qnk(double z, double a, double b, double g, double k, double c){
  /*
  * derivative of the quantile of the g-and-k distribution
  */
  double out = b * pow(1.0 + z * z, k) *
    ((1.0 + c * tanh(g * z/2.0)) *
    ((1.0 + (2.0 * k + 1.0) * z*z) / (1.0 + z*z)) +
    (c*g*z / (2.0 * pow(cosh(g*z/2.0), 2.0))));
  return(out);
}

//[[Rcpp::export]]
double deriv_QnkX(double x, double z, double a, double b, double g, double k, double c){
  /*
   * derivative of the target function
   */
  double out = (b * pow(1.0 + z * z, k) *
    ((1.0 + c * tanh(g * z/2.0)) *
    ((1.0 + (2.0 * k + 1.0) * z*z) / (1.0 + z*z)) +
    (c*g*z / (2.0 * pow(cosh(g*z/2.0), 2.0))))) * 2 * (Qnk(z, a, b, g, k, c) - x);
  return(out);
}

//[[Rcpp::export]]
double optimize_Qnk(double x, double a, double b, double g, double k, double c,
            double tol = 0.00001, int max_iter = 1000){
              /*
              * find Z such that (Q(z) - x)^2 = 0
              */
  double z = a; 
  double eps = 0.001; 
  double z_old;

  // start loop
  int iter = 1;
  while(pow(Qnk(z_old, a, b, g, k, c) - x, 2) > tol && iter < max_iter){
    z_old = z; 
    z = z_old - pow(Qnk(z_old, a, b, g, k, c) - x, 2) / deriv_QnkX(x, z_old, a, b, g, k, c);
    iter += 1;
  }  
  return z;
}

//[[Rcpp::export]]
double PDFkn(double x, double a, double b, double g, double k, double c,
             double tol = 0.00001, int max_iter = 1000){
  double z = optimize_Qnk(x, a, b, g, k, c, tol, max_iter);
  double out = normpdf(z) / deriv_Qnk(z, a, b, g, k, c);
  return out;
}

/*
 * ABC sampling scheme for the g&k distribution
 * in the following functions: 
 *  - update the partition of a PY process (in an efficient way)
 *  - ABC sampling scheme
 *  - adaptive ABC sampling scheme
 */

//[[Rcpp::export]]
void update_part_PY(arma::vec &temp_part,
                    arma::vec part,
                    arma::mat param,
                    arma::mat &tparam,
                    double theta,
                    double sigma,
                    vec hyperparam){

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
  tparam.resize(uniq_temp.n_elem, 4);

  // clean
  int indx = 0;
  for(arma::uword j = 0; j < uniq_temp.n_elem; j++){
    if(uniq_temp(j) < k){
      temp_part(arma::find(temp_part == uniq_temp(j))).fill(j);
      tparam(j,0) = param(uniq_temp(j), 0);
      tparam(j,1) = param(uniq_temp(j), 1);
      tparam(j,2) = param(uniq_temp(j), 2);
      tparam(j,3) = param(uniq_temp(j), 3);
    } else {
      temp_part(arma::find(temp_part == uniq_temp(j))).fill(j);
      tparam(j,1) =  1.0 / arma::randg(arma::distr_param(hyperparam(2), 1.0 / hyperparam(3)));
      tparam(j,0) = arma::randn() * hyperparam(1) + hyperparam(0);
      tparam(j,2) = arma::randn() * hyperparam(5) + hyperparam(4);
      tparam(j,3) = 1.0 / arma::randg(arma::distr_param(hyperparam(6), 1.0 / hyperparam(7)));
    }
  }
}
 

// [[Rcpp::export]]
Rcpp::List ABC_MCMC_gnk(arma::vec data,
                        int niter,
                        int nburn,
                        double theta,
                        double sigma,
                        vec hyperparam,
                        double eps,
                        int p){
  
  // initialize results
  arma::mat part_results(niter - nburn, data.n_elem, arma::fill::zeros);
  arma::vec dist_results(niter - nburn);
  
  arma::vec part(data.n_elem);
  arma::vec temp_part(data.n_elem);
  arma::vec data_synth(data.n_elem);
  arma::mat param(1, 4);
  arma::mat tparam(1, 4);
  arma::uvec sort_indices;
  arma::vec t_diff;
  
  double dist_temp;
  
  // fill
  part.fill(0);
  temp_part.fill(0);
  param(0, 0) = hyperparam(0);
  param(0, 1) = hyperparam(3) / (hyperparam(2) - 1);
  param(0, 2) = hyperparam(4);
  param(0, 3) = hyperparam(7) / (hyperparam(6) - 1);
  tparam = param;
  
  // time
  int start_s = clock();
  int current_s;
  
  // for while loop
  int n_accepted = 0;
  
  // main loop
  while(n_accepted  < niter){
    
    update_part_PY(temp_part, part, param, tparam,
                   theta, sigma,hyperparam);
    
    // sample the synthetic data
    for(arma::uword j = 0; j < temp_part.n_elem; j++){
      data_synth(j) = Qnk(arma::randn(), tparam(temp_part(j),0), tparam(temp_part(j),1), 
                 tparam(temp_part(j),2), tparam(temp_part(j),3), 0.8);
    }
    
    // save the results
    sort_indices = sort_index(data_synth);
    dist_temp = pow(arma::accu(pow(abs(data - data_synth(sort_indices)), p)), 1.0 / p);
    
    if(dist_temp < eps){
      part = temp_part;
      param = tparam;
      
      if(n_accepted >= nburn){
        dist_results(n_accepted - nburn) = dist_temp;
        part_results.row(n_accepted - nburn) = temp_part(sort_indices).t();
      }
      
      n_accepted += 1;
      
      if((n_accepted + 1) % 100 == 0){
        current_s = clock();
        Rcpp::Rcout << "Completed:\t" << (n_accepted + 1) << "/" << niter << " - in " <<
          double(current_s-start_s)/CLOCKS_PER_SEC << " sec\n";
      }
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

// [[Rcpp::export]]
Rcpp::List adapt_ABC_MCMC_gnk(arma::vec data,
                              int niter,
                              int nburn,
                              double theta,
                              double sigma,
                              double eps0,
                              double eps_star,
                              int p,
                              int m,
                              double tau1,
                              double tau2, 
                              vec hyperparam, 
                              int nupd = 100,
                              bool adapt_fix = true){
  
  double eps = eps0;
  double lEsum = 0;
  double lEsum2 = 0;
  
  // initialize results
  arma::mat part_results(niter - nburn, data.n_elem, arma::fill::zeros);
  arma::vec dist_results(niter - nburn);
  // arma::vec dist_results2(1);
  arma::vec temp_dist(m);
  int t_idx_dist = 0;
  
  arma::vec part(data.n_elem);
  arma::vec temp_part(data.n_elem);
  arma::vec data_synth(data.n_elem);
  arma::mat param(1, 4);
  arma::mat tparam(1, 4);
  arma::uvec sort_indices;
  arma::vec t_diff;
  
  double dist_temp, beta1, beta2, beta3;
  
  // fill
  part.fill(0);
  temp_part.fill(0);
  param(0, 0) = hyperparam(0);
  param(0, 1) = hyperparam(3) / (hyperparam(2) - 1);
  param(0, 2) = hyperparam(4);
  param(0, 3) = hyperparam(7) / (hyperparam(6) - 1);
  tparam = param;
  
  // time
  int start_s = clock();
  int current_s;
  
  // for while loop
  int n_accepted = 0;
  int n_loop = 0;
  int n_loop_old;
  // main loop
  while(n_accepted  < niter){
    update_part_PY(temp_part, part, param, tparam,
                   theta, sigma, hyperparam);
    
    // sample the synthetic data
    for(arma::uword j = 0; j < temp_part.n_elem; j++){
      data_synth(j) = Qnk(arma::randn(), tparam(temp_part(j),0), tparam(temp_part(j),1), 
                 tparam(temp_part(j),2), tparam(temp_part(j),3), 0.8);
    }
    
    // save the results
    sort_indices = sort_index(data_synth);
    dist_temp = pow(arma::accu(pow(abs(data - data_synth(sort_indices)), p)), 1.0 / p);
    
    // Rcpp::Rcout << dist_temp << "\t" << eps << "\t" << eps_star << "\t" << n_accepted << "\n\n";
    
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
    
    // Rcpp::Rcout << eps << "\n\n";
    
    if(dist_temp < eps){
      part = temp_part;
      param = tparam;
      
      if(n_accepted >= nburn){
        dist_results(n_accepted - nburn) = dist_temp;
        part_results.row(n_accepted - nburn) = temp_part(sort_indices).t();
      }
      
      n_accepted += 1;
      
      if((n_accepted + 1) % nupd == 0){
        current_s = clock();
        Rcpp::Rcout << "Completed:\t" << (n_accepted + 1) << "/" << niter << " - in " <<
          double(current_s-start_s)/CLOCKS_PER_SEC << " sec\n";
      }
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

/*
 * marginal sampling scheme (Neal 8 algorithm) for the g&k distribution
 * in the following functions: 
 *  - clean parameters
 *  - update cluster allocation
 *  - acceleration step
 *  - 
 */

void clean_param_gnk(vec &clust, 
                     mat &param){
  int k = param.n_rows;
  int u_bound;
  
  // for all the used parameters
  for(arma::uword i = 0; i < k; i++){
    
    // if a cluster is empty
    if((int) arma::sum(clust == i) == 0){
      
      // find the last full cluster, then swap
      for(arma::uword j = k; j > i; j--){
        if((int) arma::sum(clust == j) != 0){
          
          // SWAPPING!!
          clust( arma::find(clust == j) ).fill(i);
          param.swap_rows(i,j);
          break;
        }
      }
    }
  }
  
  // reduce dimensions
  u_bound = 0;
  for(arma::uword i = 0; i < k; i++){
    if(arma::accu(clust == i) > 0){
      u_bound += 1;
    }
  }
  
  // resize object to the correct dimension
  param.resize(u_bound,param.n_cols);
}

void update_cluster_allocation_gnk(vec data, 
                                   vec &clust, 
                                   mat &param,
                                   int m, 
                                   double mass,
                                   double sigma_PY, 
                                   vec hyperparam){
  // initialize quantities
  int n = clust.n_elem;
  int d = data.n_cols;
  int k, temp_idx, temp_clust;
  vec probs;
  mat tparam(m, 4);
  vec m1vec(m); 
  m1vec.fill(1.0);
  
  for(uword j = 0; j < m; j++){
    tparam(j,0) = arma::randn() * hyperparam(1) + hyperparam(0);
    tparam(j,1) =  1.0 / arma::randg(arma::distr_param(hyperparam(2), 1.0 / hyperparam(3)));
    tparam(j,2) = arma::randn() * hyperparam(5) + hyperparam(4);
    tparam(j,3) = 1.0 / arma::randg(arma::distr_param(hyperparam(6), 1.0 / hyperparam(7)));
  }
  
  for(uword i = 0; i < n; i++){
    
    // Rcpp::Rcout << "qui_in_2\t" << i << "\n"; 
    
    bool req_clean = false;
    if(arma::sum(clust == clust[i]) == 1){
      req_clean = true;
    }
    
    // Rcpp::Rcout << "qui_in_2_A\t" << tparam.n_rows << "\t" << param.n_rows << "\t" << clust(i) << "\n"; 
    
    if(req_clean){
      temp_idx = rint(m1vec);
      tparam.row(temp_idx) = param.row(clust(i));
    }
    
    // Rcpp::Rcout << "qui_in_2_B\t" << i << "\n"; 
    
    clust(i) = param.n_rows + 1;
    if(req_clean){
      clean_param_gnk(clust, param);
    }
    // Rcpp::Rcout << "qui_in_2_bis\t" << i << "\n"; 
    
    // initialize useful quantities
    k = param.n_rows;
    probs.resize(k+m);
    probs.fill(0);
    
    for(uword j = 0; j < k; j++){
      probs(j) = (accu(clust == j) - sigma_PY) * 
        PDFkn(data(i), param(j,0), param(j,1), param(j,2), param(j,3), 0.8);
    }
    // Rcpp::Rcout << "qui_in_2_tris\t" << i << "\n"; 
    for(uword j = 0; j < m; j++){
      probs(k + j) = (mass + k * sigma_PY) / m  * 
        PDFkn(data(i), tparam(j,0), tparam(j,1), tparam(j,2), tparam(j,3), 0.8);
    }
    probs.elem(find_nonfinite(probs)).zeros();
    // Rcpp::Rcout << "qui_in_2\n" << clust.t() << "\n\n" << probs.t() <<"\n";
    temp_clust = rint(probs);
    // Rcpp::Rcout << "qui_in_3\t" << temp_clust << "\n";
    if(temp_clust > k - 1){
      clust(i) = k;
      param.resize(k+1, 4);
      param.row(k) = tparam.row(temp_clust - k);
      tparam(temp_clust - k,0) = arma::randn() * hyperparam(1) + hyperparam(0);
      tparam(temp_clust - k,1) =  1.0 / arma::randg(arma::distr_param(hyperparam(2), 1.0 / hyperparam(3)));
      tparam(temp_clust - k,2) = arma::randn() * hyperparam(5) + hyperparam(4);
      tparam(temp_clust - k,3) = 1.0 / arma::randg(arma::distr_param(hyperparam(6), 1.0 / hyperparam(7)));
    } else {
      clust(i) = temp_clust;
    }
    // Rcpp::Rcout << "qui_in_3\t" << temp_clust << "\n\n";
  }
}

// [[Rcpp::export]]
Rcpp::List marginal_gnk(arma::vec data,
                        int niter,
                        int nburn,
                        double theta,
                        double sigma,
                        int m, 
                        vec hyperparam, 
                        int nupd = 100){
  
  // initialize results
  arma::mat part_results(niter - nburn, data.n_elem, arma::fill::zeros);
  arma::vec clust(data.n_elem);
  arma::mat param(1, 4);
  
  // fill
  clust.fill(0);
  param(0, 0) = hyperparam(0);
  param(0, 1) = hyperparam(3) / (hyperparam(2) - 1);
  param(0, 2) = hyperparam(4);
  param(0, 3) = hyperparam(7) / (hyperparam(6) - 1);
  
  // time
  int start_s = clock();
  int current_s;
  
  // main loop
  for(uword iter = 0; iter < niter; iter ++){
    update_cluster_allocation_gnk(data, clust, param, m, theta, sigma, hyperparam);
    clean_param_gnk(clust, param);
    
    if(iter >= nburn){
      part_results.row(iter - nburn) = clust.t();
    }
    
    if((iter + 1) % nupd == 0){
      current_s = clock();
      Rcpp::Rcout << "Completed:\t" << (iter + 1) << "/" << niter << " - in " <<
        double(current_s-start_s)/CLOCKS_PER_SEC << " sec\n";
    }
    // check for interruption
    Rcpp::checkUserInterrupt();
  }
  int end_s = clock();
  
  // return the results
  Rcpp::List resu;
  resu["part_results"] = part_results;
  resu["time"] = double(end_s-start_s)/CLOCKS_PER_SEC;
  return resu;
}