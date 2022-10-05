// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace arma;

//----------------- ABC-MCMC for random partitions

double my_quantile(vec V, double p){
  vec temp = sort(V);
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

// [[Rcpp::export]]
double Qnk(double z, double a, double b, double g, double k, double c){
  /*
   * Q(z) of the g-and-k distribution - Z distributed as a N(0,1)
   */
  double out = a + b * (1 + c * tanh(g * z / 2.0)) * z * pow(1 + z * z, k);
  return out; 
}

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

double PDFkn(double x, double a, double b, double g, double k, double c,
             double tol = 0.00001, int max_iter = 1000){
  double z = optimize_Qnk(x, a, b, g, k, c, tol, max_iter);
  double out = normpdf(z) / deriv_Qnk(z, a, b, g, k, c);
  return out;
}

// Function for the sampler
/*
 * ABC sampling scheme for the g&k distribution
 */

void update_part_MFM_gnk(vec &temp_part,
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
  tparam.resize(uniq_temp.n_elem, 4);
  
  // clean
  int indx = 0;
  for(arma::uword j = 0; j < uniq_temp.n_elem; j++){
    if(uniq_temp(j) < k_old){
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
Rcpp::List ABC_MCMCw_MFM_gnk(vec data,
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
  mat param(1, 4);
  mat tparam(1, 4);
  uvec sort_indices;
  vec t_diff;
  
  double dist_temp;
  int n = data.n_elem;
  
  // fill
  part.fill(0);
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
  Rcpp::Rcout << "\nStarting the ABC-MCMC algorithm ...";
  while(n_accepted  < niter){
    
    update_part_MFM_gnk(temp_part, part, param, tparam,
                   gamma, Vnt_ABC, hyperparam);
    
    // sample the synthetic data
    for(arma::uword j = 0; j < temp_part.n_elem; j++){
      data_synth(j) = Qnk(arma::randn(), tparam(temp_part(j),0), tparam(temp_part(j),1), 
                 tparam(temp_part(j),2), tparam(temp_part(j),3), 0.8);
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
  int end_s = clock();
  
  // return the results
  Rcpp::List resu;
  resu["dist"] = dist_results;
  resu["part_results"] = part_results;
  resu["time"] = double(end_s-start_s)/CLOCKS_PER_SEC;
  return resu;
}

// [[Rcpp::export]]
Rcpp::List adapt_ABC_MCMCw_MFM_gnk(vec data,
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
  mat param(1, 4);
  mat tparam(1, 4);
  uvec sort_indices;
  vec t_diff;
  
  double dist_temp;
  
  // fill
  part.fill(0);
  param(0, 0) = hyperparam(0);
  param(0, 1) = hyperparam(3) / (hyperparam(2) - 1);
  param(0, 2) = hyperparam(4);
  param(0, 3) = hyperparam(7) / (hyperparam(6) - 1);
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

    update_part_MFM_gnk(temp_part, part, param, tparam,
                   gamma, Vnt_ABC, hyperparam);
    
    for(arma::uword j = 0; j < temp_part.n_elem; j++){
      data_synth(j) = Qnk(arma::randn(), tparam(temp_part(j),0), tparam(temp_part(j),1), 
                 tparam(temp_part(j),2), tparam(temp_part(j),3), 0.8);
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
  int end_s = clock();
  
  // return the results
  Rcpp::List resu;
  resu["dist"] = dist_results;
  resu["part_results"] = part_results;
  resu["time"] = double(end_s-start_s)/CLOCKS_PER_SEC;
  resu["final_eps"] = eps;
  return resu;
}
