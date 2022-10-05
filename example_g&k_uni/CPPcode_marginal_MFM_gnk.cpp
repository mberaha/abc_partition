#include <RcppArmadillo.h>
#include <RcppDist.h>
using namespace arma;
// [[Rcpp::depends(RcppArmadillo, RcppDist)]]


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
 * functions for the g&k distribution, in order
 *  - the quantile as function of a gaussian distribution
 *  - its derivative
 *  - the derivative of the target function (Q(z) - x)^2 w.r.t. z
 *  - Newton-Raphson optimization for the target function
 *  - density function
 */

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

//-----------------------------------------------------------------
// compute the frequencies of a given vector of allocations

vec freq_vec(vec vector){
  vec uniq = unique(vector);
  int n_item = uniq.n_elem;
  vec result(n_item);

  for(uword j = 0; j < n_item; j++){
    result(j) = (int) accu(vector == uniq(j));
  }
  return(result);
}

//-----------------------------------------------------------------
// clean an univariate location-scale allocation,
// by destroying all the empty clusters

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

//-----------------------------------------------------------------
// sample a random integer, given a vector of log weights

int rint_log(vec lweights){

  double u = randu();
  vec probs;
  double mw = max(lweights);

  probs = exp(lweights - mw);
  probs /= sum(probs);
  probs = cumsum(probs);

  for(uword k = 0; k < probs.n_elem; k++) {
    if(u <= probs[k]) {
      return k;
    }
  }
  return -1;
}

//-----------------------------------------------------------------

void update_cluster_allocation_gnk(vec data, 
                                   vec &clust, 
                                   mat &param,
                                   int m, 
                                   double gamma,
                                   vec Vnt_MAR, 
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
    
    bool req_clean = false;
    if(arma::sum(clust == clust[i]) == 1){
      req_clean = true;
    }
    
    if(req_clean){
      temp_idx = rint(m1vec);
      tparam.row(temp_idx) = param.row(clust(i));
    }
    
    clust(i) = param.n_rows + 1;
    if(req_clean){
      clean_param_gnk(clust, param);
    }
    
    // initialize useful quantities
    k = param.n_rows;
    probs.resize(k+m);
    probs.fill(0);
    
    for(uword j = 0; j < k; j++){
      probs(j) = (accu(clust == j) + gamma) * 
        PDFkn(data(i), param(j,0), param(j,1), param(j,2), param(j,3), 0.8);
    }
    
    for(uword j = 0; j < m; j++){
      probs(k + j) = (Vnt_MAR(k) * gamma) / m  * 
        PDFkn(data(i), tparam(j,0), tparam(j,1), tparam(j,2), tparam(j,3), 0.8);
    }
    
    probs.elem(find_nonfinite(probs)).zeros();
    temp_clust = rint(probs);
    
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
  }
}


//---------------------------------------------------------
// main marginal
// [[Rcpp::export]]
Rcpp::List main_univ_MFM_gnk(vec Y,
                             int niter,
                             int nburn,
                             int thin,
                             vec hyperparam,
                             double gamma,
                             int m,
                             vec Vnt_MAR){

  int nitem = (niter - nburn) / thin;
  mat res_clust(nitem, Y.n_elem);

  // initialize results
  arma::mat part_results(niter - nburn, Y.n_elem, arma::fill::zeros);
  arma::vec clust(Y.n_elem);
  arma::mat param(1, 4);
  
  // fill
  clust.fill(0);
  param(0, 0) = hyperparam(0);
  param(0, 1) = hyperparam(3) / (hyperparam(2) - 1);
  param(0, 2) = hyperparam(4);
  param(0, 3) = hyperparam(7) / (hyperparam(6) - 1);

  //loop
  int res_index = 0;
  int start_s = clock();
  int current_s;
  int nupd = round(niter / 10);
  
  for(uword iter = 0; iter < niter; iter++){
    
    update_cluster_allocation_gnk(Y, clust, param, m, gamma, Vnt_MAR, hyperparam);
    clean_param_gnk(clust, param);
    
    // save the results
    if((iter >= nburn) & ((iter + 1) % thin == 0)){
      res_clust.row(res_index) = clust.t();
      res_index += 1;
    }

    if((iter + 1) % nupd == 0){
      current_s = clock();
      Rcpp::Rcout << "Completed:\t" << (iter + 1) << "/" << niter << " - in " <<
        double(current_s-start_s)/CLOCKS_PER_SEC << " sec\n";
    }
    Rcpp::checkUserInterrupt();
  }
  double time = double(current_s-start_s)/CLOCKS_PER_SEC;

  Rcpp::List results;
  results["clust"] = res_clust;
  results["time"] = time;
  return results;
}
