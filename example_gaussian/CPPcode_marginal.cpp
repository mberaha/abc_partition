#include <RcppArmadillo.h>
#include <RcppDist.h>
using namespace arma;
// [[Rcpp::depends(RcppArmadillo, RcppDist)]]

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

void para_clean_univ(vec &means,
                     vec &variances,
                     vec &clust){
  int k = means.n_elem;
  double tmu;
  double tvar;
  int u_bound;

  // the loop starts from 1, 'cause the cluster 0 denotes the singletons
  for(arma::uword i = 0; i < k; i++){

    if((int) arma::sum(clust == i) == 0){
      for(arma::uword j = k; j > i; j--){
        if((int) arma::sum(clust == j) != 0){

          clust( arma::find(clust == j) ).fill(i);
          tmu = means[i];
          means[i] = means[j];
          means[j] = tmu;
          
          tvar = variances[i];
          variances[i] = variances[j];
          variances[j] = tvar;

          break;
        }
      }
    }
  }

  u_bound = 0;
  for(arma::uword i = 0; i < k; i++){
    if(arma::accu(clust == i) > 0){
      u_bound += 1;
    }
  }

  means.resize(u_bound);
  variances.resize(u_bound);
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
// acceleration step: update the group specific parameters

void accelerate_PY(mat Y,
                   vec &means,
                   vec &variances,
                   vec clust,
                   double m0,
                   double k0,
                   double a0,
                   double b0){
  vec tdata;
  double an, kn, xtemp, ytemp, data_m, mn, bn;
  int nj, ngj;

  for(uword j = 0; j < means.n_rows; j++){

    nj  = accu(clust == j);
    tdata = Y(find(clust == j));

    // calculate the updated hyperparameters
    data_m = mean(tdata);
    kn = (k0 + nj);
    mn = ((m0 * k0) + nj * data_m) / kn;
    an = a0 + (nj / 2.0);
    bn = b0 + ( accu(pow(tdata - mean(tdata), 2)) + k0 * nj / (k0 + nj) * pow(mean(tdata) - m0, 2) ) / 2;

    // sample the new values
    variances(j) = 1.0 / arma::randg(arma::distr_param(an, 1 / bn));
    means(j)     = arma::randn() * sqrt(variances(j) / kn) + mn;
  }
}

//-----------------------------------------------------------------

void update_cluster_allocation(vec Y,
                               vec &clust,
                               vec &means,
                               vec &variances,
                               double m0,
                               double k0,
                               double a0,
                               double b0,
                               double mass,
                               double sigma_PY){

  vec probs, temp_clust;
  int sampled;
  int k, n;
  double temp_predictive, kn, mn, an, bn;

  n = Y.n_elem;
  for(uword i = 0; i < clust.n_elem; i++){

    // if the cluster is a singleton of the discrete component
    // destroy the cluster
    bool req_clean = false;
    if(arma::sum(clust == clust[i]) == 1){
      req_clean = true;
    }

    clust(i) = means.n_elem + 5;
    if(req_clean){
      para_clean_univ(means,variances, clust);
    }

    k = means.n_elem;
    probs.resize(k + 1);

    // calculate the probabilities of being allocated in the diffuse component,
    // in the active clusters of the discrete component or in a new cluster
    for(arma::uword j = 0; j < k; j++){
      probs(j)  = log(accu(clust == j) - sigma_PY) - log(mass + n - 1)
      - 0.5 * log(variances(j) * 2 * M_PI) - 0.5 * pow(Y(i) - means(j), 2) / variances(j);
    }
    probs(k) = log(mass + k * sigma_PY) - log(mass + n - 1)
      + d_lst(Y(i), 2 * a0, m0, sqrt(b0 / a0 * (1.0 + 1.0 / k0)), true);

    // sample the cluster for the i-th obs
    sampled = rint_log(probs);
    clust(i) = sampled;
    
    // Rcpp::Rcout << (exp(probs) / sum(exp(probs))).t() << "\n\n--------\n";
    // if the cluster is new and not a singletons
    // generate an atom
    if(sampled == k){
      means.resize(k+1);
      variances.resize(k+1);

      kn = (k0 + 1.0);
      mn = ((m0 * k0) + Y(i)) / kn;
      an = a0 + (1.0 / 2.0);
      bn = b0 + (pow(m0, 2) * k0 + pow(Y(i), 2) - pow(mn, 2) * kn) / 2;

      variances(k) = 1.0 / arma::randg(arma::distr_param(an, 1.0 / bn));
      means(k) = arma::randn() * sqrt(variances(k) / kn) + mn;
    }
  }
}

//---------------------------------------------------------
// main marginal
// [[Rcpp::export]]
Rcpp::List main_univ_PY(vec Y,
                        int niter,
                        int nburn,
                        int thin,
                        double m0,
                        double k0,
                        double a0,
                        double b0,
                        double theta,
                        double sigma){

  int nitem = (niter - nburn) / thin;
  mat res_clust(nitem, Y.n_elem);

  //quantities
  vec clust(Y.n_elem);
  vec means(1);
  vec variances(1);
  
  clust.fill(1);
  means.fill(0.0);
  variances.fill(1.0);

  //loop
  int res_index = 0;
  int start_s = clock();
  int current_s;
  int nupd = round(niter / 10);
  
  for(uword iter = 0; iter < niter; iter++){

    update_cluster_allocation(Y, clust, means, variances, m0, k0, a0, b0, theta, sigma);
    accelerate_PY(Y, means, variances, clust, m0, k0, a0, b0);
    
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

//---------------------------------------------------------
//---------------------------------------------------------
//---------------------------------------------------------



