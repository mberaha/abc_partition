#include "abc_py.hpp"

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


std::tuple<arma::vec, arma::mat, double> ABC_MCMC(
    arma::vec data, int nrep, double theta, double sigma, double m0,
    double k0, double a0, double b0, double eps0, int p,
    std::string dist) {

  Distance* d;
  if (dist == "wasserstein")
    d = new UniformDiscreteWassersteinDistance();
  else if (dist == "sorting") {
    d = new SortingDistance1d();
    data = arma::sort(data);
  }
  else {

  }

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

  double lEsum = 0;
  double lEsum2 = 0;
  double eps = eps0;
  double meanEps, meanEps2;

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

    std::tuple<arma::uvec, double> dist_out = d->compute(data, data_synth);
    dist_results(iter) = std::get<1>(dist_out);
    part_results.row(iter) = temp_part(std::get<0>(dist_out)).t();

    lEsum  += log(dist_results(iter));
    lEsum2 += pow(log(dist_results(iter)), 2);
    meanEps = lEsum / (iter + 1);
    meanEps2 = lEsum2 / (iter + 1);
    eps = exp(log(eps0) / pow(iter + 1, 2)) *
          exp(meanEps - 2.33 * (meanEps2 - meanEps * meanEps));

    if(dist_results(iter) < eps){
      part = temp_part;
      param = tparam;
    }
  }
  int end_s = clock();

  std::cout << "part_results: \n"; part_results.print();

  // return the results
  return std::make_tuple(
    dist_results, part_results, double(end_s-start_s)/CLOCKS_PER_SEC);
}
