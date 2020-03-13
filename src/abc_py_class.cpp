#include "abc_py_class.hpp"

AbcPy::AbcPy(
        int n_data, double theta, double sigma,
        double eps0, std::string distance, int max_iter,
        double entropic_eps, double threshold):
            n_data(n_data), theta(theta),
            sigma(sigma), eps0(eps0) {
    if (distance == "wasserstein")
      d = new UniformDiscreteWassersteinDistance(max_iter);
    else if (distance == "sorting")
      d = new SortingDistance1d();
    else if (distance == "sinkhorn")
      d = new UniformSinkhorn(entropic_eps, threshold, max_iter, false);
    else if (distance == "greenkhorn")
      d = new UniformSinkhorn(entropic_eps, threshold, max_iter, true);

    part = arma::vec(n_data, arma::fill::zeros);
    temp_part = arma::vec(n_data, arma::fill::zeros);
}

void AbcPy::updateUrn() {
    int n = part.n_elem;
    tvec.fill(0.0);
    tvec.head(n) = part;

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
}

void AbcPy::step() {
    tvec.resize(2 * part.n_elem);
    updateUrn();
    temp_part = tvec.tail(n_data);
    uniq_temp = unique(temp_part);

    updateParams();
    generateSyntData();
}

std::tuple<arma::vec, arma::mat, double> AbcPy::run(int nrep) {
    dist_results.resize(nrep);
    part_results.resize(nrep, n_data);

    int start_s = clock();
    for (int iter=0; iter < nrep; iter++) {
        step();
        // std::cout << "data: "; data.t().print();
        // std::cout << "data_synt: "; data_synt.t().print();
        std::tuple<arma::uvec, double> dist_out = d->compute(data, data_synt);
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
          saveCurrParam();
        }
    }

    int end_s = clock();
    return std::make_tuple(
      dist_results, part_results, double(end_s-start_s)/CLOCKS_PER_SEC);
}

AbcPyUniv::AbcPyUniv(
    const arma::mat &data_, double theta, double sigma, double eps0,
    double a0, double b0, double k0, double m0, std::string distance,
    int max_iter, double entropic_eps, double threshold):
        AbcPy(data_.n_rows, theta, sigma, eps0, distance, max_iter,
              entropic_eps, threshold),
        data(data_), data_synt(data_), a0(a0), b0(b0), k0(k0), m0(m0) {

    param.resize(1, 2);
    param(0, 0) = m0;
    param(0, 1) = b0 / (a0 - 1);
    tparam = param;
}


void AbcPyUniv::updateParams() {
    tparam.resize(uniq_temp.n_elem, 2);
    int k = param.n_rows;
    for(arma::uword j = 0; j < uniq_temp.n_elem; j++){
        temp_part(arma::find(temp_part == uniq_temp(j))).fill(j);
        if(uniq_temp(j) < k)
            tparam.row(j) = param.row(uniq_temp(j));
         else {
            tparam(j,1) =  1.0 / arma::randg(arma::distr_param(a0, 1.0 / b0));
            tparam(j,0) = arma::randn() * sqrt(tparam(j,1) / k0) + m0;
         }
    }
}


void AbcPyUniv::generateSyntData() {
    for(arma::uword j = 0; j < temp_part.n_elem; j++){
        data_synt(j) = arma::randn() *
            sqrt(tparam(temp_part(j),1)) + tparam(temp_part(j),0);
    }
}


AbcPyMultiv::AbcPyMultiv(
    const arma::mat &data_, double theta, double sigma, double eps0,
    double df, const arma::mat& prior_prec_chol,
    double k0, const arma::vec& m0, std::string distance,
    int max_iter, double entropic_eps, double threshold):
        AbcPy(data_.n_rows, theta, sigma, eps0, distance, max_iter,
              entropic_eps, threshold),
        data(data_), data_synt(data_), df(df), k0(k0), m0(m0),
        prior_prec_chol(prior_prec_chol) {

    mean.reserve(1000);
    tmean.reserve(1000);
    prec_chol.reserve(1000);
    tprec_chol.reserve(1000);

    mean.push_back(m0);
    tmean.push_back(m0);

    prec_chol.push_back(
        arma::mat(prior_prec_chol.n_rows, prior_prec_chol.n_rows,
                  arma::fill::eye));

    tprec_chol.push_back(
        arma::mat(prior_prec_chol.n_rows, prior_prec_chol.n_rows,
                  arma::fill::eye));
}


void AbcPyMultiv::updateParams() {
    tmean.resize(uniq_temp.n_elem);
    tprec_chol.resize(uniq_temp.n_elem);
    int k = mean.size();

    for(arma::uword j = 0; j < uniq_temp.n_elem; j++){
        temp_part(arma::find(temp_part == uniq_temp(j))).fill(j);
        if(uniq_temp(j) < k) {
            tmean[j] = mean[uniq_temp[j]];
            tprec_chol[j] = prec_chol[uniq_temp[j]];
        } else {
            arma::mat chol_prec = rwishart_chol(df, prior_prec_chol);
            tprec_chol[j] = chol_prec;
            tmean[j] = rnorm_prec_chol(m0, sqrt(k0) * chol_prec);
        }
    }
}


void AbcPyMultiv::generateSyntData() {
    for(arma::uword j = 0; j < temp_part.n_elem; j++){
        arma::vec currmean = tmean[temp_part(j)];
        arma::mat currprec = tprec_chol[temp_part(j)];
        data_synt.row(j) = rnorm_prec_chol(currmean, currprec).t();
    }
}

AbcPyGraph::AbcPyGraph(
    const std::vector<arma::mat> &data_, double theta, double sigma,
    double eps0, const arma::mat &prior_prec_chol,
    const arma::vec &m0, std::string distance,
    int max_iter, double entropic_eps, double threshold) : 
      n_data(data_.size()), theta(theta), sigma(sigma), eps0(eps0),
      m0(m0), prior_prec_chol(prior_prec_chol)
{

  // TODO: change to greedy maybe
  d = new GraphSinkhorn(entropic_eps, threshold, max_iter, false);
  part = arma::vec(n_data, arma::fill::zeros);
  temp_part = arma::vec(n_data, arma::fill::zeros);

  data.resize(data_.size());
  data_synt.resize(data_.size());
  for (int i = 0; i < n_data; i++)
  {
    data[i] = Graph(data_[i]);
  }

  param.reserve(1000);
  param.reserve(1000);

  param.push_back(m0);
  param.push_back(m0);
}

void AbcPyGraph::updateUrn()
{
  int n = part.n_elem;
  tvec.fill(0.0);
  tvec.head(n) = part;

  arma::vec uniq = unique(part);
  int k_max = uniq.n_elem;
  arma::vec tfreqs(k_max + n);
  tfreqs.fill(0.0);

  for (arma::uword j = 0; j < k_max; j++)
  {
    tfreqs(j) = arma::accu(part == uniq(j)) - sigma;
  }

  for (arma::uword i = n; i < 2 * n; i++)
  {
    double t_bound = arma::randu() * (i + theta);
    int k = -1;
    double accu_val = 0.0;

    // loop
    while (t_bound >= accu_val)
    {
      k += 1;
      if (k == k_max)
      {
        break;
      }
      accu_val += tfreqs(k);
    }

    if (k < k_max)
    {
      tfreqs(k) += 1;
      tvec(i) = k;
    }
    else
    {
      tfreqs(k) = 1 - sigma;
      tvec(i) = k;
      k_max += 1;
    }
  }
}

void AbcPyGraph::updateParams()
{
  tparam.resize(uniq_temp.n_elem);
  int k = param.size();

  for (arma::uword j = 0; j < uniq_temp.n_elem; j++)
  {
    temp_part(arma::find(temp_part == uniq_temp(j))).fill(j);
    if (uniq_temp(j) < k)
    {
      tparam[j] = param[uniq_temp[j]];
    }
    else
    {
      tparam[j] = rnorm_prec_chol(m0, prior_prec_chol);
    }
  }
}

void AbcPyGraph::generateSyntData()
{
  for (arma::uword j = 0; j < temp_part.n_elem; j++)
  {
    arma::vec currparam = tparam[temp_part(j)];
    data_synt[j] = data[j];
    // data_synt[j] = Graph(simulator.simulate_graph(currparam));
  }
}

void AbcPyGraph::step()
{
  tvec.resize(2 * part.n_elem);
  updateUrn();
  temp_part = tvec.tail(n_data);
  uniq_temp = unique(temp_part);

  updateParams();
  generateSyntData();
}

std::tuple<arma::vec, arma::mat, double> AbcPyGraph::run(int nrep)
{
  dist_results.resize(nrep);
  part_results.resize(nrep, n_data);

  int start_s = clock();
  for (int iter = 0; iter < nrep; iter++)
  {
    step();
    // std::cout << "data: "; data.t().print();
    // std::cout << "data_synt: "; data_synt.t().print();
    std::tuple<arma::uvec, double> dist_out = d->compute(data, data_synt);
    dist_results(iter) = std::get<1>(dist_out);
    part_results.row(iter) = temp_part(std::get<0>(dist_out)).t();

    lEsum += log(dist_results(iter));
    lEsum2 += pow(log(dist_results(iter)), 2);
    meanEps = lEsum / (iter + 1);
    meanEps2 = lEsum2 / (iter + 1);
    eps = exp(log(eps0) / pow(iter + 1, 2)) *
          exp(meanEps - 2.33 * (meanEps2 - meanEps * meanEps));

    if (dist_results(iter) < eps)
    {
      part = temp_part;
      saveCurrParam();
    }
  }

  int end_s = clock();
  return std::make_tuple(
      dist_results, part_results, double(end_s - start_s) / CLOCKS_PER_SEC);
}
