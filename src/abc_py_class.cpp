#include "abc_py_class.hpp"

AbcPy::AbcPy(
        arma::mat data, double theta, double sigma,
        double eps0, std::string distance):
            data(data), n_data(data.n_rows), theta(theta), sigma(sigma),
            eps0(eps0), data_synt(data) {
    if (distance == "wasserstein")
      d = new UniformDiscreteWassersteinDistance();
    else if (distance == "sorting")
      d = new SortingDistance1d();
    else if (distance == "sinkhorn")
      d = new UniformSinkhorn();

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
    // std::cout << "AbcPy::step" << std::endl;
    tvec.resize(2 * part.n_elem);
    updateUrn();

    temp_part = tvec.tail(n_data);
    uniq_temp = unique(temp_part);

    updateParams();
    generateSyntData();
}

std::tuple<arma::vec, arma::mat, double> AbcPy::run(int nrep) {
    // std::cout << "AbcPy::run" << std::endl;
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
