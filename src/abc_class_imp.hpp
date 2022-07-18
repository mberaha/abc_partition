#ifndef ABC_CLASS_IMP
#define ABC_CLASS_IMP

#include <stdexcept>

template <typename data_t, typename param_t, class Kernel>
AbcPy<data_t, param_t, Kernel>::AbcPy(
    std::vector<data_t> data, std::vector<param_t> inits,
    double theta, double sigma, double eps0, double eps_star,
    std::string distance, Kernel kernel, int max_iter,
    double entropic_eps, double threshold) : 
        data(data), n_data(data.size()), theta(theta),
        sigma(sigma), eps0(eps0), eps_star(eps_star), kernel(kernel)
{

    if (distance == "sinkhorn")
        d = new UniformSinkhorn<data_t>(entropic_eps, threshold, max_iter, false);
    else if (distance == "greenkhorn")
        d = new UniformSinkhorn<data_t>(entropic_eps, threshold, max_iter, true);
    else if (distance == "wasserstein")
        d = new UniformWasserstein<data_t>;
    else
        throw std::invalid_argument("invalid distance identifier");

    param = inits;
    tparam = inits;
    part = arma::randi(data.size(), arma::distr_param(0, inits.size() - 1));
    // part = arma::ivec(data.size(), arma::fill::zeros);
    // part.tail(int(data.size() / 2)).fill(1);
    std::cout << "part: " << part.t() << std::endl;
    temp_part = part;
    // std::cout << "generating dataset" << std::endl;
    data_synt = kernel.generate_dataset(temp_part, tparam);
    std::cout << "INIT_DIST: " << std::get<1>(
        d->compute(data, data_synt)) << std::endl;
    // std::cout << "generating dataset - DONE" << std::endl;

}

template <typename data_t, typename param_t, class Kernel>
void AbcPy<data_t, param_t, Kernel>::updateUrn()
{
    int n = part.n_elem;
    tvec.fill(0);
    tvec.head(n) = part;

    arma::ivec uniq = unique(part);
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

template <typename data_t, typename param_t, class Kernel>
void AbcPy<data_t, param_t, Kernel>::step()
{

    tvec.resize(2 * part.n_elem);
    updateUrn();
    temp_part = tvec.tail(n_data);
    uniq_temp = unique(temp_part);

    updateParams();
    generateSyntData();
}

template <typename data_t, typename param_t, class Kernel>
void AbcPy<data_t, param_t, Kernel>::updateParams()
{

    tparam.resize(uniq_temp.n_elem);
    int k = param.size();
    for (arma::uword j = 0; j < uniq_temp.n_elem; j++)
    {
        temp_part(arma::find(temp_part == uniq_temp(j))).fill(j);
        if (uniq_temp(j) < k)
            tparam[j] = param[uniq_temp(j)];
        else
            tparam[j] = kernel.sample_prior();
    }
}

template <typename data_t, typename param_t, class Kernel>
void AbcPy<data_t, param_t, Kernel>::generateSyntData()
{

    std::vector<data_t> temp = kernel.generate_dataset(temp_part, tparam);
    for (int i=0; i < n_data; i++) {
        data_synt[i] = temp[i];
    }
}

template <typename data_t, typename param_t, class Kernel>
void AbcPy<data_t, param_t, Kernel>::saveCurrParam()
{
    param = tparam;
}

template <typename data_t, typename param_t, class Kernel>
double AbcPy<data_t, param_t, Kernel>::run(int nrep, int nburn, bool adapt_only_burn) {
    dist_results.resize(nrep + 1);
    part_results.resize(nrep + 1, n_data);
    arma::ivec real_data_part = part;

    std::cout << "data: " << data.size() << std::endl;

    if (log)
        param_log.resize(nrep);

    int start_s = clock();

    int n_accept = 0;
    int iter = 0;
    eps = eps0;
    std::cout << "eps0: " << eps0 << std::endl;

    while (n_accept < nrep) {
        iter += 1;
        step();
        std::tuple<arma::uvec, double> dist_out = d->compute(data, data_synt);
        double d = std::get<1>(dist_out);

        if (iter % 200 == 0) {
            arma::ivec uniq = unique(temp_part);
            int k1_max = uniq.n_elem;
            uniq = unique(part);
            int k2_max = uniq.n_elem;
            std::cout << "Iter: " << iter << " / " << nrep << "; "
                      << "accepted: " << n_accept 
                      << ", eps: " << eps 
                      << ", dist: " <<  d  
                      << ", n_clust_prop: " << k1_max
                      << ", n_clust_acc: " << k2_max << std::endl;
        }


        if (d < eps) {
            // std::cout << "ACCEPT!" << std::endl;
            if (log)
                param_log[n_accept] = param;

            n_accept += 1;

            part = temp_part;
            saveCurrParam();

            dist_results(n_accept) = d;
            real_data_part = part(std::get<0>(dist_out));
            part_results.row(n_accept) = real_data_part.t();

            // std::cout << "Done" << std::endl;
        }

        if ( (adapt_only_burn && (n_accept < nburn)) || (!adapt_only_burn) )
            update_eps(d, n_accept);


        if (iter > 5000000)
            break;
    }

    // for (int iter = 0; iter < nrep; iter++) {
    //     step();
    //     std::tuple<arma::uvec, double> dist_out = d->compute(data, data_synt);
    //     dist_results(iter) = std::get<1>(dist_out);
    //     lEsum += std::log(dist_results(iter));
    //     lEsum2 += pow(std::log(dist_results(iter)), 2);
    //     meanEps = lEsum / (iter + 1);
    //     meanEps2 = lEsum2 / (iter + 1);
    //     eps = exp(std::log(eps0) / pow(iter + 1, 2)) *
    //           exp(meanEps - 2.33 * (meanEps2 - meanEps * meanEps));
    //     if (dist_results(iter) < eps)
    //     {
    //         num_accepted += 1;
    //         part = temp_part;
    //         saveCurrParam();
    //         real_data_part = part(std::get<0>(dist_out));
    //         part_results.row(iter) = real_data_part.t();
    //     } else {
    //         part_results.row(iter) = real_data_part.t();
    //     }

    //     if (log)
    //         param_log[iter] = param;
        

    //     if (iter % 10 == 0) {
    //         std::cout << "Iter: " << iter << " / " << nrep << "; "
    //                   << "accepted: " << 1.0 * num_accepted / iter << "%" 
    //                   << std::endl;
    //     }
    // }

    int end_s = clock();
    return double(end_s - start_s) / CLOCKS_PER_SEC;
}

template <typename data_t, typename param_t, class Kernel>
void AbcPy<data_t, param_t, Kernel>::update_eps(double dist, int iter) {
    // TODO -> this updates without the betas
    double discount = (0.05 - 1.0 * (dist < eps_star));
    double decay = pow(1.0 * iter, (-2.0 / 3.0));
    eps_star *=  exp(decay * discount);
    eps = eps_star;
}

#endif
