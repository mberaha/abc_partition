#include "graph.hpp"

using namespace arma;

void Graph::compute_laplacian()
{
    vec degree = sum(adj_mat, 1) + 0.01;
    vec inv_degree = pow(degree, -1);
    mat A = arma::sqrt(diagmat(inv_degree));

    laplacian = mat(n_nodes, n_nodes, fill::eye) - A * adj_mat * A;
}

void Graph::compute_eigenvalues()
{
    eig_sym(eigenvalues, eigenvectors, laplacian);
}

arma::mat GraphSimulator::simulate_graph(arma::vec param, int n_nodes)
{
    Rcpp::NumericVector out = simulate_ergm_(param, n_nodes);
    return arma::mat(out.begin(), n_nodes, n_nodes);
}

double graph_dist(const Graph &g1, const Graph &g2)
{
    int n1 = g1.get_n_edges();
    int n2 = g2.get_n_edges();
    const arma::vec eig1 = g1.get_eigenvalues();
    const arma::vec eig2 = g2.get_eigenvalues();
    double out = arma::sum(arma::pow(eig1 - eig2, 2));
    // out *= 1.0 * (std::abs(n1 - n2) + 1.0) / (n1 + n2 + 1.0);
    out *= 1.0 * (std::abs(n1 - n2) + 1.0) / (std::min({n1, n2}) + 1.0);
    return out;
}

arma::mat pairwise_dist(const std::vector<Graph> &x,
                        const std::vector<Graph> &y) 
{
    arma::mat out(x.size(), y.size());
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < x.size(); i++)
    {
        for (int j = 0; j < y.size(); j++)
        {
            out(i, j) = graph_dist(x[i], y[j]);
        }
    }
    return out;
}

// std::tuple<arma::uvec, double> GraphSinkhorn::compute(
//     const std::vector<Graph> &real_data,
//     const std::vector<Graph> &synth_data)
// {
//     if (!init_done)
//         init(real_data, synth_data);

//     compute_cost(real_data, synth_data);

//     dist = -10;
//     call();

// #pragma omp parallel for
//     for (int i = 0; i < n_out; i++)
//     {
//         perm(i) = transport.row(i).index_max();
//     }
//     return std::make_tuple(perm, dist);
// }

// void GraphSinkhorn::init(const std::vector<Graph> &real_data,
//                          const std::vector<Graph> &synth_data)
// {
//     n_in = real_data.size();
//     n_out = synth_data.size();

//     w_in.resize(n_in);
//     w_in.fill(1.0 / n_in);
//     w_out.resize(n_in);
//     w_out.fill(1.0 / n_out);

//     cost_mat.resize(n_in, n_out);
//     transport.resize(n_in, n_out);
//     perm.resize(n_out);
//     init_done = true;
// }

// void GraphSinkhorn::compute_cost(const std::vector<Graph> &real_data,
//                                  const std::vector<Graph> &synth_data)
// {
// #pragma omp parallel for collapse(2)
//     for (int i = 0; i < n_in; ++i)
//     {
//         for (int j = 0; j < n_out; ++j)
//         {
//             cost_mat(i, j) = graph_dist(real_data[i], synth_data[j]);
//         }
//     }
// }

// AbcPyGraph::AbcPyGraph(
//     const std::vector<arma::mat> &data_, double theta, double sigma,
//     double eps0, const arma::mat &prior_var_chol,
//     const arma::vec &m0, std::string distance,
//     const std::vector<arma::vec> &inits,
//     int max_iter, double entropic_eps, double threshold) : n_data(data_.size()), theta(theta), sigma(sigma), eps0(eps0),
//                                                            m0(m0), prior_var_chol(prior_var_chol)
// {
//     // TODO: change to greedy maybe
//     d = new GraphSinkhorn(entropic_eps, threshold, max_iter, false);

//     data.resize(data_.size());
//     data_synt.resize(data_.size());
//     for (int i = 0; i < n_data; i++)
//     {
//         data[i] = Graph(data_[i]);
//     }

//     n_nodes = data[0].get_n_nodes();

//     param.reserve(1000);
//     tparam.reserve(1000);

//     if (inits.size())
//     {
//         for (const arma::vec val : inits)
//             param.push_back(val);
//     }
//     else
//     {
//         param.push_back(m0);
//     }
//     tparam = param;

//     std::cout << "initial params: " << std::endl;
//     for (int i = 0; i < param.size(); i++)
//     {
//         std::cout << param[i] << std::endl;
//     }

//     int NUM_CLUS_INIT = 5;
//     part = arma::randi(data.size(), arma::distr_param(0, NUM_CLUS_INIT));
//     temp_part = arma::ivec(n_data, arma::fill::zeros);
// }

// void AbcPyGraph::updateUrn()
// {
//     int n = part.n_elem;
//     tvec.fill(0.0);
//     tvec.head(n) = part;

//     arma::ivec uniq = unique(part);
//     int k_max = uniq.n_elem;
//     arma::vec tfreqs(k_max + n);
//     tfreqs.fill(0.0);

//     for (arma::uword j = 0; j < k_max; j++)
//     {
//         tfreqs(j) = arma::accu(part == uniq(j)) - sigma;
//     }

//     for (arma::uword i = n; i < 2 * n; i++)
//     {
//         double t_bound = arma::randu() * (i + theta);
//         int k = -1;
//         double accu_val = 0.0;

//         // loop
//         while (t_bound >= accu_val)
//         {
//             k += 1;
//             if (k == k_max)
//             {
//                 break;
//             }
//             accu_val += tfreqs(k);
//         }

//         if (k < k_max)
//         {
//             tfreqs(k) += 1;
//             tvec(i) = k;
//         }
//         else
//         {
//             tfreqs(k) = 1 - sigma;
//             tvec(i) = k;
//             k_max += 1;
//         }
//     }
// }

// void AbcPyGraph::updateParams()
// {
//     tparam.resize(uniq_temp.n_elem);
//     int k = param.size();

//     for (arma::uword j = 0; j < uniq_temp.n_elem; j++)
//     {
//         temp_part(arma::find(temp_part == uniq_temp(j))).fill(j);
//         if (uniq_temp(j) < k)
//         {
//             tparam[j] = param[uniq_temp[j]];
//         }
//         else
//         {
//             arma::vec aux = arma::randn<arma::vec>(m0.n_elem);
//             tparam[j] = prior_var_chol * aux + m0;
//         }
//     }
// }

// void AbcPyGraph::generateSyntData()
// {

//     for (arma::uword j = 0; j < temp_part.n_elem; j++)
//     {
//         arma::vec currparam = tparam[temp_part(j)];
//         arma::mat g = simulator.simulate_graph(currparam, n_nodes);
//         data_synt[j] = Graph(g);
//     }
// }

// void AbcPyGraph::step()
// {

//     tvec.resize(2 * part.n_elem);
//     updateUrn();
//     temp_part = tvec.tail(n_data);
//     uniq_temp = unique(temp_part);

//     updateParams();
//     generateSyntData();
// }

// std::tuple<
//     arma::vec, arma::imat, double,
//     std::vector<arma::mat>>
// AbcPyGraph::run(int nrep)
// {
//     dist_results.resize(nrep);
//     part_results.resize(nrep, n_data);
//     param_results.resize(nrep);

//     int start_s = clock();
//     for (int iter = 0; iter < nrep; iter++)
//     {
//         if (int((iter + 1) % 100) == 0)
//         {
//             std::cout << "Iter: " << iter + 1 << " / " << nrep << std::endl;
//         }
//         step();
//         // std::cout << "data: "; data.t().print();
//         // std::cout << "data_synt: "; data_synt.t().print();
//         std::tuple<arma::uvec, double> dist_out = d->compute(data, data_synt);
//         dist_results(iter) = std::get<1>(dist_out);
//         part_results.row(iter) = temp_part(std::get<0>(dist_out)).t();

//         lEsum += log(dist_results(iter));
//         lEsum2 += pow(log(dist_results(iter)), 2);
//         meanEps = lEsum / (iter + 1);
//         meanEps2 = lEsum2 / (iter + 1);
//         eps = exp(log(eps0) / pow(iter + 1, 2)) *
//               exp(meanEps - 2.33 * (meanEps2 - meanEps * meanEps));

//         if (dist_results(iter) < eps)
//         {
//             part = temp_part;
//             saveCurrParam();
//         }

//         param_results[iter] = vstack(param);

//         if (int(iter + 1) % 5000 == 0)
//         {
//             part_results.save("part_chkpt.csv", arma::csv_ascii);
//             dist_results.save("dists_chkpt.csv", arma::csv_ascii);
//         }
//     }

//     int end_s = clock();
//     return std::make_tuple(
//         dist_results, part_results, double(end_s - start_s) / CLOCKS_PER_SEC,
//         param_results);
// }
