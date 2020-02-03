#include "sinkhorn.hpp"

// arma::vec softmin(const arma::vec& f, const arma::vec& g, double eps) {
//     return
// }

void sinkhorn(arma::vec weights_in, arma::vec weights_out,
              arma::mat cost, double eps,
              double threshold, int max_iter, int norm_p, arma::mat* transport,
              double* dist) {

    arma::mat K = arma::exp(- cost / eps);
    // std::cout << "K: \n" << K << std::endl;
    arma::vec inv_weights_in = arma::pow(weights_in, -1);
    arma::mat Kp(K.n_rows, K.n_cols);

    for (int i=0; i < K.n_rows; i++)
        Kp.row(i) = K.row(i) * inv_weights_in(i);

    // std::cout << "Kp: \n" << Kp << std::endl;

    arma::vec KtransposeU;
    arma::vec u(weights_in.n_elem, arma::fill::ones);
    u /= weights_in.n_elem;

    arma::vec v(weights_out.n_elem, arma::fill::ones);
    v /= weights_out.n_elem;

    arma::vec uprev, vprev, tmp2;

    double err = threshold + 5;

    arma::vec KpU;

    int i=0;
    while (i < max_iter && err > threshold) {
        i += 1;
        uprev = u;
        vprev = v;
        KtransposeU = K.t() * u;
        // std::cout << "KtransposeU: " << KtransposeU.t() << std::endl;
        v = weights_out / KtransposeU;
        // std::cout << "v: " << v.t() << std::endl;
        u = arma::pow(Kp * v, -1);
        // std::cout << "u: " << u.t() << std::endl;
        if  (int(i % 10) == 0) {
            // compute the right marginal
            tmp2 = arma::sum(diagmat(u) * K * diagmat(v), 0).t();
            tmp2 /= arma::accu(tmp2);
            err = arma::norm(tmp2 - weights_out, norm_p);
        }
    }

    *transport =  diagmat(u) * K * diagmat(v);
    double d = arma::accu(*transport % cost);
    *dist = d;
    return;
}
