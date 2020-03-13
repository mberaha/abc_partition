#include "sinkhorn.hpp"


void sinkhorn(const arma::vec &weights_in, const arma::vec &weights_out,
              const arma::mat &cost, double eps,
              double threshold, int max_iter, int norm_p, arma::mat* transport,
              double* dist) {

    double minus_eps = -1.0 / eps;
    arma::mat K = arma::exp(cost * minus_eps);
    arma::vec inv_weights_in = arma::pow(weights_in, -1);
    arma::mat Kp(K.n_rows, K.n_cols);

    for (int i=0; i < K.n_rows; i++)
        Kp.row(i) = K.row(i) * inv_weights_in(i);

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
        v = weights_out / KtransposeU;
        u = arma::pow(Kp * v, -1);
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

void greenkhorn(const arma::vec &weights_in, const arma::vec &weights_out,
                const arma::mat &cost, double eps,
                double threshold, int max_iter, int norm_p, arma::mat* transport,
                double* dist, bool uniform) {
    double stopThr = 1.0;
    int i1, i2;
    double max_viol1, max_viol2;
    double old_u, old_v;
    double new_u, new_v;
    double Ki1dotV, Ki2dotU;
    arma::vec Krowi1V, Kcoli2U;

    double minus_eps = -1.0 / eps;
    arma::mat K = arma::exp(cost * minus_eps);

    arma::vec u, v;
    if (uniform) {
        u = weights_in;
        v = weights_in;
    } else {
        u = arma::vec(weights_in.n_elem);
        u.fill(1.0 / weights_in.n_elem);

        v = arma::vec(weights_out.n_elem);
        v.fill(1.0 / weights_out.n_elem);
    }

    arma::mat G = arma::diagmat(u) * K * arma::diagmat(v);

    arma::vec viol1 = arma::sum(G, 1) - weights_in;
    arma::vec viol2 = arma::sum(G, 0).t() - weights_out;

    int niter = 0;

    while ((stopThr > threshold) & (niter < max_iter)) {
        niter += 1;
        i1 = arma::abs(viol1).index_max();
        i2 = arma::abs(viol2).index_max();
        max_viol1 = std::abs(viol1[i1]);
        max_viol2 = std::abs(viol2[i2]);
        stopThr = std::max(max_viol1, max_viol2);

        if (max_viol1 > max_viol2) {
            old_u = u[i1];
            Ki1dotV = arma::dot(K.row(i1), v);
            new_u = weights_in[i1] / Ki1dotV;
            u[i1] = new_u;
            Krowi1V = K.row(i1).t() % v;
            G.row(i1) = Krowi1V.t() * new_u;

            viol1[i1] = new_u * Ki1dotV - weights_in[i1];
            viol2 += Krowi1V * (new_u - old_u);
        } else {
            old_v = v[i2];
            Ki2dotU = arma::dot(K.col(i2), u);
            new_v = weights_out[i2] / Ki2dotU;
            v[i2] = new_v;
            Kcoli2U = K.col(i2) % u;
            G.col(i2) = Kcoli2U * new_v;

            viol2[i2] = new_v * Ki2dotU - weights_out[i2];
            viol1 += Kcoli2U * (new_v - old_v);
        }
    }
    *transport = G;
    *dist = arma::accu(*transport % cost);
}
