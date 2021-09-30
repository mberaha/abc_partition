#include "utils.hpp"

arma::mat vstack(const std::vector<arma::vec> &rows) {
    arma::mat out(rows.size(), rows[0].n_elem);
    for (int i=0; i < rows.size(); i++)
        out.row(i) = rows[i].t();

    return out;
}

std::vector<arma::vec> to_vectors(const arma::mat &mat) {
    std::vector<arma::vec> out(mat.n_rows);
    for (int i=0; i < mat.n_rows; i++)
        out[i] = mat.row(i).t();

    return out;
}

arma::mat pairwise_dist(const arma::mat &x, const arma::mat &y) {

    arma::mat out(x.n_rows, x.n_rows);

    #pragma omp parallel for
    for (int i = 0; i < x.n_rows; ++i)
    {
        arma::rowvec curr = x.row(i);
        out.row(i) = arma::sum(
                         arma::abs(y.each_row() - curr), 1)
                         .t();
    }
    return out;
}


arma::mat pairwise_dist(const std::vector<double> &x, 
                        const std::vector<double> &y)
{
    arma::vec xx(x.data(), x.size());
    arma::vec yy(y.data(), y.size());
    return pairwise_dist(xx, yy);
}

arma::mat pairwise_dist(const std::vector<arma::vec> &x,
                        const std::vector<arma::vec> &y)
{
    arma::mat out(x.size(), y.size());
    #pragma omp parallel for
    for (int i = 0; i < x.size(); ++i)
    {
        const arma::vec curr = x[i];
        std::vector<double> dists;
        std::transform(y.begin(), y.end(), std::back_inserter(dists),
                       [curr](const arma::vec other) { 
                           return arma::accu(arma::abs(curr - other)); } );
        out.row(i) = arma::rowvec(dists.data(), dists.size());
    }
    return out;
}

void pairwise_dist_rowmajor(const std::vector<double> &x,
                            const std::vector<double> &y, double* out) {
    for (int i = 0; i < x.size(); ++i) {
        for (int j = 0; j < y.size(); ++j)
            out[i*y.size() + j] = std::abs(x[i] - y[j]);
    }
}

void pairwise_dist_rowmajor(const std::vector<arma::vec> &x,
                            const std::vector<arma::vec> &y, double* out) {
    for (int i = 0; i < x.size(); ++i) {
        for (int j = 0; j < y.size(); ++j)
            out[i*y.size() + j] = arma::accu(arma::abs(x[i] - y[j]));
    }
}
