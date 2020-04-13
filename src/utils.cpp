#include "utils.hpp"

arma::mat vstack(const std::vector<arma::vec> &rows) {
    arma::mat out(rows.size(), rows[0].n_elem);
    for (int i=0; i < rows.size(); i++)
        out.row(i) = rows[i].t();

    return out;
}