#include "time_series.hpp"

#include <math.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/hilbert_sort.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/spatial_sort.h>
#include <CGAL/Spatial_sort_traits_adapter_d.h>
#include <boost/iterator/counting_iterator.hpp>
#include <CGAL/Cartesian_d.h>
#include <CGAL/Homogeneous_d.h>

#include <stdio.h>
#include <stdlib.h> //malloc

typedef CGAL::Cartesian_d<double> Kernel;
typedef Kernel::Point_d Point_d;

typedef CGAL::Spatial_sort_traits_adapter_d<Kernel, Point_d *> Search_traits_d;

TimeSeries::TimeSeries(arma::vec ts, int k): ts(ts), k_lag(k) {
    num_steps = ts.n_elem;
    if (k > 0) {
        compute_lag();
        hilbert_sort();
    }
}

void TimeSeries::compute_lag() {
    lags.resize(k_lag + 1, num_steps);
    lags.fill(arma::datum::nan);
    lags.row(0) = ts.t();
    for (int k=1; k < k_lag + 1; k++ ) {
        lags.row(k).tail(num_steps - k) = ts.head(num_steps - k).t(); 
    }

    arma::uvec cols_to_remove = arma::linspace<arma::uvec>(0, k_lag - 1);
    lags.shed_cols(cols_to_remove);
}

void TimeSeries::compute_lag(int k) 
{
    this->k_lag = k;
    compute_lag();
}


void TimeSeries::hilbert_sort() 
{
    int i, k;
    arma::uvec sort_index(lags.n_cols, arma::fill::zeros);
    // sort_index.resize(lags.n_cols);

    double *x = lags.memptr();
    double *work = new double[lags.n_rows];

    std::vector<Point_d> points;
    for (i = 0; i < lags.n_cols; i++)
    {
        for (k = 0; k < lags.n_rows; k++)
        {
            work[k] = x[lags.n_rows * i + k];
        }
        Point_d point(lags.n_rows, work + 0, work + lags.n_rows);
        points.push_back(point) ;
    }

    std::vector<std::ptrdiff_t> indices;
    indices.reserve(points.size());

    std::copy(boost::counting_iterator<std::ptrdiff_t>(0),
              boost::counting_iterator<std::ptrdiff_t>(points.size()),
              std::back_inserter(indices));

    CGAL::hilbert_sort(indices.begin(), indices.end(), Search_traits_d(&(points[0])));

    for (i = 0; i < points.size(); i++)
        sort_index[i] = indices[i];

    sorted_lags = lags.cols(sort_index);
    is_sorted = true;
}

double dist(const TimeSeries &ts1, const TimeSeries &ts2,
            double ground_p, double p)
{
    // arma::mat lags1 = ts1.get_sorted_lags();
    // arma::mat lags2 = ts2.get_sorted_lags();
    // arma::mat lags_diff = arma::abs(lags1 - lags2);
    // arma::vec tmp = arma::pow(arma::sum(arma::pow(lags_diff, p), 0),
    //                           1.0 / ground_p).t();

    arma::vec tmp =
        arma::pow(
            arma::sum(
                arma::pow(
                    arma::abs(ts1.get_sorted_lags() - ts2.get_sorted_lags()),
                    p),
                0),
            1.0 / ground_p).t();
    return arma::accu(arma::pow(tmp, p)) / tmp.n_elem;
}

arma::mat pairwise_dist(const std::vector<TimeSeries> &x,
                        const std::vector<TimeSeries> &y) {
    arma::mat out(x.size(), y.size());
    #pragma omp parallel for collapse(2)
    for (int i=0; i < x.size(); i++) {
        for (int j=0; j < y.size(); j++) {
            out(i, j) = dist(x[i], y[j]);
        }
    }
    return out;
}
