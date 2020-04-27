#ifndef TIME_SERIES_HPP
#define TIME_SERIES_HPP

#include "include_arma.hpp"

class TimeSeries {
protected:
    arma::vec ts;

    arma::mat lags;
    arma::mat sorted_lags;
    int num_steps;
    int k_lag;

    bool is_sorted = false;

public:
    TimeSeries() {}
    ~TimeSeries() {}

    TimeSeries(arma::vec ts, int k=1);

    void compute_lag();

    void compute_lag(int k);

    void hilbert_sort();

    arma::mat get_lags() const { return lags; } 
    arma::mat get_sorted_lags() const { return sorted_lags; }

    arma::vec get_ts() const { return ts; }


};


double dist(const TimeSeries& ts1, const TimeSeries& ts2, 
            double ground_p = 2.0, double p = 1.0);

arma::mat pairwise_dist(const std::vector<TimeSeries> &x,
                        const std::vector<TimeSeries> &y);

#endif