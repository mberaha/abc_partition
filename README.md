## Make Random Partitions Great Again!

Code for reproducing the numerical illustrations in the paper ``Bayesian nonparametric model based clustering with intractable distributions: an ABC approach'' by Mario Beraha and Riccardo Corradin.

In particular:

1. The folder ``example_gaussian'' contains all the source code to run the example in Section 4.1 on univariate Gaussian data

2. The folder ``example_g&k_uni'' contains all the source code to run the example in Section 4.2

3. The executable ``run_gnk'' runs the example on the multivariate g-and-k data, can be compiled by executing

```
    make run_gnk
```

4. The executable ``run_ts'' runs the example on time-series data. You must have installed the CGAL library on your machine to compile this.

5. The file ``run_airlines.R'' runs the network clustering example (watch out! Runtimes are very slow for this particular example)

If you don't want to install CGAL on your machine (everything but the time-series example will still work), modify the Makefile by removing all the occurrences of "-DUSE_CGAL"
