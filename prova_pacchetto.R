setwd("/home/mario/PhD/abc/abc_partition/")

library(devtools)
library(Rcpp)
library(pkgbuild)
library(RcppArmadillo)


pkgbuild::clean_dll()
Rcpp::compileAttributes(".")
devtools::load_all(".", recompile = T)

n = 50
datas = matrix(data=c(rnorm(n, 0, 1), rnorm(n, -5, 1)), ncol=1)
out = test(datas)




