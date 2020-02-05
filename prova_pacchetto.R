setwd("/home/mario/PhD/abc/abc_partition/")
library(devtools)
library(Rcpp)
library(pkgbuild)

pkgbuild::clean_dll()
Rcpp::compileAttributes(".")
devtools::load_all(".", recompile = T)

# devtools::load_all()
n = 50
data = c(rnorm(n, 0, 1), rnorm(n, -5, 1))
out = runAbcMCMC_univ_R(as.matrix(data), 1000, 1, 0.2, 0, 0.5, 2, 2, 500, 1, "sorting")


