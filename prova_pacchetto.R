setwd("/home/mario/PhD/abc/abc_partition/")
install.packages("RcppArmadillo")

library(devtools)
library(Rcpp)
library(pkgbuild)
library(RcppArmadillo)

RcppArmadillo.package.skeleton(name = "abcpp", list = character(), 
                               environment = .GlobalEnv, path = ".", force = FALSE, 
                               code_files = character(), example_code=FALSE)

pkgbuild::clean_dll()
devtools::load_all(".", recompile = T)

Rcpp::compileAttributes(".")
devtools::load_all()
n = 50
datas = matrix(data=c(rnorm(n, 0, 1), rnorm(n, -5, 1)), ncol=1)
out = test(datas)




