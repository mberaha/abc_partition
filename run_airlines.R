rm(list=ls())
library(ergm)

simulate_ergm <- function(theta, nnodes=100) {
  y = network(nnodes, directed=F) 
  as.matrix(simulate(
    y ~ degree(0) + degree(1) + degree(10) + degree(50) + degree(70), 
    coef=theta, burnin=100, interval=100))
}

library(Rcpp)
library(devtools)
compileAttributes(pkgdir = ".", verbose = getOption("verbose"))
load_all(reset=T, recompile=T)

networks = readRDS("airline_networks.RData")

var_chol = diag(1, 5, 5) * 500
m0 = rep(0, 5)
eps = 30

inits = list(c(50, 60, -1000, -1000, 22), c(0, 0, 50, 10, 22))

out = runAbcMCMC_graph(networks, 100000, 1, 0.1, m0, var_chol, inits, eps, "aaaa")

saveRDS(out, file="airlines_out.RData")

