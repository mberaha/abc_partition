rm(list=ls())
library(ergm)

simulate_ergm <- function(theta, nnodes=100) {
  y = network(nnodes, directed=F) 
  as.matrix(simulate(
    y ~ degree(0) + degree(1) + degree(10) + degree(50) + degree(70), 
    coef=theta))
}

library(Rcpp)
library(devtools)
compileAttributes(pkgdir = ".", verbose = getOption("verbose"))
load_all(reset=T, recompile=T)


networks = list()
data_per_clus = 20

for (i in 1:data_per_clus) {
  networks[[i]] = simulate_ergm(c(50, 60, -1000, -1000, 22), 100)
}

for (i in 1:data_per_clus) {
  networks[[i + data_per_clus]] = simulate_ergm(c(0, 0, 50, 10, 22), 100)
}

prec_chol = diag(1, 5, 5) / 100000
m0 = rep(0, 5)
eps = 30

out = runAbcMCMC_graph(networks, 100000, 2, 0.2, m0, prec_chol, eps, "aaaa")

saveRDS(out, file="sim_out.RData")
