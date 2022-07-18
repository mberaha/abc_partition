rm(list=ls())
library(ergm)
library(Rcpp)
library(devtools)

compileAttributes(pkgdir = ".", verbose = getOption("verbose"))
devtools::install()

library("abcpp")

networks = readRDS("data/airline_networks.RData")
start_net = network(networks[[12]], directed=F)

simulate_ergm <- function(theta, nnodes=100) {
  as.matrix(simulate(
    start_net ~ edges + degree(0) + degree(1) + degrange(from=2, to=10) + degrange(from=11, to=50),
    coef=c(theta, 1), burnin=100, interval=100))
}





var_chol = diag(1, 4, 4) * 10
m0 = c(-4, 3, 15, -20)
eps = 90

inits = list(c(-3, 2, -1, -1), 
             c(-10, 0, 0, -15),
             c(-1, 10, 30, -40))

nrep = 10
out = run_graph(networks, m0, var_chol, nrep, nrep, 1, 0.1, eps, eps, "wasserstein", inits)
saveRDS(out, file="airlines_out.RData")

