rm(list=ls())
library(ergm)

networks = readRDS("airline_networks.RData")
start_net = network(networks[[12]], directed=F)

simulate_ergm <- function(theta, nnodes=100) {
  as.matrix(simulate(
    start_net ~ edges + degree(0) + degree(1) + degree(10) + degree(50) + degree(70), 
    coef=theta, burnin=100, interval=100))
}

library(Rcpp)
library(devtools)
compileAttributes(pkgdir = ".", verbose = getOption("verbose"))
load_all(reset=T, recompile=T)



var_chol = diag(1, 6, 6) * 30
m0 = c(-10, -40, 0, 0, -100, -20)
eps = 30

inits = list(c(-10, 100, 100, -100, -100, -100), 
             c(-37, -11, -11, -14, -100, 63),
             c(1, 15, 3, 100, 100, -100))

out = runAbcMCMC_graph(networks, 100000, 1, 0.1, m0, var_chol, inits, eps, "aaaa")

saveRDS(out, file="airlines_out.RData")

