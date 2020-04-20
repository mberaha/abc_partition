rm(list=ls())
library(ergm)

networks = readRDS("airline_networks.RData")
start_net = network(networks[[12]], directed=F)

simulate_ergm <- function(theta, nnodes=100) {
  as.matrix(simulate(
    start_net ~ edges + degree(0) + degrange(from=2, to=10) + degrange(from=11, to=50) + degrange(from=50, to=70),
    coef=c(theta, 1), burnin=100, interval=100))
}

library(Rcpp)
library(devtools)
compileAttributes(pkgdir = ".", verbose = getOption("verbose"))
load_all(reset=T, recompile=T)



var_chol = diag(1, 4, 4) * 10
m0 = c(-4, 3, 15, -20)
eps = 300

inits = list(c(-3, 2, -1, -1), 
             c(-10, 0, 0, -15),
             c(-1, 10, 30, -40))

out = runAbcMCMC_graph(networks, 100000, 1, 0.1, m0, var_chol, inits, eps, "aaaa")
saveRDS(out, file="airlines_out.RData")

