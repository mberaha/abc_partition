setwd("/home/mario/PhD/abc/abc_partition/")
library(Rcpp)
library(devtools)
compileAttributes(pkgdir = ".", verbose = getOption("verbose"))
load_all()

res <- c()
for(i in 1:1000000){
  res[i] <- sample_PY(values = c(0,0,0,1,1,2), theta = 1, sigma = 0.2)
}

table(res) / 1000000

c(2.8,1.8,0.8, 1.6) / sum(c(2.8,1.8,0.8, 1.6))


#--------------------

part <- part2 <- c(0,0,0,1,1,2)
param <- param2 <- matrix(c(1,1,1,1,1,1), ncol = 2)
prova <- update_part_PY(part, part2, param, param2, 1, 0.2, 0, 1, 2, 1)
c(2.8, 1.8, 0.8, 1.6) / 7
prova

#-----------------------

n <- 40
datas <- sort(c(rnorm(n, -3, 1), rnorm(n, 3, 1)))
prova <- ABC_MCMC(data = datas, nrep = 10^6,
                  theta = 1, sigma = 0.2, m0 = 0,
                  k0 = 0.2, a0 = 2,
                  b0 = 1, eps = 150,
                  p = 1) 
prova$time
hist(prova$dist, xlim = c(0, 500), breaks = 1000)
sum(prova$dist < 50)
parts <- prova$part_results[prova$dist < quantile(prova$dist, probs = 0.01),]

# compute the posterior similarity matrix
PSM <- matrix(0, ncol = length(datas), nrow = length(datas))
for(j in 1:length(datas)){
  for(k in 1:length(datas)){
    PSM[j,k] <- mean(parts[,j] == parts[,k])
  }
}

# compute the distance for each partition to the PSM
d_abs <- c()
for(i in 1:nrow(parts)){
  d_abs[i] <- sum(abs(1 * (sapply(parts[i,], 
                                  function(x) x == parts[i,])) - PSM))
  
}

# Get the optimal one
hist(datas, breaks = 30)
points(x = datas, y = rep(0,length(datas)), col = parts[which.min(d_abs),] + 5, pch = 20)
parts[which.min(d_abs),]




