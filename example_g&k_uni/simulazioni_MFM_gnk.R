set.seed(42)

library(ggplot2)
library(ggpubr)
library(doParallel)
library(coda)
library(R.utils)
library(matrixStats)

# Useful functions 

BL <- function(c1, c2){
library(latex2exp)
  n <- length(c1)
  temp_mat <- matrix(0, ncol = n, nrow = n)
  for(i in 1:n){
    for(j in 1:i){
      if((c1[i] == c1[j] & c2[i] != c2[j]) | (c1[i] != c1[j] & c2[i] == c2[j])){
        temp_mat[i,j] <- 1
      }
    }
  }
library(Rcpp)
  return(sum(temp_mat) * 2 / (n * (n - 1)))
}

VI <- function(c1, c2){
  n <- length(c1)
  
  T1 <- table(c1) / n
  T2 <- table(c2) / n
  
  H1 <- - sum(T1 * log(T1))
  H2 <- - sum(T2 * log(T2))
  
  T12 <- table(c1, c2) / n
  I12 <- sum(T12 * log(T12 / T1 %*% t(T2)), na.rm = T)
  
  return((H1 + H2 - 2 * I12) / log(n))
}

single_weight_log <- function(lambda, gamma, n, t, M){
  temp <- sapply(1:M, function(x){
    if(x < t){
      -10^(-15)
    } else {
      lgamma(gamma * x) - lgamma(gamma * x + n) + lgamma(x + 1) - lgamma(x - t + 1) +
        dpois(x - 1, lambda = lambda, log = T)
    }
  })
  logSumExp(temp)
}


function_weights_ABC <- function(lambda, gamma, n, M){
  temp_out <- matrix(0, nrow = n + 1, ncol = 2 * n + 1)
  pb <- txtProgressBar(min = 0, max = nrow(temp_out), style = 3)  
  for(i in 1:nrow(temp_out)){
    temp_out[i,] <- sapply(1:ncol(temp_out), function(t) 
      exp(single_weight_log(lambda, gamma, n - 1 + i, t, M) - 
            single_weight_log(lambda, gamma, n - 1 + i, t - 1, M)))
    setTxtProgressBar(pb, i)
  }
  close(pb)
  return(temp_out)
}


# Simulation study: 
#     generate n = (100, 250) values from a mixture of two Gaussian
#     estimate a Gaussian mixture model with two spec. of the ABC sampler, two spec. of the adaptive ABC sampler
#     and the marginal sampler
#

ncl <- detectCores() - 1
nABC <- 10^5
niter <- 15000
nburn <- 5000

#------------- 

nrep <- 50
niter <- 15000
nburn <- 5000
nupd <- 15000
nrep <- 50
n <- c(100, 250)
eps <- log(n)^2.5 / 2
lambda <- 1
gamma <- 1

Vnt_ABC <- list()
Vnt_ABC[[1]] <- function_weights_ABC(lambda = lambda, gamma = gamma, n = n[1], M = 10^3)
Vnt_ABC[[2]] <- function_weights_ABC(lambda = lambda, gamma = gamma, n = n[2], M = 10^3)

#------------- 

cl <- makeCluster(ncl, outfile = "message.txt")
registerDoParallel(cl)
writeLines(c(""), "print_results.txt")
writeLines(c(""), "print_time_out.txt")

resultsMFM <- foreach(i = 1:nrep, .packages = c("Rcpp", "RcppArmadillo"), 
                    .export = c("effectiveSize", "withTimeout", "as.mcmc"), 
                    .noexport = c("ABC_MCMCw_MFM_gnk", "adapt_ABC_MCMCw_MFM_gnk", "main_univ_MFM_gnk", "compute_psm",
                                  "compute_dist", "VI_LB", "Qnk")) %dopar% {
                      sourceCpp("code_CPP_MFM_gnk.cpp")
                      sourceCpp('CPPcode_marginal_MFM_gnk.cpp')
                      sourceCpp('Utility.cpp')
                      
                      res <- array(0, dim = c(4, 5, 3))
                      partM <- partM2 <- partABCad <- partABCad2 <- partT <- list()
                      
                      for(j in 1:2){
                        
                        # sample the data
                        temp_ass <- sample(c(0,1), size = n[j], replace = T, prob = c(0.75, 0.25))
                        data <- sapply(temp_ass, function(x) ifelse(x == 1, Qnk(z = rnorm(1), a = 3, b = 0.5, g = 0.4, k = 0.5, c = 0.8), 
                                                                    Qnk(z = rnorm(1), a = -3, b = 0.75, g = -0.9, k = 0.1, c = 0.8)))
                        indx <- order(data)
                        data_sort <- data[indx]
                        partOPT <- temp_ass[indx]
                        partT[[j]] <- partOPT
                        
                        #------ ABC adaptive -- fixed adapt
                        
                        time_temp <- system.time(mod1 <- adapt_ABC_MCMCw_MFM_gnk(data = data_sort, niter = niter, nburn = nburn, gamma = gamma,
                                                                                 Vnt_ABC = Vnt_ABC[[j]][-1,-1], hyperparam = c(0, 5, 2, 1, 0, 5, 2, 1), 
                                                                                 eps0 = 10 * log(n[j]), eps_star = 10 * log(100), p = 1, adapt_fix = TRUE))[1]
                        
                        res[1,1,j] <- time_temp / (niter - nburn) 
                        parts <- mod1$part_results
                        PSM <- compute_psm(parts)
                        est_par <- parts[which.min(compute_dist(parts, PSM)),]
                        est_par2 <- parts[which.min(VI_LB(parts, PSM)),]
                        res[1,2,j] <- effectiveSize(as.mcmc(apply(parts, 1, function(x) sum(table(x) * log(table(x))))))
                        partABCad[[j]] <- est_par2
                        res[1,3,j] <- effectiveSize(as.mcmc(apply(parts, 1, function(x) length(unique(x)))))
                        res[1,4,j] <- VI(partOPT + 1, est_par + 1)
                        res[1,5,j] <- VI(partOPT + 1, est_par2 + 1)
                        
                        #------ ABC adaptive -- full adapt
                        
                        time_temp <- system.time(mod2 <- adapt_ABC_MCMCw_MFM_gnk(data = data_sort, niter = niter, nburn = nburn, gamma = gamma,
                                                                            Vnt_ABC = Vnt_ABC[[j]][-1,-1], hyperparam = c(0, 5, 2, 1, 0, 5, 2, 1), 
                                                                            eps0 = 10 * log(n[j]), eps_star = 10 * log(100), p = 1, adapt_fix = FALSE))[1]
                        
                        res[2,1,j] <- time_temp / (niter - nburn) 
                        parts <- mod2$part_results
                        PSM <- compute_psm(parts)
                        est_par <- parts[which.min(compute_dist(parts, PSM)),]
                        est_par2 <- parts[which.min(VI_LB(parts, PSM)),]
                        res[2,2,j] <- effectiveSize(as.mcmc(apply(parts, 1, function(x) sum(table(x) * log(table(x))))))
                        partABCad2[[j]] <- est_par2
                        res[2,3,j] <- effectiveSize(as.mcmc(apply(parts, 1, function(x) length(unique(x)))))
                        res[2,4,j] <- VI(partOPT+1, est_par + 1)
                        res[2,5,j] <- VI(partOPT + 1, est_par2 + 1)
                        
                        #------ marginal -- small m
                        
                        time_temp <- system.time(mod3 <- main_univ_MFM_gnk(Y = data_sort, niter = niter, nburn = nburn, thin = 1,
                                                                      hyperparam = c(0, 5, 2, 1, 0, 5, 2, 1), gamma = gamma, 
                                                                      m = 10, Vnt_MAR = Vnt_ABC[[j]][1,-1]))[1]
                        
                        res[3,1,j] <- time_temp / (niter - nburn) 
                        parts <- mod3$clust
                        PSM <- compute_psm(parts)
                        est_par <- parts[which.min(compute_dist(parts, PSM)),]
                        est_par2 <- parts[which.min(VI_LB(parts, PSM)),]
                        res[3,2,j] <- effectiveSize(as.mcmc(apply(parts, 1, function(x) sum(table(x) * log(table(x))))))
                        partM[[j]] <- est_par2
                        res[3,3,j] <- effectiveSize(as.mcmc(apply(parts, 1, function(x) length(unique(x)))))
                        res[3,4,j] <- VI(partOPT+1, est_par + 1)
                        res[3,5,j] <- VI(partOPT + 1, est_par2 + 1)
                        
                        #------ marginal -- large m
                        
                        time_temp <- system.time(mod4 <- main_univ_MFM_gnk(Y = data_sort, niter = niter, nburn = nburn, thin = 1,
                                                                           hyperparam = c(0, 5, 2, 1, 0, 5, 2, 1), gamma = gamma, 
                                                                           m = 100, Vnt_MAR = Vnt_ABC[[j]][1,-1]))[1]
                        
                        res[4,1,j] <- time_temp / (niter - nburn) 
                        parts <- mod4$clust
                        PSM <- compute_psm(parts)
                        est_par <- parts[which.min(compute_dist(parts, PSM)),]
                        est_par2 <- parts[which.min(VI_LB(parts, PSM)),]
                        res[4,2,j] <- effectiveSize(as.mcmc(apply(parts, 1, function(x) sum(table(x) * log(table(x))))))
                        partM2[[j]] <- est_par2
                        res[4,3,j] <- effectiveSize(as.mcmc(apply(parts, 1, function(x) length(unique(x)))))
                        res[4,4,j] <- VI(partOPT+1, est_par + 1)
                        res[4,5,j] <- VI(partOPT + 1, est_par2 + 1)
                        
                      }
                      cat("finished ", i, "\n", file = "print_results.txt", append = TRUE)
                      list(res, list(partABCad, partABCad2, partM, partM2, partT))                  
                    }

stopImplicitCluster()
stopCluster(cl)
save.image(file = "workspace_simulations_gnk_MFM.Rdata")
