set.seed(42)

library(ggplot2)
library(ggpubr)
library(doParallel)
library(coda)
library(latex2exp)
library(Rcpp)
library(R.utils)

# Useful functions 

BL <- function(c1, c2){
  n <- length(c1)
  temp_mat <- matrix(0, ncol = n, nrow = n)
  for(i in 1:n){
    for(j in 1:i){
      if((c1[i] == c1[j] & c2[i] != c2[j]) | (c1[i] != c1[j] & c2[i] == c2[j])){
        temp_mat[i,j] <- 1
      }
    }
  }
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
n <- c(100, 250)
# eps <- c(9, 16, 24, 35)
# eps <- c(22.5,37.5)
eps <- sqrt((n) * log(n))
eps0 <- n#c(20, 90, 150)

#------------- 

cl <- makeCluster(ncl, outfile = "message.txt")
registerDoParallel(cl)
writeLines(c(""), "print_results.txt")
writeLines(c(""), "print_time_out.txt")

results2 <- foreach(i = 1:nrep, .packages = c("Rcpp", "RcppArmadillo"), 
                    .export = c("effectiveSize", "withTimeout"), 
                    .noexport = c("adapt_ABC_MCMCw", "ABC_MCMCw", "compute_dist", "VI_LB", "compute_psm")) %dopar% {
                      sourceCpp("code_CPP.cpp")
                      sourceCpp('CPPcode_marginal.cpp')
                      
                      res <- array(0, dim = c(5,5,2))
                      partM <- partABC <- partABC2 <- partABCad <- partABCad2 <- partT <- list()
                      
                      time_out <- T
                      time_idx <- 0
                      state <- "undone"
                      while(isTRUE(time_out)){
                        time_idx <- time_idx + 1
                        cat("finished time", time_idx, "of rep ", i, "\n", file = "print_time_out.txt", append = TRUE)
                        withTimeout(expr = {
                          while(state == "undone"){
                            tryCatch({
                              for(j in 1:2){
                                temp_ass <- sample(c(-3,3), size = n[j], replace = T, prob = c(0.75, 0.25))
                                data <- rnorm(n[j]) + temp_ass
                                indx <- order(data)
                                data_sort <- data[indx]
                                partOPT <- ifelse(temp_ass[indx] == -3, 0, 1)
                                partT[[j]] <- partOPT
                                
                                #------ ABC ------------------------------------------
                                time_temp <- system.time(mod1 <- ABC_MCMCw(data = data_sort, niter = niter, nburn = nburn,
                                                                           theta = 1, sigma = 0.2, hyperparam = c(0, 0.5, 2, 2), 
                                                                           eps = eps[j], p = 2))[1]
                                res[1,1,j] <- time_temp / (niter - nburn) 
                                parts <- mod1$part_results
                                PSM <- compute_psm(parts)
                                est_par <- parts[which.min(compute_dist(parts, PSM)),]
                                est_par2 <- parts[which.min(VI_LB(parts, PSM)),]
                                res[1,2,j] <- effectiveSize(coda::as.mcmc(apply(parts, 1, function(x) sum(table(x) * log(table(x))))))
                                partABCad[[j]] <- est_par2
                                res[1,3,j] <- effectiveSize(coda::as.mcmc(apply(parts, 1, function(x) length(unique(x)))))
                                res[1,4,j] <- VI(partOPT + 1, est_par + 1)
                                res[1,5,j] <- VI(partOPT + 1, est_par2 + 1)
                                
                                #------ ABC 2 -----------------------------------------
                                
                                time_temp <- system.time(mod2 <- ABC_MCMCw(data = data_sort, niter = niter, nburn = nburn,
                                                                           theta = 1, sigma = 0.2, hyperparam = c(0, 0.5, 2, 2), 
                                                                           eps = 0.8 * eps[j], p = 2))[1]
                                
                                res[2,1,j] <- time_temp / (niter - nburn) 
                                parts <- mod2$part_results
                                PSM <- compute_psm(parts)
                                est_par <- parts[which.min(compute_dist(parts, PSM)),]
                                est_par2 <- parts[which.min(VI_LB(parts, PSM)),]
                                res[2,2,j] <- effectiveSize(coda::as.mcmc(apply(parts, 1, function(x) sum(table(x) * log(table(x))))))
                                partABCad[[j]] <- est_par2
                                res[2,3,j] <- effectiveSize(coda::as.mcmc(apply(parts, 1, function(x) length(unique(x)))))
                                res[2,4,j] <- VI(partOPT + 1, est_par + 1)
                                res[2,5,j] <- VI(partOPT + 1, est_par2 + 1)
                                
                                #------ ABC_ad ------------------------------------------
                                
                                time_temp <- system.time(mod3 <- adapt_ABC_MCMCw(data = data_sort, niter = niter, nburn = nburn,
                                                                                 theta = 1, sigma = 0.2, hyperparam = c(0, 0.5, 2, 2), 
                                                                                 eps0 = 1, eps_star = log(length(data)), 
                                                                                 p = 2, m = 100, tau1 = 1000, tau2 = 1000, adapt_fix = TRUE))[1]
                                
                                res[3,1,j] <- time_temp / (niter - nburn) 
                                parts <- mod3$part_results
                                PSM <- compute_psm(parts)
                                est_par <- parts[which.min(compute_dist(parts, PSM)),]
                                est_par2 <- parts[which.min(VI_LB(parts, PSM)),]
                                res[3,2,j] <- effectiveSize(coda::as.mcmc(apply(parts, 1, function(x) sum(table(x) * log(table(x))))))
                                partABCad[[j]] <- est_par2
                                res[3,3,j] <- effectiveSize(coda::as.mcmc(apply(parts, 1, function(x) length(unique(x)))))
                                res[3,4,j] <- VI(partOPT + 1, est_par + 1)
                                res[3,5,j] <- VI(partOPT + 1, est_par2 + 1)
                                
                                
                                #------ ABC_ad ------------------------------------------
                                
                                time_temp <- system.time(mod4 <- adapt_ABC_MCMCw(data = data_sort, niter = niter, nburn = nburn,
                                                                                 theta = 1, sigma = 0.2, hyperparam = c(0, 0.5, 2, 2), 
                                                                                 eps0 = 1, eps_star = log(length(data)), 
                                                                                 p = 2, m = 100, tau1 = 1000, tau2 = 1000, adapt_fix = FALSE))[1]
                                
                                res[4,1,j] <- time_temp / (niter - nburn) 
                                parts <- mod4$part_results
                                PSM <- compute_psm(parts)
                                est_par <- parts[which.min(compute_dist(parts, PSM)),]
                                est_par2 <- parts[which.min(VI_LB(parts, PSM)),]
                                res[4,2,j] <- effectiveSize(coda::as.mcmc(apply(parts, 1, function(x) sum(table(x) * log(table(x))))))
                                partABCad[[j]] <- est_par2
                                res[4,3,j] <- effectiveSize(coda::as.mcmc(apply(parts, 1, function(x) length(unique(x)))))
                                res[4,4,j] <- VI(partOPT + 1, est_par + 1)
                                res[4,5,j] <- VI(partOPT + 1, est_par2 + 1)
                                
                                #------ MAR ------------------------------------------
                                
                                time_temp <- system.time(mod5 <- main_univ_PY(Y = data_sort, niter = niter, nburn = nburn, 
                                                                              thin = 1, m0 = 0, k0 = 0.5, 
                                                                              a0 = 2, b0 = 2, theta = 1, sigma = 0.2))[1]
                                
                                res[5,1,j] <- time_temp / (niter - nburn) 
                                parts <- mod5$clust
                                PSM <- compute_psm(parts)
                                est_par <- parts[which.min(compute_dist(parts, PSM)),]
                                est_par2 <- parts[which.min(VI_LB(parts, PSM)),]
                                res[5,2,j] <- effectiveSize(coda::as.mcmc(apply(parts, 1, function(x) sum(table(x) * log(table(x))))))
                                partABCad[[j]] <- est_par2
                                res[5,3,j] <- effectiveSize(coda::as.mcmc(apply(parts, 1, function(x) length(unique(x)))))
                                res[5,4,j] <- VI(partOPT + 1, est_par + 1)
                                res[5,5,j] <- VI(partOPT + 1, est_par2 + 1)
                                
                                # ------------------------------------------
                                
                                partT[[j]] <- partOPT
                              }
                              state <- "done"
                            }, 
                            error = function(e){state = "undone"})
                          }
                          time_out <- F
                        }, timeout = 86400, onTimeout = "silent")
                      }
                      cat("finished ", i, "\n", file = "print_results.txt", append = TRUE)
                      list(res, list(partM, partABC, partABC2, partABCad, partABCad2, partT))                   
                    }

stopImplicitCluster()
stopCluster(cl)
save.image(file = "workspace_simulations_gaussian.Rdata")
