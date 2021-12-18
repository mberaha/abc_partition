library(doParallel)
library(coda)
library(Rcpp)
set.seed(42)

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
#     generate n = (100, 250) values from a mixture of two g&k distributions
#     estimate a g&k mixture model with two implementations of the adaptive ABC sampler
#     and the marginal sampler, with a Monte Carlo approximation of the prob. of sampling a new value
#

ncl <- 7
niter <- 15000
nburn <- 5000
nupd <- 15000
nrep <- 50
n <- c(100, 250, 1000)
eps <- log(n)^2.5 / 2
cl <- makeCluster(ncl, outfile = "message.txt")
registerDoParallel(cl)
writeLines(c(""), "print_results.txt")

#---------------------
# start the simulation 
results <- foreach(i = 1:nrep, .packages = c("Rcpp", "RcppArmadillo"), 
                   .export = c("effectiveSize"),
                   .noexport = c("ABC_MCMC_gnk", "adapt_ABC_MCMC_gnk", "compute_dist", "update_part_PY", 
                                 "Qnk", "deriv_Qnk", "deriv_QnkX", "optimize_Qnk", "PDFkn", "marginal_gnk")) %dopar% {
                     sourceCpp("code_CPP_gnk.cpp")
                      
                      res <- array(0, dim = c(4, 5, 3))
                      partM <- partM2 <- partABCad <- partABCad2 <- partT <- list()
                      
                      for(j in 1:3){
                        
                        # sample the data
                        temp_ass <- sample(c(0,1), size = n[j], replace = T, prob = c(0.75, 0.25))
                        data <- sapply(temp_ass, function(x) ifelse(x == 1, Qnk(z = rnorm(1), a = 3, b = 0.5, g = 0.4, k = 0.5, c = 0.8), 
                                                       Qnk(z = rnorm(1), a = -3, b = 0.75, g = -0.9, k = 0.1, c = 0.8)))
                        indx <- order(data)
                        data_sort <- data[indx]
                        partOPT <- temp_ass[indx]
                        partT[[j]] <- partOPT
                        
                        #------ ABC adaptive -- fixed adapt
                        
                        time_temp <- system.time(mod1 <- adapt_ABC_MCMC_gnk(data = data_sort, niter = niter, nburn = nburn, theta = 1, sigma = 0.2, 
                                                hyperparam = c(0, 5, 2, 1, 0, 5, 2, 1), nupd = nupd, 
                                                eps0 = 10 * log(n[j]), eps_star = 10 * log(100), p = 1, m = 100, 
                                                tau1 = 1000, tau2 = 1000, adapt_fix = TRUE))[1]
                        
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
                        
                        time_temp <- system.time(mod2 <- adapt_ABC_MCMC_gnk(data = data_sort, niter = niter, nburn = nburn, theta = 1, sigma = 0.2, 
                                                hyperparam = c(0, 5, 2, 1, 0, 5, 2, 1), nupd = nupd, 
                                                eps0 = 10 * log(n[j]), eps_star = 10 * log(100), p = 1, m = 100, 
                                                tau1 = 1000, tau2 = 1000, adapt_fix = FALSE))[1]
                        
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
                        
                        time_temp <- system.time(mod3 <- marginal_gnk(data = data_sort, niter = niter, nburn = nburn,
                                                  theta = 1, sigma = 0.2, m = 10, hyperparam = c(0, 5, 2, 1, 0, 5, 2, 1), nupd = nupd))[1]
                        
                        res[3,1,j] <- time_temp / (niter - nburn) 
                        parts <- mod3$part_results
                        PSM <- compute_psm(parts)
                        est_par <- parts[which.min(compute_dist(parts, PSM)),]
                        est_par2 <- parts[which.min(VI_LB(parts, PSM)),]
                        res[3,2,j] <- effectiveSize(as.mcmc(apply(parts, 1, function(x) sum(table(x) * log(table(x))))))
                        partM[[j]] <- est_par2
                        res[3,3,j] <- effectiveSize(as.mcmc(apply(parts, 1, function(x) length(unique(x)))))
                        res[3,4,j] <- VI(partOPT+1, est_par + 1)
                        res[3,5,j] <- VI(partOPT + 1, est_par2 + 1)
                        
                        #------ marginal -- large m
                        
                        time_temp <- system.time(mod4 <- marginal_gnk(data = data_sort, niter = niter, nburn = nburn,
                                             theta = 1, sigma = 0.2, m = 100, hyperparam = c(0, 5, 2, 1, 0, 5, 2, 1), nupd = nupd))[1]
                        
                        res[4,1,j] <- time_temp / (niter - nburn) 
                        parts <- mod4$part_results
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
save.image(file = "workspace_simulations_gnk.Rdata")

# #-------------------------
load("workspace_simulations_gnk.Rdata")
library(ggplot2)
library(latex2exp)

dim(results[[1]][[1]])
res_array <- array(0, dim = c(50, dim(results[[1]][[1]])),
                   dimnames = list(paste("rep", 1:50),
                                   c("ABCad1", "ABCad2", "MAR", "MAR2"),
                                   c("time", "ESSk", "part_dist", "ESSentropy"),
                                   c("n = 100", "n = 250", "n = 1000")))
for(i in 1:50){
  res_array[i,,,] <- results[[i]][[1]]
}

df_plot <- data.frame(time = c(as.vector(res_array[,,1,1]), as.vector(res_array[,,1,2]), as.vector(res_array[,,1,3])),
                      ESSk = c(as.vector(res_array[,,2,1]), as.vector(res_array[,,2,2]), as.vector(res_array[,,2,3])),
                      part_dist = c(as.vector(res_array[,,4,1]), as.vector(res_array[,,4,2]), as.vector(res_array[,,4,3])),
                      ESSentropy = c(as.vector(res_array[,,3,1]), as.vector(res_array[,,3,2]), as.vector(res_array[,,3,3])),
                      mod = rep(c(rep("ABCad1", 50), rep("ABCad2", 50), rep("M1", 50), rep("M2", 50)), 3),
                      size = rep(n, each = 200))
df_plot <- df_plot[df_plot$size != 1000, ]

# names(df_plot) <- c("time", "rand", "accepted", "ESS_entropy", "type", "n")
# df_plot$time <- c(as.vector(res_array[,,2]), as.vector(res_array[,,5]), as.vector(res_array[,,8]), as.vector(res_array[,,11]))
# df_plot$rand <- c(as.vector(res_array[,,1]), as.vector(res_array[,,4]), as.vector(res_array[,,7]), as.vector(res_array[,,10]))
# df_plot$accepted <- c(as.vector(res_array[,,3]), as.vector(res_array[,,6]), as.vector(res_array[,,9]), as.vector(res_array[,,12]))
# df_plot$ESS_entropy <- c(as.vector(res_array[,,13]), as.vector(res_array[,,14]), as.vector(res_array[,,15]), as.vector(res_array[,,16]))
# df_plot <- df_plot[df_plot$n != 40,]

# neword <- levels(factor(df_plot$n))[c(3,4,1,2)]
p1 <- ggplot(df_plot, aes(x = mod, y = time, fill = factor(size))) +
  geom_boxplot(aes(fill = factor(size)), width = 0.6, fatten = 1, alpha = 0.6) +
  scale_y_continuous(trans = "log10", labels = function(x) format(x, scientific = FALSE)) +
  ylab(TeX('time ($\\log_{10}$ scale)')) +
  xlab(TeX("algorithm")) +
  # ggtitle(TeX('Time for a single realization (avg)')) +
  scale_fill_grey() +
  theme_bw() +
  theme(legend.title = element_blank(),
        panel.grid.minor = element_blank())

p2 <- ggplot(df_plot, aes(x = mod, y = ESSentropy, fill = factor(size))) +
  geom_boxplot(aes(fill = factor(size)), width = 0.6, fatten = 1, alpha = 0.6) +
  # scale_y_continuous(trans = "log10", labels = function(x) format(x, scientific = FALSE)) +
  ylab(TeX('ESS')) +
  xlab(TeX("algorithm")) +
  # ggtitle(TeX('Effective sample size (entropy)')) +
  theme_bw() +
  scale_fill_grey() +
  theme(legend.title = element_blank(),
        panel.grid.minor = element_blank())

p3 <- ggplot(df_plot, aes(x = mod, y = ESSk, fill = factor(size))) +
  geom_boxplot(aes(fill = factor(size)), width = 0.6, fatten = 1, alpha = 0.6) +
  # scale_y_continuous(trans = "log10", labels = function(x) format(x, scientific = FALSE)) +
  ylab(TeX('ESS')) +
  xlab(TeX("algorithm")) +
  # ggtitle(TeX('Effective sample size (number of clusters)')) +
  theme_bw() +
  scale_fill_grey() +
  theme(legend.title = element_blank(),
        panel.grid.minor = element_blank())

p4 <- ggplot(df_plot, aes(x = mod, y = part_dist, fill = factor(size))) +
  geom_boxplot(aes(fill = factor(size)), width = 0.6, fatten = 1, alpha = 0.6) +
  # scale_y_continuous(trans = "log10", labels = function(x) format(x, scientific = FALSE)) +
  ylab(TeX('VI')) +
  xlab(TeX("algorithm")) +
  # ggtitle(TeX('VI distance from the true partition')) +
  theme_bw() +
  scale_fill_grey() +
  theme(legend.title = element_blank(),
        panel.grid.minor = element_blank())
# 
# pdf(file = "plot1.pdf", width = 9, height = 4, onefile = F)
# ggpubr::ggarrange(p1, p4, common.legend = T, legend = "bottom")
# dev.off()
# 
# pdf(file = "plot2.pdf", width = 9, height = 4, onefile = F)
# ggpubr::ggarrange(p2, p3, common.legend = T, legend = "bottom")
# dev.off()

#----------
plot_all <- ggpubr::ggarrange(p1g, p4g, p2g, p3g, p1, p4, p2, p3, nrow = 2, 
                              ncol = 4, common.legend = T, legend = "bottom", heights = c(1, 0.925))
pdf(file = "plot_all.pdf", width = 16, height = 7, onefile = F)
plot_all
dev.off()
