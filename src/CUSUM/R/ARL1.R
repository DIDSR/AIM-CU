
#-----------------------------------------
#--author: smriti.prathapan
#-----------------------------------------

library(cusumcharter)
library(ggplot2)
library(RcppCNPy)
library(spc)

#----------xcusum.ad---------------------------------------
k <- 0.5 #Change this to 1 to run row 3 and 4
h <- 4
#mu1=0.5
mu0=0
mu1 <- c(0,.25,.5, 0.6, 0.61, 0.66, .656,.75,1,1.03,1.1, 1.33, 1.54)
ARL4 <- sapply(mu0=mu1,mu1,k=k,h=h,sided="two",xcusum.ad)
h <- 5
ARL5 <- sapply(mu0=mu1,mu1,k=k,h=h,sided="two",xcusum.ad)
round(cbind(mu0,mu1,ARL4,ARL5),digits=2)



