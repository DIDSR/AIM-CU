#-----------------------------------------
#--author: smriti.prathapan
#-----------------------------------------

library(cusumcharter)
library(ggplot2)
library(RcppCNPy)
library(spc)

#Code for table 4
#-------Find k given h------------

L0 <- 50
h <- 4
k <-  xcusum.crit.L0h(L0, h)
k
#----------------------------------

#Code for table 5
#----------xcusum.ad---------------------------------------
#k <- 0.1
h <- 4
#mu1=0.5
mu0=0
mu1 <- c(0,0.1,0.2,0.3,0.4,0.5,.6,.7,0.8,0.9,1,1.10, 1.20,1.30, 1.40, 1.50, 1.60)
ARL_1 <- sapply(mu0=mu1,mu1,k=k,h=h,sided="two",xcusum.ad)
#h <- 5
#ARL5 <- sapply(mu0=mu1,mu1,k=k,h=h,sided="two",xcusum.ad)
round(cbind(mu0,mu1,ARL_1),digits=2)
#---------------------------------------------------------
