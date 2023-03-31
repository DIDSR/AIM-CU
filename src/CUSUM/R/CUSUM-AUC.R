library(cusumcharter)
library(ggplot2)
library(RcppCNPy)

AUC <- npyLoad("/home/smriti.prathapan/R/CUSUM example/test_AUC-Day0-200.npy")
mu <- mean(AUC[1:100])
sigma <- sd(AUC[1:100])

CU <- cusum_control(AUC, target=0.863, std_dev=sigma, k=0.05, h=0.5)

out <- write.csv(CU,  "/home/smriti.prathapan/note1/AUC_CUSUM200-hk.csv")


p6 <- cusum_control_plot(CU, 
                         xvar=obs, 
                         facet_var = "positive and negative changes", 
                         title_txt= "Highlights above the control limits")
p6


