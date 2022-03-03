I created a Bayesian classifier of spectral images of Mars. The data can be downloaded using this link: http://cs.iupui.edu/~mdundar/CRISM/CRISM_labeled_pixels_ratioed.mat
I assumed a multivariate normal likelihood for the data, with a multivariate normal prior for mu and an inverse Wishart prior for sigma.
x~N(mu,sigma)
mu~N(mu_0, sigma/k)
sigma~W^-1(sigma_0,m)

After completing the derivation, the posterior predictive distribution turned out to be a student-t distribution.

This data is very difficult to deal classify, so I was only able to achieve 75% accuracy, and 55% class level accuracy.
