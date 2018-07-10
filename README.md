VAE on CIFAR
============

Some experiments with VAE on CIFAR dataset.

Observations so far:

- Start with a small dataset subset to early catch any errors. The network should be able to easily remember all samples.
- It seems like a good idea to set the latent vector variance to 0 during prediction. Just make sure that you actually set the variance to 0 - not the logarithm of the variance in which case you actually set the variance to 1! :D
