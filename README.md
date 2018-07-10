VAE on CIFAR
============

Some experiments with VAE on CIFAR dataset.

General observations:

- Start with a small dataset subset to early catch any errors. The network should be able to easily remember all samples.
- It seems like a good idea to set the latent vector variance to 0 during prediction. Just make sure that you actually set the variance to 0 - not the logarithm of the variance in which case you actually set the variance to 1! :D
- Sigmoid can be used to constrain the pixel intensity values of the output image. Those values need to be in range between 0 and 1.

Sigmoid 0-1 pixel value constraints
-----------------------------------

Pixel distribution without sigmoid:

![Pixel distribution without sigmoid](res/no_sigmoid_pixel_dist.png)

Pixel distribution wit sigmoid:

![Pixel distribution without sigmoid](res/sigmoid_pixel_dist.png)

- The figures above show histograms of the pixel intensity values.
- The original image distributions are on the left. The predicted distributions are on the right.
- The orange line shows the histogram for the raw output values while the blue line and bars show the clipped values (clipping is not required for sigmoid).

One might presume that performing clipping might cause most of the probability mass to focus on the edge values 0 and 1 since all values that lied beyond that range would be snapped into one of those values. This is actually the case for the value of 1 where we can observe a large peek.

Such peek can be also observed for the original image distribution, however the plot around this peek seems to be smoother. Similar smoothness can be observed for the sigmoid predictions, but the height of the peek is usually much lower.
