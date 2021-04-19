---
layout: post
title: GANs Training Journey Pokemon
github: https://github.com/BenoitLeguay/GAN_IconClass
---

GANs are a framework for teaching a DL model to capture the training data’s distribution so we can generate new data from that same distribution. GANs were invented by Ian Goodfellow in 2014 and first described in the paper [Generative Adversarial Nets](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf). They are made of two distinct models, a *generator* and a *discriminator*. The job of the generator is to spawn ‘fake’ images that look like the training images. The job of the discriminator is to look at an image and output whether or not it is a real training image or a fake image from the generator. During training, the generator is constantly trying to outsmart the discriminator by generating better and better fakes, while the discriminator is working to become a better detective and correctly classify the real and fake images. The equilibrium of this game is when the generator is generating perfect fakes that look as if they came directly from the training data, and the discriminator is left to always guess at 50% confidence that the generator output is real or fake. (@pytorch)



The goal of this training journey is not to create the next Pokemon generation but to observe and compare GANs performance through modification in its architecture, training components, hyper parameters etc.. 



## Dataset



The Pokemon sprites dataset has advantages, uniform low resolution images with a wide diversity. 

![gan-dataset.png]({{site.baseurl}}/images/gans/gan-dataset.png)

The images are 64x64 resolution, with 3 channels. The only preprocessing made is a minmax scale. The goal of this operation is to normalize images between -1 and 1, this can make training faster and reduces the chance of getting stuck in local optima.

$$x_{norm} = 2*(\frac{x - x_{min}}{x_{max} - x_{min}}) - 1$$





We will compare 3 GANs versions: 

- Deep Convolutional GANs
- Wasserstein GANs with gradient penalty
- Auxiliary Classifier GANs



## A) DCGANs

DCGANs is simple version of GANs (described hereinabove) that uses convolutional layers. Let's see how our DCGAN works on this task. 

**Discriminator:**

![DCGAN discriminator.png]({{site.baseurl}}/images/gans/DCGAN discriminator.png)

Each convolutional block consists of a Conv2d/ Batch Normalization/ LeakyReLu sequence. The output of the *Discriminator* is then fed into a sigmoid function. This gives us a score between 0 and 1, that is, the likelihood to be either a real or a fake sample. 



**Generator:**

![DCGAN generator.png]({{site.baseurl}}/images/gans/DCGAN generator.png)

Each convolutional block consists of a ConvTranspose2d/ Batch Normalization/ Dropout/ LeakyReLu sequence. Since my real images are normalized between -1 and 1, I use the Tanh activation function as output for my *Generator*. 



As a unit test, I like to test my GAN to reproduce a single image. This is also a good comparison tool across multiple GANs architecture when talking about learning pace mostly. 

### 1) Training

![dcgan-10e.png]({{site.baseurl}}/images/gans/dcgan-10e.png) *10 epochs*

![dcgan-30e.png]({{site.baseurl}}/images/gans/dcgan-30e.png) *30 epochs*

![dcgan-70e.png]({{site.baseurl}}/images/gans/dcgan-70e.png) *70 epochs*

![dcgan-110e.png]({{site.baseurl}}/images/gans/dcgan-110e.png) *110 epochs*

![dcgan-150e.png]({{site.baseurl}}/images/gans/dcgan-150e.png) *150 epochs*





![dcgan-fid.svg]({{site.baseurl}}/images/gans/dcgan-fid.svg) *Frechet Inception Distance over epochs*

The Frechet Inception Distance score, or FID for short, is a metric that calculates the distance between feature vectors calculated for real and generated images



![dcgan-discri.svg]({{site.baseurl}}/images/gans/dcgan-discri.svg) *Discriminator Loss over updates*

$$L_{D} = \frac{1}{m} \sum^{m}_{i=1}[log D(x^{(i)})+log(1 - D(G(z^{(i)})))]$$



![dcgan-gen.svg]({{site.baseurl}}/images/gans/dcgan-gen.svg) *Generator Loss over updates*

 $$L_G = \frac{1}{m}\sum^{m}_{i=1}-log(D(G(z^{(i)})))$$



## B)  WGANs with GP



