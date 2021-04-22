---
layout: post
title: GANs Training Journey Pokemon
github: https://github.com/BenoitLeguay/GAN_IconClass
---

GANs are a framework for teaching a DL model to capture the training data’s distribution so we can generate new data from that same distribution. GANs were invented by Ian Goodfellow in 2014 and first described in the paper [Generative Adversarial Nets](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf). They are made of two distinct models, a *generator* and a *discriminator*. The job of the generator is to spawn ‘fake’ images that look like the training images. The job of the discriminator is to look at an image and output whether or not it is a real training image or a fake image from the generator. During training, the generator is constantly trying to outsmart the discriminator by generating better and better fakes, while the discriminator is working to become a better detective and correctly classify the real and fake images. The equilibrium of this game is when the generator is generating perfect fakes that look as if they came directly from the training data, and the discriminator is left to always guess at 50% confidence that the generator output is real or fake. (@pytorch)

![gan-schema.svg]({{site.baseurl}}/images/gans/gan-schema.svg)



The goal of this training journey is not to create the next Pokemon generation but to observe and compare GANs performance through modification in its architecture, training components, hyper parameters etc.. 

<br /><br />

## Dataset



The Pokemon sprites dataset has advantages, uniform low resolution images with a wide diversity. 

![gan-dataset.png]({{site.baseurl}}/images/gans/gan-dataset.png)

The images are 64x64 resolution, with 3 channels. The only preprocessing made is a minmax scale. The goal of this operation is to normalize images between -1 and 1, this can make training faster and reduces the chance of getting stuck in local optima.

$$x_{norm} = 2*(\frac{x - x_{min}}{x_{max} - x_{min}}) - 1$$<br />





We will compare 3 GANs versions: 

- Deep Convolutional GANs
- Wasserstein GANs with gradient penalty
- Auxiliary Classifier GANs

<br /><br />

## A) DCGANs

DCGANs is simple version of GANs (described hereinabove) that uses convolutional layers for both the discriminator and the generator. Let's see how our DCGAN works on this task.   <br /><br />

**Discriminator:**

![DCGAN discriminator.png]({{site.baseurl}}/images/gans/DCGAN discriminator.png)

Each convolutional block consists of a Conv2d/ Batch Normalization/ LeakyReLu sequence. The output of the *Discriminator* is then fed into a sigmoid function. This gives us a score between 0 and 1, that is, the likelihood to be either a real or a fake sample.   <br />

The loss is the Binary Cross entropy, commonly used for binary classification tasks.

$$L_{D} = \frac{1}{m} \sum^{m}_{i=1}[log D(x^{(i)})+log(1 - D(G(z^{(i)})))]$$

<br />



**Generator:**

![DCGAN generator.png]({{site.baseurl}}/images/gans/DCGAN generator.png)

This network has a Batch Normalization, Dropout, LeakyReLu layers at each convolutional block. We'll discuss the different upsample method in another part. Since my real images are normalized between -1 and 1, I use the Tanh function as output activation. The network takes as input a random vector of size $$Z_{dim} = 100$$ sampled from the normal distribution. Then, we define the generator function that maps the latent vector Z to the data space. <br />

 $$L_G = \frac{1}{m}\sum^{m}_{i=1}-log(D(G(z^{(i)})))$$

<br />



**Unit test**

As a unit test, I like to test my GAN to reproduce a single image. This is also a good comparison tool across multiple GANs architecture, when talking about learning pace mostly. <br />

![dcgan-1p-real.png]({{site.baseurl}}/images/gans/dcgan-1p-real.png) <br />

Our GANs is fed with the same image during the whole training. The dataloader contains 1000 times the same image. <br />

![dcgan-1p-10e.png]({{site.baseurl}}/images/gans/dcgan-1p-10e.png) *10 epochs*

![dcgan-1p-50e.png]({{site.baseurl}}/images/gans/dcgan-1p-50e.png) *50 epochs*

![dcgan-1p-150e.png]({{site.baseurl}}/images/gans/dcgan-1p-150e.png) *150 epochs*

<br /><br />

### 1) Training

![dcgan-10e.png]({{site.baseurl}}/images/gans/dcgan-10e.png) *10 epochs*

![dcgan-30e.png]({{site.baseurl}}/images/gans/dcgan-30e.png) *30 epochs*

![dcgan-70e.png]({{site.baseurl}}/images/gans/dcgan-70e.png) *70 epochs*

![dcgan-110e.png]({{site.baseurl}}/images/gans/dcgan-110e.png) *110 epochs*

![dcgan-150e.png]({{site.baseurl}}/images/gans/dcgan-150e.png) *150 epochs*<br />



<br />

![dcgan-fid.svg]({{site.baseurl}}/images/gans/dcgan-fid.svg) *Frechet Inception Distance over epochs*

The Frechet Inception Distance score, or FID for short, is a metric that calculates the distance between feature vectors calculated for real and generated images<br /><br />



![dcgan-discri.svg]({{site.baseurl}}/images/gans/dcgan-discri.svg) *Discriminator Loss over updates*

   <br /><br />



![dcgan-gen.svg]({{site.baseurl}}/images/gans/dcgan-gen.svg) *Generator Loss over updates*





<br /><br />

  

## B)  WGANs with Gradient Penalty

Wasserstein Generative Adversarial Networks are an extension to the Vanilla GAN, the main modification lies in the Discriminator, here named **Critic**, network objective. Instead of estimating a probability of being a real sample, the network outputs a score that shows the realness of the input. It can be compared to the Value function $$V(s)$$ used in Reinforcement Learning. Hence, the loss is modified accordingly, but first let's talk about the Wasserstein distance. 

The mathematical idea behind WGANs is the Wasserstein distance (or earth mover distance). That is, a measure of distance between 2 distributions. We can sum up the role of the **Generator** as a Wasserstein distance optimizer through iterations, the **Critic** being the W distance function approximation.  

*[...] we define a form of GAN called Wasserstein-GAN that minimizes a  reasonable and efficient approximation of the EM distance, and we theoretically show that the corresponding optimization problem is sound.* (https://arxiv.org/abs/1701.07875)

<br />

The intuition between the gradient penalty comes from the calculation of the Wasserstein distance. I won't cover this here, but in a few words, we need our **Critic** function to be 1-Lipschitz (i.e gradient norm at 1 everywhere). To ensure this we add this gradient penalty term in our critic loss function. 

<br /><br />

**Critic**

![DCGAN discriminator.png]({{site.baseurl}}/images/gans/DCGAN discriminator.png)

As we have seen above, the architecture of the WGANs' Critic does not differ from the Discriminator, except in the output activation, being the Identity function in the WGANs' case.  It is important to mention that I remove the **Batch Normalization Layer** because it takes us away from our objective.

<br />

*[...] but batch normalization changes the form of the discriminator’s problem from mapping a single input to a single output to mapping from an entire batch of inputs to a batch of outputs. Our penalized training objective is no longer valid in this setting, since we penalize the norm of the critic’s gradient with respect to each input independently, and not the entire batch [...]* (https://papers.nips.cc/paper/2017/file/892c3b1c6dccd52936e27cbd0ff683d6-Paper.pdf)

<br /><br />

**Generator:**

![DCGAN generator.png]({{site.baseurl}}/images/gans/DCGAN generator.png)

It does not differ from the **DCGAN** one.  

<br /><br />



## C) Auxiliary Classifier GANs

ACGANs are a type of GAN where you add a label classification module to improve GANs understanding on the real data. This also allows to generate label-based fake samples.  It is an improvement from the Conditional GANs architecture. 

The discriminator seeks to maximize the probability of correctly classifying real and fake images and correctly predicting the class label of a real or fake image. The generator seeks  to minimize the ability of the discriminator to discriminate real and  fake images whilst also maximizing the ability of the discriminator predicting the class label of real and fake images. (@machinelearningmastery)









## D) Architecture & Hyper Parameters

<br />

**Generator upsampling methods**



<br /><br />

**Label smoothing**



<br /><br />

**Number of features in both networks**



<br /><br />

**Convolution Transpose or Upsample and Convolution**



<br /><br />