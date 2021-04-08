---
layout: post
title: Generative Adversarial Networks (GANs)
github: https://github.com/BenoitLeguay/GAN_IconClass
---

Curious about GANs framework, I wanted to deal with it in depth. I completed the online course ([available on Coursera](https://www.coursera.org/specializations/generative-adversarial-networks-gans)), and then I decided to implement most of framework versions (vanilla DCGANs, WGANs, Conditional GANs etc...)

GANs are powerful architecture that can learn to generate data with same statistics of a given set. They have many applications: generate unreal human faces based on true ones, up-scale image or video resolution, create DNA sequences with certain properties,  etc... 

The main idea behind GAN is to use a *Generator* that creates fake examples and a *Discriminator* that tries to find these fake examples among the reals. Through an iterative process, it's likely that the generator creates more accurate example, and the discriminator classifies better fake and reals examples. Both the *Discriminator* and the *Generator* are neural networks. This duel between our 2 networks involves a *Game Theory* component, that we'll discuss later on. 



We will start with a very basic example, using a simple GAN architecture and dataset.

### 1) Vanilla GAN

**Dataset**

Let's start with the dataset, I generate a population from a Gaussian Multivariate distribution with random mean and covariance. For visualization purpose, I make this example with 2-dimensional data. 

![gans-real.png]({{site.baseurl}}/images/gans/gans-real-2d.png)

*Real examples (2-dimensional)*

About the architecture, we've got 2 fully connected neural networks with 2 hidden layers. 

**GANs architecture**

*Generator*

The generator takes a random vector as input of size *z* and outputs a vector having the same size as $$x_i$$. 

*Discriminator*

The discriminator takes a input shape vector (real and fake) and output a classification score. 



Thanks to this example, we can easily observe and compare our real and fake data. Here below we have the evolution of the fake generated during the training process:

![gans-fake-real.gif]({{site.baseurl}}/images/gans/gans-fake-real.gif)

*Generated fake along with real examples regarding epochs*

GANs learn the distribution statistics and can output almost real example with a quit good variety. Obviously the goal is to produce diversified example and to prevent mode collapse.  

A **mode collapse** refers to a generator model that is only capable of generating one or a small subset of different outcomes, or modes. The output resembles a real one, the discriminator is fooled, but the Generator can only imitate this little area of the distribution. 

![gans-mode_collapse.png]({{site.baseurl}}/images/gans/gans-mode_collapse.png)

*Example of a Generator suffering from mode collapse*



An interesting parameter to play with in this example is the dimension of the Z input. Z is the noise vector, a random normal generated vector that is the *Generator's* input. The bigger the dimension of Z, the more information is contained in it and thus easier it is for the *Generator* to achieve good diversity. Let's compare 2 training process with Z having different lengths.  

![gans-zdim-1.gif]({{site.baseurl}}/images/gans/gans-zdim-1.gif)

*Z dimension: 1*

![gans-zdim-100.gif]({{site.baseurl}}/images/gans/gans-zdim-100.gif)

*Z dimension: 100*

On one side, the generated population lacks diversity, GANs do well inferring the overall trend of the data but is unable to capture the variance on another dimension. On the other side, our GANs reach a better variety but are more biased toward the training set.

# 2) DCGAN & WGAN

(in progress)







### 3) Training Journey



(https://labs.brill.com/ictestset/)

#### Initial

##### Intro

Firstly we will work on initial dataset. They are gray-scale images of decorated initial letters with small amount of text around. 

![image-20210218163403957]({{site.baseurl}}/images/gans/image-20210218163403957.png)

The task can be tricky due to the variance among the letter position, style and size. We would like our GAN to output real letters with decoration around and text in the background. We do not have any labeled on these photos.

**Dataset size: 3000 examples**

#### 25 Jan 2021

I try a Wasserstein GAN with Gradient Penalty. Below you can see the network architecture I use for the generator.  I used a **ConvTranspose**-**BatchNorm**-**ReLU** structure for every *Generator's* layer and a **Conv2d**-**BatchNorm**-**ReLU** for the Critic/ Discriminator. 

![image-20210218164518666]({{site.baseurl}}/images/gans/image-20210218164518666.png) 

```python
{	
	'gen': {'n_feature': 64, 
			'n_channel': 1, 
			'lr': 0.0001, 
			'betas': (0.9, 0.999)},
 	'critic': {	'n_channel': 1, 
               	'n_feature': 64,
  				'lr': 0.0001, 
               	'betas': (0.9, 0.999)},
 	'z_dim': 50,
 	'gradient_penalty_factor': 2,
}
```

![image-20210218171538806]({{site.baseurl}}/images/gans/image-20210218171538806.png) **50 epoch:** it has already a grasp on the general structure of the image. The result are encouraging. 



![image-20210218172439604]({{site.baseurl}}/images/gans/image-20210218172439604.png)**100 epoch:** it starts creating more complex structure on the top left corner, trying to imitate a letter.





![image-20210218165237866]({{site.baseurl}}/images/gans/image-20210218165237866.png) Though It is not able to outputs real letters and suffers from mode collapse:



**I might be forced to work on training stability to have better results.**

#### 12 Feb 2021



I modify some of the GAN components:

- Add **Dropout** for the generator. New layer structure:  **ConvTranspose**-**BatchNorm**-**Dropout**-**ReLU**.

- Add **Instance Noise** to the model (https://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/).

- Go from **ReLU** to **LeakyReLU** for the generator. 

- Add **Gaussian initialization** for network weights.

  

![image-20210222164233489]({{site.baseurl}}/images/gans/image-20210222164233489.png) 

**50 epoch:** The results seem better with this new architecture. It reaches more precise images with less epoch. It also outputs different "letter" size and shape which is good especially since we want to avoid **mode collapse**

![image-20210222163040584]({{site.baseurl}}/images/gans/image-20210222163040584.png) ![image-123]({{site.baseurl}}/images/gans/image-123.png) ![image-20210222171941566]({{site.baseurl}}/images/gans/image-1234.png)

**300 epoch:** The result is good in my opinion. Since the classifier has no label for each letter it is complicated to be sharp concerning letter precision. Though, it is still able to output look-alike letters as we can see above: **H, D, B, N etc..** 

On the real set, we often see a ruler on one side (for dimension purpose). It is funny to see it on the fake generated. 

Even though I'm quite happy with the result I would like my letters to be more realistic and the image less blurry. I should maybe work on a letter classification task (Conditional GAN). Also I may want to explore architecture that allows bigger image generation (*Progressive Growing GAN*). Finally, as a reinforcement learning enthusiast, I'm curious about using **experience replay**  in the context of GAN.