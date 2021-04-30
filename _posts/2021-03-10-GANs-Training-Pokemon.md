---
layout: post
title: GANs Training Journey Pokemon
github: https://github.com/BenoitLeguay/GAN_IconClass
---



GANs are a framework for teaching a DL model to capture the training data’s distribution so we can generate new data from that same distribution. GANs were invented by Ian Goodfellow in 2014 and first described in the paper [Generative Adversarial Nets](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf). They are made of two distinct models, a *generator* and a *discriminator*. The job of the generator is to spawn ‘fake’ images that look like the training images. The job of the discriminator is to look at an image and output whether or not it is a real training image or a fake image from the generator. During training, the generator is constantly trying to outsmart the discriminator by generating better and better fakes, while the discriminator is working to become a better detective and correctly classify the real and fake images. The equilibrium of this game is when the generator is generating perfect fakes that look as if they came directly from the training data, and the discriminator is left to always guess at 50% confidence that the generator output is real or fake. (@pytorch)

![gan-schema.svg]({{site.baseurl}}/images/gans/gan-schema.svg)

<br />

The goal of this training journey is not to create the next Pokemon generation but to observe and compare GANs performance through modification in its architecture, training components, hyper parameters etc.. 

<br />

We will compare 3 GANs versions: 

- Deep Convolutional GANs
- Wasserstein GANs with gradient penalty
- Auxiliary Classifier GANs

<br /><br />

## Dataset

The Pokemon sprites dataset has advantages, uniform low resolution images with a wide diversity. 

![gan-dataset.png]({{site.baseurl}}/images/gans/gan-dataset.png)

The images are 64x64 resolution, with 3 channels. The only preprocessing made is a minmax scale. The goal of this operation is to normalize images between -1 and 1, this can make training faster and reduces the chance of getting stuck in local optima.

$$x_{norm} = 2*(\frac{x - x_{min}}{x_{max} - x_{min}}) - 1$$

<br /><br />

## A) DCGANs

DCGANs is simple version of GANs (described hereinabove) that uses convolutional layers for both the discriminator and the generator. Let's see how our DCGAN works on this task.   <br /><br />

The generator tries to minimize the following function while the discriminator tries to maximize it:

$$E_x[log(D(x))] + E_z[log(1 - D(G(z)))]$$

<br />

#### **Discriminator:**

![DCGAN discriminator.png]({{site.baseurl}}/images/gans/DCGAN discriminator.png)

Each convolutional block consists of a Conv2d/ Batch Normalization/ LeakyReLu sequence. The output of the *Discriminator* is then fed into a sigmoid function. This gives us a score between 0 and 1, that is, the likelihood to be either a real or a fake sample.   <br />

$$L_D = E_x[log(D(x))] + E_z[log(1 - D(G(z)))]$$

It can be seen as a sum of 2 binary cross entropy loss where labels are ones for the real examples and zeros for the fakes.  

<br />



#### **Generator:**

![DCGAN generator.png]({{site.baseurl}}/images/gans/DCGAN generator.png)

This network has a Batch Normalization, Dropout, LeakyReLu layers at each convolutional block. We'll discuss the different upsample method in another part. Since my real images are normalized between -1 and 1, I use the Tanh function as output activation. The network takes as input a random vector of size $$Z_{dim} = 100$$ sampled from the normal distribution. The generator function maps the latent vector Z to the data space. <br />

The loss function is different from the minimax one we defined before, but the general idea remains. This comes from the original paper, it avoids vanishing gradient early in training.  The $$D(x)$$ terms is removed since it is invariant to the generator.  

 $$L_G = E_z[-log(D(G(z)))]$$

<br />



#### **Unit test**

As a unit test, I like to make my GAN to reproduce a single image. This is also a good comparison tool across multiple GANs architectures, when talking about learning pace mostly. <br />

![dcgan-1p-real.png]({{site.baseurl}}/images/gans/dcgan-1p-real.png) <br />

Our GANs is fed with the same image during the whole training. The dataloader contains 1000 times the same image. <br />

![dcgan-1p-10e.png]({{site.baseurl}}/images/gans/dcgan-1p-10e.png) *10 epochs*

![dcgan-1p-50e.png]({{site.baseurl}}/images/gans/dcgan-1p-50e.png) *50 epochs*

![dcgan-1p-150e.png]({{site.baseurl}}/images/gans/dcgan-1p-150e.png) *150 epochs*



Now we know that our DCGANs flow works we can train it on the whole dataset.

<br /><br />

#### Training example

![dcgan-10e.png]({{site.baseurl}}/images/gans/dcgan-ex-e10.png) *10 epochs*

![dcgan-50e.png]({{site.baseurl}}/images/gans/dcgan-ex-e50.png) *50 epochs*

![dcgan-100e.png]({{site.baseurl}}/images/gans/dcgan-ex-e100.png) *100 epochs*

![dcgan-250e.png]({{site.baseurl}}/images/gans/dcgan-ex-e250.png) *250 epochs*

![dcgan-400e.png]({{site.baseurl}}/images/gans/dcgan-ex-e400.png) *400 epochs*

![dcgan-800e.png]({{site.baseurl}}/images/gans/dcgan-ex-e800.png) *800 epochs*



<br />

<br />

![dcgan-discri.svg]({{site.baseurl}}/images/gans/dcgan-ex-dloss.png)<br /> *Discriminator Loss over updates*

<br />

![dcgan-gen.svg]({{site.baseurl}}/images/gans/dcgan-ex-gloss.png)<br /> *Generator Loss over updates*

<br />

![dcgan-ex-facc.png]({{site.baseurl}}/images/gans/dcgan-ex-facc.png)<br />*Discriminator accuracy on fake examples*

<br />

![dcgan-ex-racc.png]({{site.baseurl}}/images/gans/dcgan-ex-racc.png)<br />*Discriminator accuracy on real examples*

<br />

![dcgan-ex-fid.png]({{site.baseurl}}/images/gans/dcgan-ex-fid.png) <br />*Frechet Inception Distance over epochs*

The Frechet Inception Distance score, or FID for short, is a metric that calculates the distance between feature vectors calculated for real and generated images. It uses the *Inception_v3* deep neural network to extract a latent space of every images.

<br /><br />

  

## B)  WGANs with Gradient Penalty

Wasserstein Generative Adversarial Networks are an extension to the Vanilla GAN, the main modification lies in the Discriminator, here named **Critic**, network objective. Instead of estimating a probability of being a real sample, the network outputs a score that shows the realness of the input. It can be compared to the Value function $$V(s)$$ used in Reinforcement Learning. Hence, the loss is modified accordingly, but first let's talk about the Wasserstein distance. 

Indeed, the mathematical idea behind WGANs is the Wasserstein distance (or earth mover distance). That is, a measure of distance between 2 distributions. In this context, we can see the **Critic** as a function approximation, useful for the Wasserstein distance. 

*[...] we define a form of GAN called Wasserstein-GAN that minimizes a reasonable and efficient approximation of the EM distance, and we theoretically show that the corresponding optimization problem is sound.* (https://arxiv.org/abs/1701.07875)

<br />

It is fair to see the WGAN loss function as an approximation of the Wasserstein distance. Indeed, the original paper shows that we can write the Wasserstein formula down as.

$$W_{dist}(P_{real}, P_{fake}) = \frac{1}{K}\sup_{\lVert f \rVert_L<1}E_{x \sim P_{real}}[f(x)] - E_{x \sim P_{fake}}[f(x)]$$

The **Critic** network tries to approximate $$f$$. A good estimation is the key point of this architecture, this is why you often see people updating severals time the Critic for one Generator update. Yet, this formula comes with a constraint. The gradient penalty answers it back. 

 <br />

So, the intuition between the gradient penalty comes from the calculation of the Wasserstein distance. I won't cover the full maths here, but in a few words, we need our **Critic** function to be 1-Lipschitz (i.e gradient norm at 1 everywhere). To ensure this we add this gradient penalty term in our critic loss function. 

Another WGAN version tries to guarantee this constraint by clipping the weight update. Its stability really depends on the clipping hyper parameter and thus the gradient penalty version is considered more reliable. Though, it is computationally expensive. 

<br /><br />

#### **Critic**

![DCGAN discriminator.png]({{site.baseurl}}/images/gans/DCGAN discriminator.png)

As we have seen above, the architecture of the WGANs' Critic does not differ from the Discriminator, except in the output activation, being the Identity function here. It is important to mention that I remove the **Batch Normalization Layer** because it takes us away from our objective.

<br />

*[...] but batch normalization changes the form of the discriminator’s problem from mapping a single input to a single output to mapping from an entire batch of inputs to a batch of outputs. Our penalized training objective is no longer valid in this setting, since we penalize the norm of the critic’s gradient with respect to each input independently, and not the entire batch [...]* (https://papers.nips.cc/paper/2017/file/892c3b1c6dccd52936e27cbd0ff683d6-Paper.pdf)

Perhaps most importantly, the loss of the **Critic** appears to relate to the quality of images created by the Generator.

Specifically, the lower the loss of the **Critic** when evaluating  generated images, the higher the expected quality of the generated  images. This is important as unlike other GANs that seek stability in terms of finding an equilibrium between two models, the WGANs seeks convergence, lowering generator loss.



$$L_c=E_z[C(G(z))]-E_x[C(x)]+\lambda(\lVert \nabla_{\hat{x}}D(\hat{x}) \rVert_2-1)²$$

with $$\hat{x} = \epsilon x+(1-\epsilon) G(z)$$

<br /><br />

#### **Generator:**

![DCGAN generator.png]({{site.baseurl}}/images/gans/DCGAN generator.png)

Not much to say here, it does not differ from the **DCGANs** one.  

Concerning the loss, our generator wants to maximize the critic output, therefore we minimize the negative  average of the critic score among fake samples.

$$L_G= - E_z[C(G(z))]$$

<br /><br />

#### **Unit test**

<br />

**Real samples**

![DCGAN generator.png]({{site.baseurl}}/images/gans/DCGAN generator.png)

<br />

**Training**

![DCGAN generator.png]({{site.baseurl}}/images/gans/wgan-ex-1p-10e.png)*10 epochs*

![DCGAN generator.png]({{site.baseurl}}/images/gans/wgan-ex-1p-25e.png)*25 epochs*

![DCGAN generator.png]({{site.baseurl}}/images/gans/wgan-ex-1p-40e.png)*40 epochs*

![DCGAN generator.png]({{site.baseurl}}/images/gans/wgan-ex-1p-60e.png)*60 epochs*

![DCGAN generator.png]({{site.baseurl}}/images/gans/wgan-ex-1p-100e.png)*100 epochs*

![DCGAN generator.png]({{site.baseurl}}/images/gans/wgan-ex-1p-200e.png)*200 epochs*

<br />

![DCGAN generator.png]({{site.baseurl}}/images/gans/wgan-ex-1p-gloss.png)<br />*Generator Loss over epochs*

<br />

![DCGAN generator.png]({{site.baseurl}}/images/gans/wgan-ex-1p-closs.png)<br />*Critic Loss over epochs*

<br />

![DCGAN generator.png]({{site.baseurl}}/images/gans/wgan-ex-1p-fid.png)<br />*Frechet Inception distance*

<br />

<br />

### Training example

![wgan-10e.png]({{site.baseurl}}/images/gans/wgan-ex-e10.png) *10 epochs*

![wgan-50e.png]({{site.baseurl}}/images/gans/wgan-ex-e50.png) *50 epochs*

![wgan-100e.png]({{site.baseurl}}/images/gans/wgan-ex-e100.png) *100 epochs*

![wgan-250e.png]({{site.baseurl}}/images/gans/wgan-ex-e250.png) *250 epochs*

![wgan-400e.png]({{site.baseurl}}/images/gans/wgan-ex-e400.png) *400 epochs*

![wgan-600e.png]({{site.baseurl}}/images/gans/wgan-ex-e600.png) *600 epochs*

![wgan-800e.png]({{site.baseurl}}/images/gans/wgan-ex-e800.png) *800 epochs*

<br />

<br />

![dcgan-discri.svg]({{site.baseurl}}/images/gans/wgan-ex-closs.png) <br />*Discriminator Loss over updates*

<br />

![dcgan-gen.svg]({{site.baseurl}}/images/gans/wgan-ex-gloss.png) <br />*Generator Loss over updates*

<br />

![dcgan-fid.svg]({{site.baseurl}}/images/gans/wgan-ex-fid.svg)<br /> *Frechet Inception Distance over epochs*

<br /><br />



## C) Auxiliary Classifier GANs

ACGANs are a type of GAN where you add a label classification module to improve GANs understanding on the real data by stabilizing training. This also allows to generate label-based fake samples. It is an improvement from the Conditional GANs architecture. 

The discriminator seeks to maximize the probability of correctly classifying real and fake images and correctly predicting the class label of a real or fake image. The generator seeks  to minimize the ability of the discriminator to discriminate real and fake images whilst also maximizing the ability of the discriminator predicting the class label of real and fake images. (@machinelearningmastery)

All these advantages have a cost since it requires labels. In our case we use 2 different label type: 

- Pokemon Type (water, fire, grass, electric etc...). 18 unique types.
- Pokemon Body Type (quadruped, bipedal, wings, serpentine etc..). 10 unique body types. 



![acgan-schema.png]({{site.baseurl}}/images/gans/acgan-schema.png)



This architecture is compatible with most of GANs' ones, we will describe it on a DCGANs for simplification purpose. 

In the ACGANs, every generated sample (fake) receive a label in addition to the noise. This label adds information that helps the model to create sample based on class. On the other hand, the discriminator tries to predict a label for both real and fake examples. 

Depending on the GANs architecture you compute the associate loss in which you add the auxiliary loss. Typically, we use the negative log likelihood loss since we perform a multi-class prediction.  

<br />

#### **Generator**

![acgan-discri.png]({{site.baseurl}}/images/gans/acgan-discri.png)

<br />

Here is an example that shows how the label affects the generation. I use here the same noise vector with different labels (*grass, water, fire*). It has a grasp on the main color associated with the types.

![acgan-type-comparison.png]({{site.baseurl}}/images/gans/acgan-type-comparison.png)

![acgan-type-comparison2.png]({{site.baseurl}}/images/gans/acgan-type-comparison2.png)



<br />

#### **Discriminator**

![acgan-discri.png]({{site.baseurl}}/images/gans/acgan-discri.png)

To evaluate the **Discriminator** we calculate its accuracy on both auxiliary and adversarial task. Also I add the auxiliary distribution entropy to track failure. Below you have a example for training.

 ![acgan-discriminator.png]({{site.baseurl}}/images/gans/acgan-discriminator.png)



<br />

#### **Unit test**

<br />

In order to test the full flow here (main task + auxiliary task) I need 2 samples, with 2 different labels. We will be working on this dataset:

#### **Real samples**

![DCGAN generator.png]({{site.baseurl}}/images/gans/acgans-1p-real.png)

<br />

#### **Training**

![DCGAN generator.png]({{site.baseurl}}/images/gans/acgans-1p-25e.png)*25 epochs*

![DCGAN generator.png]({{site.baseurl}}/images/gans/acgans-1p-150e.png)*150 epochs*

![DCGAN generator.png]({{site.baseurl}}/images/gans/acgans-1p-200e.png)*200 epochs*

![DCGAN generator.png]({{site.baseurl}}/images/gans/acgans-1p-280e.png)*280 epochs*

![DCGAN generator.png]({{site.baseurl}}/images/gans/acgans-1p-350e.png)*350 epochs*

<br />

![DCGAN generator.png]({{site.baseurl}}/images/gans/acgans-1p-advacc.png)<br />

*Accuracy on main discriminator task*<br />

![DCGAN generator.png]({{site.baseurl}}/images/gans/acgans-1p-fauxacc.png)<br />

*Accuracy on auxiliary discriminator task for fake sample*<br />

![DCGAN generator.png]({{site.baseurl}}/images/gans/acgans-1p-rauxacc.png)<br />

*Accuracy on auxiliary discriminator task for real sample*

the task here is trivial<br />

![DCGAN generator.png]({{site.baseurl}}/images/gans/acgans-1p-loss.png) <br />*Generator and Discriminator Loss over epochs*

<br />

![DCGAN generator.png]({{site.baseurl}}/images/gans/acgans-1p-fid.png)<br />*Frechet Inception distance*

<br />

<br />

### Training example





<br /><br />

## D) Architecture & Hyper Parameters

<br />

#### **1) Hyper parameters** 

Here I talk about the hyper parameters, and regularization techniques I implemented that stabilizes training, avoiding mode collapse or divergence.

<br />

- Neural network weight initialization



<br />

- Optimizer learning rate



<br />

- Neural network features



<br />

- Z space dimension and distribution



<br />

- Label smoothing



<br />

- Instance noise



<br />

<br />

#### **2) Generator upsampling methods**

Convolutional neural networks are originally built to take an image as input, in the Generator case we want to do it the other way. To do so, we have several options. We will explore 3 different methods:

- *Transpose Convolution*

It is the operation inverse to convolution. 

![convtranspose.gif]({{site.baseurl}}/images/gans/convtranspose.gif)

<br />

- *Convolution and Upsample (nearest)*

![upsampling.png]({{site.baseurl}}/images/gans/upsampling.png)



<br />

- *Color Picker* 

Color Picker is a technique I found [here](https://github.com/ConorLazarou/PokeGAN), the idea is to generate separately each *Red Green Blue* channel.  Each is created thanks to 2 components, a color palette and a distribution over this palette, we use the latter to weight the palette and to create a single channel 64x64 matrix. The palette tensor is the same for each channel while the distribution is computed 3 times (*R,G,B*). 

<br />

**Comparison**

![comparison-upsample-convtranspose.png]({{site.baseurl}}/images/gans/comparison-upsample-convtranspose.png)

Here we have fake samples from a WGAN, with both upsample + conv method and convtranspose. The Upsample method seems to create more complex structure. It is a benefit when talking about Pokemon shape since it outputs limbs etc..  but a drawback concerning colors. The generated image has too many colors making the Pokemon unreal. Even with a long training, it appears that this architecture has a hard time creating a uniform color shape. 

The ColorPicker Generator architecture answers this handicap. Indeed it benefits from the complexity given by the upsample method and gets uniform colors with the help of the palette-oriented architecture. 





<br /><br />



![gan-meme.png]({{site.baseurl}}/images/gans/gan-meme.png)

