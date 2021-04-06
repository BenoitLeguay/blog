---
layout: post
title: Time-Series
permalink: "/time-series/basics"
---

## Time series basic manipulation

We will use a common and widely used dataset: *Monthly anti-diabetic drug sales in Australia*

![ts_drugs]({{ site.baseurl }}/images/ts_drugs.png)

We observe a strong upwards trend and a seasonality, a pattern that repeats each years. 

![box_plot_drugs.png]({{ site.baseurl }}/images/box_plot_drugs.png)

These box plots are smart and simple ways to visualize the trend and the seasonality among our time series. 



#### 1) Decompose a time series

We can decompose a time series with convolution filter applied to the data. There are 2 ways of doing it: additional decompose and multiplicative decompose. It depends on the assumption we choose, either we consider our series to be in this form  $$Y(t) = T(t)+S(t)+e(t)$$  or $$Y(t) = T(t)*S(t)*e(t)$$ where $$T$$ is the trend, $$S$$ is the seasonality and $$e$$ is the error (or the residuals).  For breakdown, firstly we estimate the trend with convolutional filters, then by removing this trend we can easily find the seasonality. 

![ts_s_decompose.png]({{ site.baseurl }}/images/ts_s_decompose.png)

*The decomposition using the multiplicative assumption*



#### 2) Make a time series stationary

A stationary time series is one whose properties do not depend on the time at which the series is observed.   We want to make a time series stationary before performing any kind of regression. In other words, it is important that the statistical properties of the time series we want to predict does not change over time. Independence is a strong property when training for regression purpose. 

There are several ways to obtain stationarity, we can directly apply a function to our series or differencing it. For example, the square root of a non-stationary function can make it stationary. 

![ts_s_root.png]({{ site.baseurl }}/images/ts_s_root.png)

One other way to achieve non stationarity is differencing. The order of difference is an hyper parameter to choose, we want the minimum order that make our series stationary. 

 ![ts_diff.png]({{ site.baseurl }}/images/ts_diff.png)

To check for stationarity among a time series we can use statistical test such as Augmented Dickey Fuller test (ADF Test). 



#### 3) Remove trend from a time series

Identify and remove trend can be useful for multiple reasons. Removing trend can help modeling the data by simplify it. The trend can be used as an external feature and help the model in the understanding. The trend also contains information about the series shape and behaviour and it can be nice to retrieve it easily.    

Here I show 2 techniques to remove trend from time series. Firstly, we can use the linear regression to estimate the trend of the series and then subtracts it. In the example bellow, we fit a linear function (blue) to our data (orange), then we simply subtract  the model from the time series. 

 ![ts_lr.png]({{ site.baseurl }}/images/ts_lr.png)

The other common way is to subtract the trend from the seasonality decomposition we just saw. 

 ![ts_se_decomp.png]({{ site.baseurl }}/images/ts_se_decomp.png)



#### 4) Remove seasonality from a time series

On the other hand, removing seasonality can be interesting in some scenarios.  When using multivariate model with multiple time series, each of them can have its own seasonality and it becomes impracticable to  deal with it. Sometimes, seasonality can misleads our series understanding regarding the context. If we want to study real estate prices from a large point of view, the rise during the summer does not show any economic signal.  

Now we learned about detrend a time series, we are going to focus about removing seasonality. Once again I'll show 2 ways of doing so. 

One very easy method is to find the seasonality period, (in our case 12 months), and to compute a rolling average with a the period as the window. 

 ![ts_deseason_rlg_avg.png]({{ site.baseurl }}/images/ts_deseason_rlg_avg.png)



A more sophisticated technique is to use the seasonality from the  decomposition and operate a subtraction.  

 ![ts_sea_decomp_sea_decomp.png]({{ site.baseurl }}/images/ts_sea_decomp_sea_decomp.png)



#### 5) Handle missing value

When trying to model our time series it is important to remove missing value before. Here we show multiple ways to handle them: forward fill, backward fill, interpolation.

 ![ts_mis_value.png]({{ site.baseurl }}/images/ts_mis_value.png)

