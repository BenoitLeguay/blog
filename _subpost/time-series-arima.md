---
title: Time Series ARIMA
layout: post
permalink: "/time-series/arima"
github: https://github.com/BenoitLeguay/time-series
---

*ARIMA* is a model used for both understand and forecasting a time-series. It stands for **A**utoRegressive **I**ntegrated **M**oving **A**verage and is a generalization of the *ARMA* model for non-stationary series. *ARIMA* has 3 main components, we will describe each of them.

To visualize models performances, we'll use the *daily minimum temperature* dataset. 

![arima dset]({{ site.baseurl }}/images/ARIMA/arima-dset.png)

<br/>

<br/>

#### Autoregressive Model

An autoregressive model predicts an outputs as being a linear combination of its previous values, called *lags*. It is often written as *AR (p)* where *p* is the regressive order, that is the number of lags to be used as predictors. It is a critic parameter to be found before any forecasting work. <br/>

Mathematically, the model can be represented by:

$$Y_t=\alpha + \beta_1Y_{t-1} + \beta_2Y_{t-2} +..+ \beta_pY_{t-p} + \epsilon_t$$

where $$\beta_i$$ is the learned weight associated to the $$Y_{t-i}$$ term, the $$i^{th}$$ lag. $$\alpha$$ is the intercept term, that allows the model to not go through the origin. $$\epsilon_t$$ is the white noise at $$t$$, sampled from a Gaussian distribution $$\mathcal{N}(0.0, \sigma²)$$. 

<br/>

To find the order of your AR model, or *p*, you use the *Partial Autocorrelation Function* (PACF). It gives the correlation of a stationary time series with its own lags, having the intermediates dependence removed. Given a time series $$Y_t$$, the partial autocorrelation of lag $$i$$ is the autocorrelation between $$Y_t$$ and $$Y_{t+i}\forall t$$ with the linear dependence of $$Y_t$$ between $$Y_{t+1}$$ and $$Y_{t+i-1}$$ removed. If the partial autocorrelation function (PACF) of the differenced series displays a sharp cutoff and/or the lag-1 autocorrelation is positive (*i.e. if the series appears slightly "underdifferenced"*) then consider adding one or more AR terms to the model. The lag beyond which the PACF cuts off is the indicated number of AR terms.

![arima pacf]({{ site.baseurl }}/images/ARIMA/arima-pacf.png)

The Partial Autocorrelation plot here above shows that a strong partial correlation until order 2 and then it gets very low. That leads us to choose $$p=2$$ for our model.

<br/>

After training our autoregressive model on the *daily minimum temperature* we obtain the hereunder result. We can find the following metrics values: *Schwarz information criterion* (SIC), *Akaike information criterion* (AIC), and the *Hannan-Quinn information criterion* (HQIC). I add the *Mean Squared Error* (MSE), *Mean Absolute Error* (MAE), and *Root Mean Squared Error* (RMSE). 

![arima pacf]({{ site.baseurl }}/images/ARIMA/arima-ar.png)

<br/>

<br/>

#### Integrated

The Integrated term in *ARIMA* means that the model performs zero, one or more difference on the series (if necessary) before any further calculation. Thus, we need to estimate *d* the number of differencing required. As we have seen in the [previous time series article]({{site.baseurl}}/time-series/basics), differencing is a method to make a series stationary, that is having a mean and standard deviation that does not change in time. So why do we need to achieve stationarity ? <br/>

You can see it as removing highly correlated variable when training a linear regression. If the mean and standard deviation change over time, you have a dependency on the past values and this won't help your model. Typically, trend is a non-stationary factor, *ARIMA* does not account for trends and will not be able to predict.

Top sum up, the *ARIMA* model include a component that deals smartly with stationarity to improve performance.

<br/>

<br/>

#### Moving-Average Model

A Moving-Average model predicts an outputs based on a linear combination of prediction residuals (*ie error*) on past values. We defined it as *MA(q)*, where *q* is the order, that is basically the number of past residuals we use to estimate a series points.  As for the AR model, finding the optimal *q* is a critic part of the algorithm since it can highly influence predicted values and computation cost.  <br/>

Mathematically, we define a moving average process as,

$$Y_t=\mu + \epsilon_t + \theta_1\epsilon_{t-1} + \theta_2\epsilon_{t-2} + .. + \theta_q\epsilon_{t-q}$$

where $$\mu$$ is the mean of the series. $$\epsilon_{t-i}$$ is the $$i^{th}$$ residuals term when calculating $$Y_t$$, $$\epsilon_t = Y_t - \hat{Y}_t$$ we assume that $$\epsilon \sim \mathcal{N(0, 1)}$$. $$\theta_i$$ is the learned parameter associated to the $$i^{th}$$ residual term. 

<br/>

To find *q* we use the Autocorrelation function (ACF). That is the correlation between points separated by *n*. It tells you how correlated points are with each other depending on how many steps you separate them. This function gives strong information on the order of our *Moving Average* model. If the autocorrelation function (ACF) of the differenced series displays a sharp cutoff and/or the lag-1 autocorrelation is negative (*i.e. if the series appears slightly "overdifferenced"*) then consider adding an MA term to the model. The lag beyond which the ACF cuts off is the indicated number of MA terms.

![arima acf]({{ site.baseurl }}/images/ARIMA/arima-acf.png)

The MA(2) model trained on our dataset produce the above results. It produces worse results than the AR(2) model. 

![arima ma]({{ site.baseurl }}/images/ARIMA/arima-ma.png)



#### ARIMA

The ARIMA is the combination of the 3 modules we explained. 