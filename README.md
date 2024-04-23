# MA324_GROUP2_QUANTILE_REGRESSION 

### Objective

Utilizing Quantile Random Forest to implement and deploy quantile regression on a dataset represents a sophisticated approach to data analysis. Quantile regression, an advanced statistical technique, facilitates the estimation of various quantiles of the response variable measurements, providing a nuanced understanding of the data distribution. This project aims to elucidate the process of applying quantile regression using the Quantile Random Forest algorithm, offering a comprehensive analysis of its implementation and utility.

### Disclaimer
The work is just for exploration purpose.  Any significant output can not be published or made commercialised without mentor's consent. You agree with the fact that it is fully mentor's discretion whether future extension of the work will bring you as a contributor.

## Background
#### Quantiles
We define the $\tau$ th quantile of $Y$ by
$$q_Y (\tau) = F^{-1}_Y(\tau) = \inf{\{y:F_Y(y)\geq\tau\}}$$
where $F_Y(y)=P(Y \geq y)$ i.e. the cumulative distribution function.

#### Linear Regression
For a given dataset $\{y_i,x_{i1},x_{i2},...,x_{ip}\}_{i=1}^n $ of n statistical units, a linear regression model assumes that the relationship between the dependent variable $y$ and the vector of regressors $x$ is linear.
i.e.
$$Y = X\omega + \beta$$
for some weights $\omega$ as slope and intercept $\beta$.

## Quantile Regression
As a linear model, QuantileRegression gives linear predictions $\hat{y}(\omega,X)=X\omega$ for the $q$-th quantile, $q \in (0,1)$. The weights or coefficients $\omega$ are then found by the following minimization problem:
$$\min_{\omega} \frac{1}{n_{\text{samples}}}\sum_{i}PB_q(y_i-X_i\omega)+\alpha||\omega||_1$$

Here, the loss consists of Pinball Loss PB (also called linear loss, optimal for quantiles) and L1 penalty controlled by parameter $\alpha$. The formula for the loss is:

$$
PB_q(t) = q\max(t,0)+(1-q)\max(-t,0)=t(q-1_{t<0})
$$

As the pinball loss is only linear in the residuals, quantile regression is much more robust to outliers than squared error based estimation of the mean.
The use of Pinball Loss for quantile regression is industry standard. The proof of optimality can be cross-verified in the following papers:
- https://doi.org/10.1016/j.ijforecast.2009.12.015
- https://arxiv.org/pdf/1102.2101.pdf
