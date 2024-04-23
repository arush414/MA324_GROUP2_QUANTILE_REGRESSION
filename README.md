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
For a given dataset $(y_i,x_{i1},x_{i2},...,x_{ip})_{i=1}^n$ of n statistical units, a linear regression model assumes that the relationship between the dependent variable $y$ and the vector of regressors $x$ is linear.
i.e.
$$Y = X\omega + \beta$$
for some weights $\omega$.
