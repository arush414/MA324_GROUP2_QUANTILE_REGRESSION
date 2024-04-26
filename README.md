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

The weights for the quantile regression equation are obtained by solving a linear programming problem formulated by using the above loss function as the target sum to be minimized, subject to constraints $y=X\omega$ where the equations represent the observed dataset.

#### Multiple Quantiles simultaneous prediction
We can extend the linear programming problem to higher dimensions since the equations are linearly independent and the loss is linear. By taking the linear combination of the contraints and the target sum, we can create a larger linear programming problem that simultaneously solves for multiple quantiles since the weights for the quantiles form indpendent equations and do not affect each other. 

### Quantile Decision Trees and Regression Forests
It is straightforward to extend a standard decision tree to provide predictions at percentiles. When a decision tree is fit, store not only the sufficient statistics of the target at the leaf node such as the mean and variance but also all the target values in the leaf node. At prediction, these are used to compute empirical quantile estimates. \
To estimate $F(Y=y|x)=q$ each target in value in $y$ is given a weight. Formally the weight given is
$$l_j=\frac{1}{T}\sum_{t=1}^T\frac{1(y_j\in L(x))}{\sum_i 1(y_i\in L(x))}$$ for leaf $L(x)$
We first find the leaf that it falls into at each tree. Then for each pair in the training data, if it is in the same leaf as the new sample, then the weight is the fraction of samples in the same leaf otherwise, zero.

## Sima Module
We have added a new module called sima module, which includes two libraries - modified sklearn and quantile forest. Modified sklearn implements simultaneous multiple quantile regression which wasn't possible using the original sklearn library.
To use modified sklearn from this module import modified sklearn from sima module.
To initialize the Quantile Regressor , just provide it with list of quantiles that you want the model to predict along with alpha value for regularisation.
model = QuantileRegressor(quantile=quantile,alpha=0) % Here you can give list of quantiles that you want instead of a single quantile for eg [0.3,0.5,0.9]
model.fit(X_train, y_train)

.
## Application
Quantile regression is employed when the assumptions of constant variance in residuals and linearity between the independent and dependent variables, which are essential for linear regression, are violated. Additionally, quantile regression is preferred when the data exhibit heteroscedasticity or when the relationship between the variables is non-linear. By estimating different quantiles of the response variable's distribution, quantile regression offers a more comprehensive understanding of the data, especially in scenarios where linear regression assumptions fail to hold. \
We apply Quantile Regression on housing dataset and list down our observations and results in this project.

#### Pros and Cons

###### Pros :
- Robustness: Quantile regression is robust to outliers and does not assume that the data are normally distributed.
- Flexibility: It allows for the estimation of different quantiles of the response variable's distribution, providing a more comprehensive understanding of the data.
- Handles Heteroscedasticity: Quantile regression is suitable for datasets with heteroscedasticity, where the variance of the residuals is not constant across the range of predictor variables.
- Non-linear Relationships: Unlike linear regression, quantile regression can model non-linear relationships between the independent and dependent variables.
- Distributional Insights: It provides insights into different parts of the response variable's distribution, offering a more nuanced analysis of the data.
###### Cons :
- Computational Complexity: Estimating multiple quantiles can increase computational complexity, especially for large datasets.
- Interpretation Challenges: Interpretation of the results can be more complex compared to linear regression, especially when estimating multiple quantiles.
- Requirement of Larger Sample Sizes: Quantile regression may require larger sample sizes compared to linear regression to obtain reliable estimates, especially for extreme quantiles.
- Limited Software Support: While many statistical software packages support linear regression, support for quantile regression may be limited, making implementation more challenging.

#### Setup
To run this code, make sure you have the necessary dependencies installed by using \
`pip install -r requirements.txt`
