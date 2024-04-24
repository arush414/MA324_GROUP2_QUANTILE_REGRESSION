# from sima import sklearn_modified
from sima.sklearn_modified.sklearn.datasets import make_regression
from sima.sklearn_modified.sklearn.model_selection import train_test_split
from sima.sklearn_modified.sklearn.linear_model import QuantileRegressor
import numpy as np
import matplotlib.pyplot as plt

# Generate some data
X, y = make_regression(n_samples=1000, n_features=1, noise=30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the quantile regression model
n = 3
quantile = np.arange(n)/n + 0.5/n  # Quantiles to predict
model = QuantileRegressor(quantile=quantile,alpha=0)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Will plot the predicted quantiles for the test set
plt.scatter(X_train, y_train, s=0.5)
for i in range(y_pred.shape[1]):
    plt.plot(X_test, y_pred[:, i], label=f"Quantile: {quantile[i]:.2f}")
plt.legend()
plt.show()

# Print the coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
