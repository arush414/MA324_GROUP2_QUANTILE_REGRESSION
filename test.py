
# Generate some data
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the quantile regression model
quantile = [0.3,0.3]  # Quantile to predict
model = QuantileRegressor(quantile=quantile)
model.fit(X_train, y_train)

# Predict on test data

y_pred = model.predict(X_test)
print(y_pred)
print("HUEHUEHUE")
print(y_test)
# Print the coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)