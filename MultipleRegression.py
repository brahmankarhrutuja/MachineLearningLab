import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data for fruits and vegetables sold
fruits_sold = np.array([5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6])
temperature = np.array([25, 28, 30, 27, 20, 35, 22, 29, 26, 32, 31, 28, 27])
vegetables_sold = np.array([10, 12, 15, 18, 20, 22, 24, 28, 30, 33, 28, 21, 19])

# Combine features into a single 2D array (X)
X = np.column_stack((fruits_sold, temperature))

# Create a linear regression model
regr = LinearRegression()
regr.fit(X, vegetables_sold)

# Generate predicted values using the model
vegetables_pred = regr.predict(X)

# Create a scatter plot to visualize the data points
plt.scatter(fruits_sold, vegetables_sold, label='Data Points')
plt.scatter(fruits_sold, vegetables_pred, label='Predicted Values')

# Plot the regression line
plt.plot(fruits_sold, vegetables_pred, linewidth=1, label='Regression Line')

# Set labels and title
plt.xlabel('Fruits Sold')
plt.ylabel('Vegetables Sold')
plt.title('Multiple Linear Regression for Fruits and Vegetables Shop')

# Show the legend
plt.legend()

# Display the plot
plt.show()
