import numpy as np
import matplotlib.pyplot as plt

# Number of fruits sold on different days
fruits_sold = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]

# Number of vegetables sold on different days
vegetables_sold = [10, 12, 15, 18, 20, 22, 24, 28, 30, 33, 28, 21, 19]

# Calculate the linear regression for the data
slope, intercept = np.polyfit(fruits_sold, vegetables_sold, 1)

# Create a function to predict vegetables sold based on fruits sold
def predict_vegetables(fruits):
    fruits = np.array(fruits)  # Convert to NumPy array
    return slope * fruits + intercept

# Create a scatter plot of the data points
plt.scatter(fruits_sold, vegetables_sold, label='Data Points')

# Plot the regression line
plt.plot(fruits_sold, predict_vegetables(fruits_sold), label='Regression Line')

# Set labels and title
plt.xlabel('Number of Fruits Sold')
plt.ylabel('Number of Vegetables Sold')
plt.title('Fruits and Vegetables Shop')

# Show the legend
plt.legend()

# Display the plot
plt.show()
