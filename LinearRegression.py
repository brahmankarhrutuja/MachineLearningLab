# Define two lists of data points for the variables 'x' and 'y'.
x = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
y = [10, 12, 15, 18, 20, 22, 24, 28, 30, 33]

# Calculate the slope, intercept, correlation coefficient (r), p-value, and standard error of the slope
# using linear regression from the scipy library.
slope, intercept, r, p, std_err = stats.linregress(x, y)

# Define a function 'myfunc(x)' that takes an input 'x' and returns the predicted value 'y' based on the
# calculated slope and intercept from the linear regression.
def myfunc(x):
    return slope * x + intercept

# Use the 'map()' function to apply the 'myfunc' function to each element of the 'x' list,
# which will create a list of predicted 'y' values based on the linear regression model.
mymodel = list(map(myfunc, x))

# Create a scatter plot of the original data points (x, y) using 'plt.scatter()'.
plt.scatter(x, y)

# Create a line plot of the predicted 'y' values (mymodel) against the 'x' values using 'plt.plot()'.
plt.plot(x, mymodel)

# Set labels for the plot
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Linear Regression')

# Display the plot using 'plt.show()'.
plt.show()
