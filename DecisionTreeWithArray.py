import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Synthetic data for customer age, purchase amount, and purchase type (fruits or vegetables)
age = np.array([25, 30, 35, 20, 40, 28, 45, 50, 22, 38])
purchase_amount = np.array([50, 60, 75, 40, 80, 55, 90, 95, 45, 70])
purchase_type = np.array(['Fruits', 'Fruits', 'Vegetables', 'Fruits', 'Vegetables', 'Fruits', 'Vegetables', 'Fruits', 'Vegetables', 'Fruits'])

# Combine features into a single 2D array (X)
X = np.column_stack((age, purchase_amount))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, purchase_type, test_size=0.2, random_state=42)

# Create a Decision Tree classifier with Gini index as the criterion
clf = DecisionTreeClassifier(criterion='gini')

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Predict purchase type for the test data
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print results
print("Predicted:", y_pred)
print("Actual:", y_test)
print("Accuracy:", accuracy)
print("\n")

# Plot the Decision Tree
plt.figure(figsize=(10, 3))
plot_tree(clf, feature_names=["Age", "Purchase Amount"], class_names=clf.classes_, filled=True, rounded=True)
plt.title("Decision Tree for Fruits and Vegetables Shop")
plt.show()
