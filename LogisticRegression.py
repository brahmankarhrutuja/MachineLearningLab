import numpy as np
from sklearn.linear_model import LogisticRegression

# Hours studied by students
X = np.array([1, 2, 3, 4, 5, 7, 6, 8, 9, 10]).reshape(-1, 1)

# Pass (1) or Fail (0) labels
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

logr = LogisticRegression()
logr.fit(X, y)

# Predict if a student who studied for 5 hours will pass the exam
predicted = logr.predict(np.array(5).reshape(-1, 1))
print(predicted)
