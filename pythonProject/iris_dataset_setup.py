import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# Load the Iris dataset
iris = load_iris()
#print(iris) #each class is grouped together so would need to randomize to use cross validation
X, y = iris.data, iris.target

# Number of samples to generate for each class
num_samples_1500 = [500, 500, 500]  # Total: 1500
num_samples_15000 = [5000, 5000, 5000]  # Total: 15000

expanded_X = []
expanded_y = []

for class_label, num_sample in enumerate(num_samples_15000):
    # Select data points of the current class
    class_indices = np.where(y == class_label)[0]
    class_data = X[class_indices]

    # Replicate existing data points
    replicated_data = np.repeat(class_data, num_sample // len(class_data), axis=0)
    remainder = num_sample % len(class_data)
    if remainder > 0:
        extra_data_indices = np.random.choice(len(class_data), remainder, replace=False)
        extra_data = class_data[extra_data_indices]
        replicated_data = np.vstack([replicated_data, extra_data])

    expanded_X.extend(replicated_data)
    expanded_y.extend([class_label] * num_sample)

expanded_X = np.array(expanded_X)
expanded_y = np.array(expanded_y)

# Split the expanded dataset into training and test sets
#X_train_1200, X_test_1200, y_train_300, y_test_300 = train_test_split(expanded_X, expanded_y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(expanded_X, expanded_y, test_size=0.2, random_state=42)


# Print the shapes of the training and test sets
print("Training set shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)

# Now, expanded_X and expanded_y contain the expanded dataset with the same distributions.
