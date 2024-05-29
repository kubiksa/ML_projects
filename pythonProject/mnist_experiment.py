from sklearn.ensemble import RandomForestClassifier
import autosklearn.classification
from sklearn import metrics
import numpy as np
from mnist_dataset_setup import (train_labels_600, train_images_600, test_labels_100, test_images_100,
                                 train_images_6000, train_labels_6000, test_images_1000, test_labels_1000,
                                 train_images_12000, train_labels_12000, test_images_2000, test_labels_2000)
#import mlflow
#import mlflow.sklearn
#mlflow.sklearn.autolog()
import neptune
import matplotlib.pyplot as plt

# Create AutoSklearn Classifier
cls = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=120)

# Initiate Neptune Run


# Define datasets
datasets = [
    ("6000 Training / 1000 Test", train_images_6000, train_labels_6000, test_images_1000, test_labels_1000),
    ("600 Training / 100 Test", train_images_600, train_labels_600, test_images_100, test_labels_100),
    ("1200_Training / T200_Test", train_images_12000, train_labels_12000, test_images_2000, test_labels_2000)
]

#need to include distribution details

# Train and evaluate models on each dataset
for dataset_name, train_images, train_labels, test_images, test_labels in datasets:
    print(f"Training and evaluating on dataset: {dataset_name}")

    # Convert images to NumPy arrays
    train_images_np = np.array(train_images)
    test_images_np = np.array(test_images)
    train_labels_np = np.array(train_labels)
    test_labels_np = np.array(test_labels)

    # Train the model
    cls.fit(train_images_np.reshape(-1, 28 * 28), train_labels_np)

    # Make predictions
    predictions = cls.predict(test_images_np.reshape(-1, 28 * 28))

    # Evaluate the model
    accuracy = metrics.accuracy_score(test_labels_np, predictions)  #for balanced data only
    test_f1_score = metrics.f1_score(test_labels_np, predictions, average='weighted')
    roc_auc = metrics.roc_auc_score(test_labels_np, predictions) #for balanced data only
    average_precision_score = metrics.average_precision_score(test_labels_np, predictions)
    test_metrics = (accuracy, test_f1_score)

    run = neptune.init_run(
        project="stephaniek-doit/autoML-first-exp",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4ZGU0ODU0Yy1hNTIyLTRhYmYtODk5MC0wODUxYjEwZTM3YTcifQ==",
    )  # your credentials

    params = dataset_name
    run["parameters"] = params

    run["minst/acc"].append(accuracy)
    run["minst/f1_score"].append(test_f1_score)
    run["minst/roc_auc"].append(roc_auc)
    run["minst/avg_precision_score"].append(average_precision_score)
    run.stop()


    #mlflow.log_metric('test_accuracy', accuracy)
    #mlflow.log_metric('test_f1_score', test_f1_score)

    print(f"Accuracy: {accuracy:.4f}")
    print(cls.show_models())


