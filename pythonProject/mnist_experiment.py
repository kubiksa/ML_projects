from sklearn.ensemble import RandomForestClassifier
import autosklearn.classification
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from mnist_dataset_setup import (#train_labels_600, train_images_600, test_labels_100, test_images_100,
                                 #train_images_6000, train_labels_6000, test_images_1000, test_labels_1000,
                                 train_images_12000, train_labels_12000, test_images_2000, test_labels_2000)
#import mlflow
#import mlflow.sklearn
#mlflow.sklearn.autolog()
from sklearn.preprocessing import label_binarize
import neptune
import matplotlib.pyplot as plt

# Create AutoSklearn Classifier
cls = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=120)

# Initiate Neptune Run


# Define datasets
datasets = [
    #("6000 Training / 1000 Test", train_images_6000, train_labels_6000, test_images_1000, test_labels_1000),
    #("600 Training / 100 Test", train_images_600, train_labels_600, test_images_100, test_labels_100),
    ("1200_Training / T2000_Test", train_images_12000, train_labels_12000, test_images_2000, test_labels_2000)
]

#need to include distribution details

# Train and evaluate models on each dataset
for dataset_name, train_images, train_labels, test_images, test_labels in datasets:
    print(f"Training and evaluating on dataset: {dataset_name}")

    # Convert images to NumPy arrays
    train_images_np = np.array(train_images) #x_train
    test_images_np = np.array(test_images)   #x_test
    train_labels_np = np.array(train_labels)  #y_train
    test_labels_np = np.array(test_labels)    #y_test



    train_images_np = train_images_np.reshape((train_images_np.shape[0], 28 * 28)).astype(np.float32)
    test_images_np = test_images_np.reshape((test_images_np.shape[0], 28 * 28)).astype(np.float32)

    # Combine the train and test datasets
    X = np.concatenate((train_images_np, test_images_np), axis=0)
    y = np.concatenate((train_labels_np, test_labels_np), axis=0)

    # Split the combined dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check combined dataset shapes
    assert X.shape[0] == y.shape[0], "Mismatch in number of combined samples"

    # Check split shapes
    assert X_train.shape[0] == y_train.shape[0], "Mismatch in number of training samples after split"
    assert X_test.shape[0] == y_test.shape[0], "Mismatch in number of testing samples after split"

    # Binarize the output for multi-class ROC AUC and average precision
    y_train_bin = label_binarize(y_train, classes=range(10))
    y_test_bin = label_binarize(y_test, classes=range(10))


    # Train the model
    cls.fit(X_train, y_train)

    # Make predictions
    predictions = cls.predict(X_test)
    y_pred_prob = cls.predict_proba(X_test)

    # Evaluate the model
    accuracy = metrics.accuracy_score(y_test, predictions)
    test_f1_score = metrics.f1_score(y_test, predictions, average='weighted')
    roc_auc = metrics.roc_auc_score(y_test_bin, y_pred_prob, average='macro', multi_class='ovr')
    avg_precision = metrics.average_precision_score(y_test_bin, y_pred_prob, average='macro')

    #accuracy = metrics.accuracy_score(test_labels_np, predictions)  #for balanced data only
    #test_f1_score = metrics.f1_score(test_labels_np, predictions, average='weighted')
    #roc_auc = metrics.roc_auc_score(y_test_bin, y_pred_prob, average='macro', multi_class='ovr')
    #avg_precision = metrics.average_precision_score(y_test_bin, y_pred_prob, average='macro')
    #roc_auc = metrics.roc_auc_score(test_labels_np, predictions) #for balanced data only
    #average_precision_score = metrics.average_precision_score(test_labels_np, predictions)
    test_metrics = (accuracy, test_f1_score)

    #mlflow.log_metric('test_accuracy', accuracy)
    #mlflow.log_metric('test_f1_score', test_f1_score)

    best_model = cls.show_models()
    ensemble_model = cls.get_models_with_weights()
    model_pipeline = cls.get_models_with_weights()[0][1]

    pipeline_steps = model_pipeline.named_steps
    for step_name, step in pipeline_steps.items():
        if hasattr(step, 'transform'):
            print(f"Step: {step_name}")
            print(step)
            # Example: Check the shape of transformed features
            transformed_X = step.transform(X_train)
            print(f"Number of features after {step_name}: {transformed_X.shape[1]}")

# Example with a tree-based model to get feature importances
    if 'classifier' in pipeline_steps and hasattr(pipeline_steps['classifier'], 'feature_importances_'):
        feature_importances = pipeline_steps['classifier'].feature_importances_
        number_of_features = len(feature_importances)

        run = neptune.init_run(
            project="stephaniek-doit/autoML-first-exp",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4ZGU0ODU0Yy1hNTIyLTRhYmYtODk5MC0wODUxYjEwZTM3YTcifQ==",
        )  # your credentials

        params = dataset_name

        run["parameters"] = params
        run['best_model'].append(best_model)
        run["no_of_features"].append(number_of_features)
        run["minst/acc"].append(accuracy)
        run["minst/f1_score"].append(test_f1_score)
        run["minst/roc_auc"].append(roc_auc)
        run["minst/avg_precision_score"].append(avg_precision)
        run.stop()


