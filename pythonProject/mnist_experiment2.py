from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import autosklearn.classification
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas as pd
from mnist_dataset_setup import (train_images_24000, train_labels_24000, test_images_4000, test_labels_4000)
#train_labels_6000, train_images_6000, test_labels_1000, test_images_1000)
#train_labels_600, train_images_600, test_labels_100, test_images_100)
#(train_images_24000, train_labels_24000, test_images_4000, test_labels_4000))
                            #(train_images_full, train_labels_full, test_images_full, test_labels_full)

         #               train_labels_6000, train_images_6000, test_labels_1000, test_images_1000
         #train_images_12000, train_labels_12000, test_images_2000, test_labels_2000


from sklearn.preprocessing import label_binarize
import neptune
import matplotlib.pyplot as plt

# Create AutoSklearn Classifier
cls = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=1500)

# Define datasets
datasets = [
    #("6000Training_1000Test", train_images_6000, train_labels_6000, test_images_1000, test_labels_1000)
    #("600Training_100Test", train_images_600, train_labels_600, test_images_100, test_labels_100)
    #("12000Training_2000Test", train_images_12000, train_labels_12000, test_images_2000, test_labels_2000)
    ("24000Training_4000Test", train_images_24000, train_labels_24000, test_images_4000, test_labels_4000)
    #("full_dataset", train_images_full, train_labels_full, test_images_full, test_labels_full)
]

# Train and evaluate models on each dataset
for dataset_name, train_images, train_labels, test_images, test_labels in datasets:
    print(f"Training and evaluating on dataset: {dataset_name}")

# Convert images to NumPy arrays
    train_images_np = np.array(train_images)
    test_images_np = np.array(test_images)
    train_labels_np = np.array(train_labels)
    test_labels_np = np.array(test_labels)

    # Flatten (Reshape) the images into 784-dimensional vectors
    x_train_flattened = train_images_np.reshape((train_images_np.shape[0], 28 * 28)).astype(np.float32)
    x_test_flattened = test_images_np.reshape((test_images_np.shape[0], 28 * 28)).astype(np.float32)

    # Split the original training set into a new training set and a validation set
    #X_train, X_test, y_train, y_test = train_test_split(x_train_flattened, train_labels, test_size=0.2, random_state=42)

    #rename
    X_train = x_train_flattened
    X_test = x_test_flattened
    y_train = train_labels_np  #1-D
    y_test = test_labels_np   #1-D

    train_x_df = pd.DataFrame(X_train).to_csv(dataset_name+'_x_train', index=False)
    train_y_df = pd.DataFrame(y_train).to_csv(dataset_name+'_y_train', index=False)
    test_x_df = pd.DataFrame(X_test).to_csv(dataset_name+'_x_test', index=False)
    test_y_df = pd.DataFrame(y_test).to_csv(dataset_name+'_y_test', index=False)

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

    print(f'Accuracy: {accuracy}')
    print(f'F1 Score: {test_f1_score}')
    print(f'ROC AUC: {roc_auc}')
    print(f'Average Precision: {avg_precision}')

    # Access the best model
    best_model = cls.show_models()
    print("Best model details:\n", best_model)

    # Access the final ensemble model's components
    ensemble_models_with_weights = cls.get_models_with_weights()

    # Access the first model in the ensemble
    model_pipeline = ensemble_models_with_weights[0][1]

    # Inspect the pipeline steps
    #pipeline_steps = model_pipeline.named_steps
    #for step_name, step in pipeline_steps.items():
    #     if hasattr(step, 'transform'):
    #         # Calculate the number of features in X
    #         num_features = X_train.shape[1]
    #         # Define PCA with the number of components equal to the number of features in X
    #         pca = PCA(n_components=num_features)
    #         # Apply PCA
    #         X_train_pca = pca.fit_transform(X_train)
    #         print(f"Step: {step_name}")
    #         print(step)
    #         # Example: Check the shape of transformed features
    #         transformed_X = step.transform(X_train)
    #         print(f"Number of features after {step_name}: {transformed_X.shape[1]}")
    #
    # # Example with a tree-based model to get feature importances
    # if 'classifier' in pipeline_steps and hasattr(pipeline_steps['classifier'], 'feature_importances_'):
    #     feature_importances = pipeline_steps['classifier'].feature_importances_
    #     number_of_features = len(feature_importances)

    # Get the final fitted pipeline
    pipeline = cls.get_models_with_weights()[0][1]
    # Extract the preprocessing steps from the pipeline
    #preprocessing_steps = pipeline.named_steps.get('feature_preprocessor')

###This stopped working as it's telling me feature agglomeration is expected 686 features
    # if preprocessing_steps:
    #     # If feature preprocessing steps exist, get the transformed feature matrix
    #     X_train_transformed = preprocessing_steps.transform(X_train)
    #     num_features_final = X_train_transformed.shape[1]
    #     print("Final number of features after preprocessing:", num_features_final)
    # else:
    #     # If no feature preprocessing steps exist, use the original feature matrix
    #     num_features_final = X_train.shape[1]
    #     print("No preprocessing steps applied, using original number of features:", num_features_final)
    run = neptune.init_run(
        project="stephaniek-doit/autoML-first-exp",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4ZGU0ODU0Yy1hNTIyLTRhYmYtODk5MC0wODUxYjEwZTM3YTcifQ==",
    )  # your credentials

    params = dataset_name

    run["parameters"] = params
    run['best_model'].append(best_model)
    #run["no_of_features"].append(number_of_features)
    run["mnist/accuracy"].append(accuracy)
    run["mnist/f1_score"].append(test_f1_score)
    run["mnist/roc_auc"].append(roc_auc)
    run["mnist/avg_precision_score"].append(avg_precision)
    #run["mnist/features"].append(num_features_final)
    run.stop()
