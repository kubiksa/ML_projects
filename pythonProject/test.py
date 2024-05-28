import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
#this was a generic script I pulled off of ChatGPT 3.5
#I just wanted to check to make sure the auto-sklearn package worked properly
#load a generic dataset in the sklearn package
X, y = sklearn.datasets.load_digits(return_X_y=True)

#split the dataset into training and testing sets
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)

#create and fit the AutoML model
automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=120, per_run_time_limit=30)
automl.fit(X_train, y_train)

#evaluate the model
y_pred = automl.predict(X_test)
accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#inspect the best model
print(automl.show_models())