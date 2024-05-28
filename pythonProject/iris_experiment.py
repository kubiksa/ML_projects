import autosklearn.classification
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
import numpy as np
from iris_dataset_setup import (X_train, X_test, y_train, y_test)


cls = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=120)
dataset = (X_train, X_test, y_train, y_test)