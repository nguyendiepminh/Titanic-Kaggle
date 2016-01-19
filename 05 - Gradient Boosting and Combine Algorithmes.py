# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 14:28:35 2016

@author: s621208
"""
from sklearn.ensemble import GradientBoostingClassifier

predictors = ["Pclass", "Sex1", "Age", "Fare", "Port", "FamilySize", "Title", "NameLength"]
alg = GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
print(scores.mean())

#average score: 82,71%

# The algorithms we want to ensemble.
# We're using the more linear predictors for the logistic regression, and everything with the gradient boosting classifier.
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex1", "Age", "Fare", "Port", "FamilySize", "Title", "NameLength"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex1", "Fare", "NameLength", "Title", "Age", "Port","FamilySize"]]
]


# Initialize the cross validation folds
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)
predictions = []
for train, test in kf:
    train_target = titanic["Survived"].iloc[train]
    full_test_predictions = []
    # Make predictions for each algorithm on each fold
    for alg, predictors in algorithms:
        # Fit the algorithm on the training data.
        alg.fit(titanic[predictors].iloc[train,:], train_target)
        # Select and predict on the test fold.  
        # The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    # Use a simple ensembling scheme -- just average the predictions to get the final classification.
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)
    
 # Put all the predictions together into one array.
predictions = numpy.concatenate(predictions, axis=0)

# Compute accuracy by comparing to the training data.
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print(accuracy)   

#accuracy : 81,93%

# The algorithms we want to ensemble.
# We're using the more linear predictors for the logistic regression, and everything with the random forest classifier.
algorithms = [
    [RandomForestClassifier(random_state=1, n_estimators=200, min_samples_split=4, min_samples_leaf=2), ["Pclass", "Sex1", "Age", "Fare", "Port", "FamilySize", "Title", "NameLength"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex1", "Fare", "NameLength", "Title", "Age", "Port","FamilySize"]]
]

# Initialize the cross validation folds
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)
predictions = []
for train, test in kf:
    train_target = titanic["Survived"].iloc[train]
    full_test_predictions = []
    # Make predictions for each algorithm on each fold
    for alg, predictors in algorithms:
        # Fit the algorithm on the training data.
        alg.fit(titanic[predictors].iloc[train,:], train_target)
        # Select and predict on the test fold.  
        # The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    # Use a simple ensembling scheme -- just average the predictions to get the final classification.
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)
    
 # Put all the predictions together into one array.
predictions = numpy.concatenate(predictions, axis=0)

# Compute accuracy by comparing to the training data.
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print(accuracy)   

#accuracy : 82,82%
