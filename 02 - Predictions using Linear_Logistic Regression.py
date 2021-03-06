# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 13:12:54 2016

@author: s621208
"""

"""
Linear Regression
"""
# Import the linear regression class
from sklearn.linear_model import LinearRegression
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold


# The columns we'll use to predict the target
#Only use numerical variable
predictors = ["Pclass", "Sex1", "Age", "SibSp", "Parch", "Fare", "Port"]

# Initialize our algorithm class
alg = LinearRegression()
# Generate cross validation folds for the titanic dataset.  
#It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)
predictions = []
for train, test in kf:
# The predictors we're using the train the algorithm.  
#Note how we only take the rows in the train folds.
    train_predictors = (titanic[predictors].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = titanic["Survived"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)
    
#The predictions are in three separate numpy arrays.  Concatenate them into one
#We concatenate them on axis 0, as they only have one axis.
import numpy as numpy
predictions = numpy.concatenate(predictions, axis=0)

# Map predictions to outcomes (only possible outcomes are 1 and 0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0

#Proportion of the values in predictions are the exact same as the values 
#in titanic["Survived"]

accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print(accuracy)

#accuracy : 78,34%

"""
Logistic Regression
"""
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

# Initialize our algorithm
alg2 = LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
accuracy2 = cross_validation.cross_val_score(alg2, titanic[predictors], titanic["Survived"], cv=3)
# Take the mean of the scores (because we have one for each fold)
print(accuracy2.mean())

#new accuracy: 78,79%