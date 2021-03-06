# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 13:18:44 2016

@author: s621208
"""

"""
To improve score :
    - Use a better machine learning algorithm.
    - Generate better features.
    - Combine multiple machine learning algorithms.
    
"""
"""
# Implimenting Random Forest

"""
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

predictors = ["Pclass", "Sex1", "Age", "SibSp", "Parch", "Fare", "Port"]

# Initialize our algorithm with the default paramters
# n_estimators is the number of trees we want to make
# min_samples_split is the minimum number of rows we need to make a split
# min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends (the bottom points of the tree)
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
print(scores.mean())

#average scores : 80,13%
#Increasing number of trees to improve the score

alg2 = RandomForestClassifier(random_state=1, n_estimators=200, min_samples_split=4, min_samples_leaf=2)
scores2 = cross_validation.cross_val_score(alg2, titanic[predictors], titanic["Survived"], cv=3)
print(scores2.mean())

#average scores : 82,38%
"""
Creating the new features

"""
"""
#The length of the name, this could pertain to how rich the person was, 
    and therefore their position in the Titanic.
#The title attached to people name could also be a good indicator of their socio-economical status
#The total number of people in a family (SibSp + Parch): 
    the more people inside one family travel together, the (probably) richer they are
""""
# Generating a familysize column
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]

# Creating the new variable for the length of people's name
titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))

# Attracting the titles 

import re

# A function to get the title from a name.
def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# Get all the titles and print how often each one occurs.
titles = titanic["Name"].apply(get_title)
print(pandas.value_counts(titles))

#Assigning number to title

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6,
                 "Major": 7, "Col": 7, "Mlle": 2, "Mme": 3, "Don": 8, "Lady": 9, 
                 "Countess": 9, "Jonkheer": 9, "Sir": 8, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles == k] = v
print(pandas.value_counts(titles))
titanic["Title"] = titles
titanic.info()

"""
Select the best new features
"""
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

predictors = ["Pclass", "Sex1", "Age", "SibSp", "Parch", "Fare", "Port", "FamilySize", "Title", "NameLength"]

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic["Survived"])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -numpy.log10(selector.pvalues_)
print(scores.mean())

# Plot the scores.  See how "Pclass", "Sex", "Title", "Namelength" and "Fare" are the best?
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()

#Random Forest with all the variables
predictors = ["Pclass", "Sex1", "Age", "SibSp", "Parch", "Fare", "Port", "FamilySize", "Title", "NameLength"]
alg = RandomForestClassifier(random_state=1, n_estimators=200, min_samples_split=4, min_samples_leaf=2)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
print(scores.mean())
#average score: 83,28%

#Only with the best features
predictors = ["Pclass", "Sex1", "Fare", "Title", "NameLength", "Port"]
alg = RandomForestClassifier(random_state=1, n_estimators=200, min_samples_split=4, min_samples_leaf=2)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
print(scores.mean())

#average score : 83,28%

#There is not differences between the 2 predictors, use the shorter version