# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:12:00 2016

@author: s621208
"""

titanic_test = pandas.read_csv("C:/Users/s621208/Downloads/data/test.csv")
titanic_test.info()

#Converting object type to categorical
#titanic_test['Sex']=pandas.Categorical(titanic_test['Sex'],ordered=False)
#titanic_test['Name']=pandas.Categorical(titanic_test['Name'],ordered=False)
#titanic_test['Ticket']=pandas.Categorical(titanic_test['Ticket'],ordered=False)
#titanic_test['Cabin']=pandas.Categorical(titanic_test['Cabin'],ordered=False)
#titanic_test['Embarked']=pandas.Categorical(titanic_test['Embarked'],ordered=False)

#missing value on Age, Fare, and Cabin (but it doesn't matter for this variable)
titanic_test["Age"] = titanic_test["Age"].fillna(titanic_test["Age"].median())
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

#Converting the Sex column 
titanic_test.loc[titanic_test["Sex"] == "male", "Sex1"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex1"] = 1

#Converting the Embarked column 
# C = Cherbourg = 1; Q = Queenstown = 2; S = Southampton = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Port"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Port"] = 2
titanic_test.loc[titanic_test["Embarked"] == "S", "Port"] = 0

titanic_test.info()

# Generating a familysize column
titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]

# Creating the new variable for the length of people's name
titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))

def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# Get all the titles and print how often each one occurs.
titles = titanic_test["Name"].apply(get_title)
print(pandas.value_counts(titles))

#Assigning number to title

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6,
                 "Col": 7, "Dona": 8, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles == k] = v
print(pandas.value_counts(titles))
titanic_test["Title"] = titles
titanic_test.info()

# We're using the more linear predictors for the logistic regression, and everything with the random forest classifier.
predictors = ["Pclass", "Sex1", "Age", "Fare", "Port", "FamilySize", "Title", "NameLength"]
algorithms = [
    [RandomForestClassifier(random_state=1, n_estimators=200, min_samples_split=4, min_samples_leaf=2), ["Pclass", "Sex1", "Age", "Fare", "Port", "FamilySize", "Title", "NameLength"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex1", "Fare", "NameLength", "Title", "Age", "Port","FamilySize"]]
]


full_predictions = []
for alg, predictors in algorithms:
    # Fit the algorithm using the full training data.
    alg.fit(titanic[predictors], titanic["Survived"])
    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
    predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:,1]
    full_predictions.append(predictions)


predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1
predictions = predictions.astype(int)
submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
submission.head()

submission.to_csv("C:/Users/s621208/Projets/output/kaggle_ndm.csv", index=False)
