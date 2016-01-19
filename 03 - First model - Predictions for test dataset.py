"""
@author: Diep Minh NGUYEN
"""
"""
First model
"""

titanic_test = pandas.read_csv("E:/AXA/AXA Entreprise IARD/Python/data/test.csv")
titanic_test.info()

#Converting object type to categorical
titanic_test['Sex']=pandas.Categorical(titanic_test['Sex'],ordered=False)
titanic_test['Name']=pandas.Categorical(titanic_test['Name'],ordered=False)
titanic_test['Ticket']=pandas.Categorical(titanic_test['Ticket'],ordered=False)
titanic_test['Cabin']=pandas.Categorical(titanic_test['Cabin'],ordered=False)
titanic_test['Embarked']=pandas.Categorical(titanic_test['Embarked'],ordered=False)

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

""""
Prediction on the test set
""""

# Initialize the algorithm class
alg = LogisticRegression(random_state=1)

# Train the algorithm using all the training data
alg.fit(titanic[predictors], titanic["Survived"])

# Make predictions using the test set.
predictions = alg.predict(titanic_test[predictors])

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
submission.head()
