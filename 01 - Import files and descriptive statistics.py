# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 12:44:46 2016

@author: s621208
"""

#import
import pandas as pandas
import numpy as numpy
#read file
titanic = pandas.read_csv("C:/Users/s621208/Downloads/data/train.csv")
titanic_test = pandas.read_csv("C:/Users/s621208/Downloads/data/test.csv")

titanic.tail()
titanic.head()
type(titanic)
titanic.info()
#Missing value on Cabin (too many), Embarked, Age
#Object type : Name, Sex, Ticket, Cabin, Embarked
#Modification

titanic['Sex']=pandas.Categorical(titanic['Sex'],ordered=False)
titanic['Name']=pandas.Categorical(titanic['Name'],ordered=False)
titanic['Ticket']=pandas.Categorical(titanic['Ticket'],ordered=False)
titanic['Cabin']=pandas.Categorical(titanic['Cabin'],ordered=False)
titanic['Embarked']=pandas.Categorical(titanic['Embarked'],ordered=False)

#Test
titanic.dtypes
#Description of variables
print(titanic.describe())
titanic["Age"].hist()
titanic["Age"].plot(kind="box")
#also titanic.boxplot("Age")
titanic["Fare"].hist()

#Qualitative variables
titanic["Sex"].value_counts() #577 males, 314 females
titanic["Cabin"].value_counts()
titanic["Embarked"].value_counts() #Port S dominates

titanic.plot(kind="scatter",x="Age",y="Fare")

#elder people (older than 60)
titanic[titanic["Age"]>60][["Sex","Pclass","Age","Survived"]]
#most of more than 60 year old people are male and non-survivor

titanic.boxplot(column="Age",by="Pclass")
#older people tend to buy higher class tickets, which makes sense

titanic.boxplot(column="Age",by="Survived")
#not much information to conclude : average age of survivor and non survivor are quite the same
# but the range of age for non survivor is larger

titanic.boxplot(column="Fare",by="Survived")
#Survivors tend to possess ticket with higher price ==> richer people tend to survive more

titanic.boxplot(column="SibSp",by="Survived")
#Not much information

titanic.boxplot(column="Parch",by="Survived")
#Not much information

table=pandas.crosstab(titanic["Survived"],titanic["Pclass"])
print(table)
from statsmodels.graphics.mosaicplot import mosaic
mosaic(titanic,["Pclass","Survived"])
#Most of victimes are from class 3

table2=pandas.crosstab(titanic["Survived"],titanic["Sex"])
print(table2)
mosaic(titanic,["Survived","Sex"])
#female are more likely survived than male passengers

table3=pandas.crosstab(titanic["Survived"],titanic["Embarked"])
print(table3)
mosaic(titanic,["Survived","Embarked"])
#most of people part from port S


#Fill missing value of variable Age by its median
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

#Fill missing value of variable Embarked by the most popular port of Embarkation
titanic["Embarked"]=titanic["Embarked"].fillna("S")
titanic.info()

#Converting the Sex column
titanic.loc[titanic["Sex"] == "male", "Sex1"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex1"] = 1

#Converting the Embarkation column
titanic.loc[titanic["Embarked"] == "S", "Port"] = 0
titanic.loc[titanic["Embarked"] == "C", "Port"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Port"] = 2

