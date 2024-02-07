import numpy as np
import pandas as pd
import constant
import re
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.tree as tree

trainData = pd.read_csv(constant.INPUTS + "train.csv")
testData = pd.read_csv(constant.INPUTS + "test.csv")
# print(trainData.head())
# print(trainData.describe())

print(testData[["Pclass", "Age", "SibSp", "Parch", "Fare"]].notna().describe())
print(testData[["Pclass", "Age", "SibSp", "Parch", "Fare"]].describe())


test = testData[["Pclass", "Age", "SibSp", "Parch", "Fare"]].fillna(30)
X = trainData[["Pclass", "Age", "SibSp", "Parch", "Fare"]].fillna(30)
y = trainData["Survived"]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
results = clf.predict(test)
d = {
    "PassengerId": testData["PassengerId"],
    "Survived": results,
}
print(d)
df = pd.DataFrame(d)
print(df)
print(df.describe())
df.to_csv('out.csv', index=False)

