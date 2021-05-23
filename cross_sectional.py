import sklearn
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
import csv

oldData = pd.read_csv("oasis_longitudinal.csv")
data = oldData.dropna()

le = preprocessing.LabelEncoder()
cls = le.fit_transform(data["Group"])

predict = "Group"

X = list(zip(data["eTIV"], data["nWBV"], data["ASF"], data["CDR"], data["CDR"]))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)

insertData = x_test #[(4,4,4,4,1), (1,1,5,5,1), (3,3,3,3,1)]
predicted = model.predict(insertData)
result = [ "Converted", "Demented", "Nondemented"]

if insertData == x_test:
    print("Accuracy: {}%\n".format(round(accuracy * 100, 2)))
for x in range(len(predicted)):
    if insertData == x_test:
        print("Predicted:", str(result[predicted[x]]) + " |", "Data:", str(x_test[x]) + " |", "Actual:", str(result[y_test[x]]))
    else:
        print("Predicted:", result[predicted[x]], "\nData:", insertData[x])
