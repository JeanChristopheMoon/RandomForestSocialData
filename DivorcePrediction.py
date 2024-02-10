import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib as mt
import matplotlib.pyplot as plt


pdf = pd.read_csv('x', header = 0)
print(pdf)

num_columns = pdf.shape[1]
print(num_columns)
feature_columns = pdf.columns[0:54]
print(feature_columns)
target_column = pdf.columns[-1]
print(target_column)

# Separate features (X) and target variable (y)
X = pdf[feature_columns]
y = pdf[target_column]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 18)
clf = RandomForestClassifier(n_estimators = 100, max_depth = 4, max_features = 3, bootstrap = True, random_state = 18).fit(x_train, y_train)

# Create our predictions
prediction = clf.predict(x_test)

confusion_matrix(y_test, prediction)

# Display accuracy score
accuracy_score(y_test, prediction)

# Display F1 score
#  we would prefer to determine a classification’s performance by its precision, recall, or F1 score.
f1_score(y_test,prediction)

#Importance Visualization 
model = RandomForestClassifier()
model.fit(X, y)

feature_importances = model.feature_importances_

plt.bar(range(len(feature_importances)), feature_importances, tick_label=X.columns)
plt.title('Feature Importance Plot')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha="right")
plt.show()
