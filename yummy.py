import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('oasis_longitudinal.csv')

# df['Group'] = (df['Group'] == 'Demented').astype(int)

# cols = ['MR Delay', 'M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']
# #
# for c in cols:
#     plt.hist(df[df['Group'] == 'Demented'][c], color='blue', label='Demented', alpha=0.7, density=True, bins=20)
#     plt.hist(df[df['Group'] == 'Non-Demented'][c], color='red', label='Non-Demented', alpha=0.7, density=True, bins=20)
#     plt.xlabel(c)
#     plt.ylabel('Frequency')
#     plt.title("Histogram of {}".format(c))
#     plt.legend()
#     plt.grid()
#     plt.show()

df = pd.get_dummies(df, columns=['M/F','Hand','Subject ID',"MRI ID"])
Input = df.drop(columns=['Group'])
result = df['Group']
input_train, input_test, output_train, output_test = train_test_split(Input, result, test_size=0.2,random_state=1)

model = DecisionTreeClassifier()
model.fit(input_train, output_train)
predictions = model.predict(input_test)
score = accuracy_score(output_test, predictions)
print("{} accuracy: {}".format("DecisionTreeClassifer", score * 100))

# model = LogisticRegression()
# model.fit(input_train, output_train)
# predictions = model.predict(input_test)
# score = accuracy_score(output_test, predictions)
# print("{} accuracy: {}".format("LogisticRegression", score * 100))

# model  = KNeighborsClassifier()
# model.fit(input_train, output_train)
# predictions = model.predict(input_test)
# score = accuracy_score(output_test, predictions)
# print("{} accuracy: {}".format("KNeighborsClassifier", score * 100))

# model = LinearDiscriminantAnalysis()
# model.fit(input_train, output_train)
# predictions = model.predict(input_test)
# score = accuracy_score(output_test, predictions)
# print("{} accuracy: {}".format("LinearDiscriminantAnalysis", score * 100))

# model = GaussianNB()
# model.fit(input_train, output_train)
# predictions = model.predict(input_test)
# score = accuracy_score(output_test, predictions)
# print("{} accuracy: {}".format("GaussianNB", score * 100))

# model = SVC(gamma='auto')
# model.fit(input_train, output_train)
# predictions = model.predict(input_test)
# score = accuracy_score(output_test, predictions)
# print("{} accuracy: {}".format("SVC", score * 100))

model = RandomForestClassifier()
model.fit(input_train, output_train)
predictions = model.predict(input_test)
score = accuracy_score(output_test, predictions)
print("{} accuracy: {}".format("RandomForestClassifier", score * 100))
