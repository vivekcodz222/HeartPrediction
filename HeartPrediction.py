import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv("D:/CODING PAGE/MLbyAYUSH/HeartPrediction/dataset/archive/heart_failure_clinical_records_dataset.csv")
data.head()

print("Shape of the data", data.shape)

print("Information about data:- ")
data.info()

print("describing the data:- ")
data.describe()

print(f"No. of Null Values:-" )
data.isnull().sum().sum()

#Exploratory data analysis

# seeing the distribution of classes, this will help us to identify which types

len_live = len(data["DEATH_EVENT"][data.DEATH_EVENT == 0])
len_death = len(data["DEATH_EVENT"][data.DEATH_EVENT == 1])

arr = np.array([len_live , len_death])
labels = ['LIVING', 'DIED']
print("Total No. Of Living Cases :- ", len_live)
print("Total No. Of Died Cases :- ", len_death)

plt.pie(arr, labels=labels, explode = [0.2,0.0] , shadow=True)
plt.show()

# inference :- we are actually working on imbalance data
# imbalance:- your data is not equally distributed between classes

# seeing the distribution of Age

sns.histplot(data["age"])
plt.show()

# selecting rows that are above age 50 and seeing died or not
age_above_50_not_died = data['DEATH_EVENT'][data.age >= 50][data.DEATH_EVENT == 0]
age_above_50_died = data['DEATH_EVENT'][data.age >= 50][data.DEATH_EVENT == 1]

len_died = len(age_above_50_died)
len_not_died = len(age_above_50_not_died)

arr1 = [len_died, len_not_died]
labels = ['DIED', 'NOT DIED']

plt.pie(arr1, labels=labels, explode = [0.2,0.0] , shadow=True)
plt.show()

print ("Total no. of died cases, ", len_died)
print("Total no. of not died cases, ", len_not_died)
# inference in most of the cases people aged above 50 not died but accordinly if you compare with above
# plot you will be seeing that died ration is comparitively higher here

patient_nhave_diabetes_0 = data['DEATH_EVENT'][data.diabetes == 1][data.DEATH_EVENT == 0]
patient_have_diabetes_1 = data['DEATH_EVENT'][data.diabetes == 1][data.DEATH_EVENT == 1]

len_d_died = len(patient_have_diabetes_1)
len_d_alive = len(patient_nhave_diabetes_0)

arr2 = [len_d_died,len_d_alive]
labels = ['Died with Diabetes', "Not Died with Diabetes"]
plt.pie(arr2, labels=labels, explode = [0.2,0.0] , shadow=True)
plt.show()

# inference:- here you can see the that the most of the person are alive who have diabetes

# checking the Correlation of our variables

corr = data.corr()
plt.subplots(figsize=(15,15))
sns.heatmap(corr, annot=True)

# interpretation of correlation matrix

''' 
 - Each square shows the correlation between the variables on each axis. Correlation ranges from -1 to +1. Values closer to zero means there is no linear trend between the two variables. 
 - The close to 1 the correlation is the more positively correlated they are; that is as one increases so does the other and the closer to 1 the stronger this relationship is.  
 - A correlation closer to -1 is similar, but instead of both increasing one variable will decrease as the other increases.  
 - The diagonals are all 1/dark green because those squares are correlating each variable to itself (so it's a perfect correlation). For the rest the larger the number and darker the color the higher the correlation between the two variables.   
 - The plot is also symmetrical about the diagonal since the same two variables are being paired together in those squares.
'''

# references:- https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

# you can do the same as here
data.corr().style.background_gradient(cmap='coolwarm')

#Dataset development
from sklearn.model_selection import train_test_split

X = data.drop('DEATH_EVENT', axis=1)
y = data["DEATH_EVENT"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
print("Shape of the X_train", X_train.shape)
print("Shape of the y_train", y_train.shape)
print("Shape of the X_test", X_test.shape)
print("Shape of the y_test", y_test.shape)

#feature Engineering
from sklearn.model_selection import train_test_split


def add_interactions(X):
    features = X.columns
    m = len(features)
    X_int = X.copy(deep=True)

    for i in range(m):

        feature_i_name = features[i]

        feature_i_data = X[feature_i_name]

        for j in range(i + 1, m):
            feature_j_name = features[j]
            feature_j_data = X[feature_j_name]
            feature_i_j_name = feature_i_name + "_x_" + feature_j_name
            X_int[feature_i_j_name] = feature_i_data * feature_j_data

    return X_int


x_train_mod = add_interactions(X_train)
x_test_mod = add_interactions(X_test)

#Model bulding
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


def evaluating_model(y_test, y_pred):
    '''
    Function for evaluating our models.
    '''
    print("Accuracy Score:- ", accuracy_score(y_test, y_pred))
    print("Precision Score:- ", precision_score(y_test, y_pred))
    print("Recall Score:- ", recall_score(y_test, y_pred))
    print("Confusion Matrix:- \n", confusion_matrix(y_test, y_pred))
# building logistic regression model as a baseline model

from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(max_iter=1000)
lr_clf.fit(X_train, y_train)

lr_clf_pred = lr_clf.predict(X_test)
y_pred = lr_clf.predict(X_test)
evaluating_model(y_test, y_pred)
# building logistic regression with StandardScaler

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

lr_clf_pip = make_pipeline(StandardScaler(), LogisticRegression())
lr_clf_pip.fit(X_train, y_train)

y_pred1 = lr_clf_pip.predict(X_test)
evaluating_model(y_test,y_pred1)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose =3)
grid.fit(X_train, y_train)
grid.best_estimator_
svc = SVC(C = 10, gamma = 0.0001)
svc.fit(X_train, y_train)
y_pred2 = svc.predict(X_test)
evaluating_model(y_test, y_pred2)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV


def randomized_search(params, runs=20, clf=DecisionTreeClassifier(random_state=2)):
    rand_clf = RandomizedSearchCV(clf, params, n_iter=runs, cv=5, n_jobs=-1, random_state=2)
    rand_clf.fit(X_train, y_train)
    best_model = rand_clf.best_estimator_
    best_score = rand_clf.best_score_

    print("Training score: {:.3f}".format(best_score))
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Test score: {:.3f}'.format(accuracy))

    return best_model


randomized_search(params={'criterion': ['entropy', 'gini'],
                          'splitter': ['random', 'best'],
                          'min_weight_fraction_leaf': [0.0, 0.0025, 0.005, 0.0075, 0.01],
                          'min_samples_split': [2, 3, 4, 5, 6, 8, 10],
                          'min_samples_leaf': [1, 0.01, 0.02, 0.03, 0.04],
                          'min_impurity_decrease': [0.0, 0.0005, 0.005, 0.05, 0.10, 0.15, 0.2],
                          'max_leaf_nodes': [10, 15, 20, 25, 30, 35, 40, 45, 50, None],
                          'max_features': ['auto', 0.95, 0.90, 0.85, 0.80, 0.75, 0.70],
                          'max_depth': [None, 2, 4, 6, 8],
                          'min_weight_fraction_leaf': [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.05]
                          })

ds_clf = DecisionTreeClassifier(max_depth=8, max_features=0.9, max_leaf_nodes=30,
                       min_impurity_decrease=0.05, min_samples_leaf=0.02,
                       min_samples_split=10, min_weight_fraction_leaf=0.005,
                       random_state=2, splitter='random')
ds_clf.fit(X_train, y_train)
pred4 = ds_clf.predict(X_test)
evaluating_model(y_test, pred4)

from sklearn.ensemble import RandomForestClassifier

randomized_search(params={
                         'min_samples_leaf':[1,2,4,6,8,10,20,30],
                          'min_impurity_decrease':[0.0, 0.01, 0.05, 0.10, 0.15, 0.2],
                          'max_features':['auto', 0.8, 0.7, 0.6, 0.5, 0.4],
                          'max_depth':[None,2,4,6,8,10,20],
                         }, clf=RandomForestClassifier(random_state=2))

rf_clf = RandomForestClassifier(max_depth=2, max_features=0.5,
                       min_impurity_decrease=0.01, min_samples_leaf=10,
                       random_state=2)
rf_clf.fit(X_train, y_train)
pred5 = rf_clf.predict(X_test)
evaluating_model(y_test, pred5)

from xgboost import XGBClassifier
xgb1 = XGBClassifier(colsample_bytree= 1.0,
 learning_rate = 0.1,
 max_depth = 4,
 n_estimators= 400,
 subsample= 1.0)

eval_set  = [(X_test, y_test)]

xgb1.fit(X_train, y_train,early_stopping_rounds=10, eval_metric="logloss",eval_set=eval_set, verbose=True)

pred6 = xgb1.predict(X_test)
evaluating_model(y_test, pred6)
from xgboost import plot_importance
# xgb.feature_importances_
plot_importance(xgb1)
plt.show()
from sklearn.ensemble import GradientBoostingClassifier

gbdt = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,max_depth=1,random_state=0)
gbdt.fit(X_train, y_train)

pred_gdbt = gbdt.predict(X_test)
evaluating_model(y_test, pred_gdbt)
# we will choose XGboost

import joblib
joblib.dump(xgb1, 'model.pkl')
model = joblib.load('model.pkl' )
model.predict(X_test)
