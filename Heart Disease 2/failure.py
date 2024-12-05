import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier,plot_importance



df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# print(df.head()) 
# print(df.shape)
# print(df.info())
# print(df.describe())
# print(df.isnull().sum().sum())

#                                    Exploratory Data Analysis - EDA

len_live = len(df["DEATH_EVENT"][df.DEATH_EVENT == 0])
len_death = len(df["DEATH_EVENT"][df.DEATH_EVENT == 1])

# arr = np.array([len_live,len_death])
# labels = ['LIVING', 'DIED']
# print("total no.of living cases:",len_live)
# print("total no.of died cases:",len_death)

# plt.pie(arr, labels=labels, explode= [0.1,0.0], shadow=True)

# sns.distplot(df["age"]) # data distribution of age
# plt.show()

#                   selecting columns that are above age 50 and seeing died or not.

age_above_50_not_died = df['DEATH_EVENT'][df.age>=50][df.DEATH_EVENT == 0]
age_above_50_died = df['DEATH_EVENT'][df.age>=50][df.DEATH_EVENT == 1]

# len_died = len(age_above_50_died)
# len_not_died = len(age_above_50_not_died)

# arr1 = np.array([len_died,len_not_died])
# labels = ['DIED','NOT DIED']

# print("above 50 age, died cases :",len_died)
# print("above 50 age,living cases :",len_not_died)

# plt.pie(arr1,labels=labels,explode=[0.2,0.0],shadow=True)
# plt.show()

#           Selecting the cases which are died or not with diabetes.

patient_have_diabetes_alive = df["DEATH_EVENT"][df.diabetes == 0][df.DEATH_EVENT == 0]
patient_have_diabetes_died = df["DEATH_EVENT"][df.diabetes == 1][df.DEATH_EVENT == 1]

# len_d_died = len(patient_have_diabetes_died)
# len_d_alive = len(patient_have_diabetes_alive)

# arr3 = np.array([len_d_alive,len_d_died])
# labels = ['Not DIED with Diabetes', 'DIED with Diabetes']
# print("no.of living cases:",len_d_alive)
# print("no.of died cases:",len_d_died)

# plt.pie(arr3, labels=labels, explode= [0.2,0.0], shadow=True)
# plt.show()

#                              Correlation Matrix

# cor = df.corr()
# plt.subplots(figsize = (15,15))
# sns.heatmap(cor,annot=True)
# plt.show()

#      Dataset Development

x = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

X_train,X_test,Y_train,Y_test = train_test_split(x, y, test_size=0.25, random_state=42)


#          Feature Engineering

# def add_interaction(x):
#     features = x.columns
#     m = len(features)
#     x_int = x.copy(deep=True)
    
#     for i in range(m):
#         feature_i_name =features[i]
#         feature_i_data = x[feature_i_name]

#         for j in range(i+1, m):
#             feature_j_name = features[j]
#             feature_j_data = x[feature_j_name]
#             x_int[feature_i_j_name] = feature_i_data * feature_j_data

#     return x_int

# x_train_mod = add_interaction(X_train)
# x_test_mode = add_interaction(X_test)

#             Model Building

def evaluating_model(Y_test, y_pred):

    # Function for evaluating the models

    print("Accuracy Score :",accuracy_score(Y_test,y_pred))
    print("Precision Score :",precision_score(Y_test,y_pred))
    print("Recall Score :",recall_score(Y_test,y_pred))
    print("Confusion Matrix : \n",confusion_matrix(Y_test,y_pred))


#       Building Logistic regression model

Log_R = LogisticRegression(max_iter=150)
Log_R.fit(X_train,Y_train)

Log_R_Predict = Log_R.predict(X_test)

y_pred = Log_R.predict(X_test)
# evaluating_model(Y_test,y_pred)

#       Logistic Regression with StandardScaler

Log_R_Pip = make_pipeline(StandardScaler(), LogisticRegression())
Log_R_Pip.fit(X_train,Y_train)

y_pred1 = Log_R_Pip.predict(X_test)
# evaluating_model(Y_test,y_pred1)

#      Support Vector Machine

param_grid ={ 'C':[0.1, 1, 10, 100, 1000],
              'gamma':[1, 0.1, 0.01, 0.001, 0.0001],
              'kernel':['rbf']
            }
# grid = GridSearchCV(SVC(), param_grid, refit=True, verbose =3)
# grid.fit(X_train,Y_train)
# print(grid.best_estimator_)

svc = SVC(C=10, gamma=0.0001)
svc.fit(X_train,Y_train)
y_pred2 = svc.predict(X_test)
# evaluating_model(Y_test,y_pred2)

#      Decision Tree 

# def randomized_search(params,runs=20,clf=DecisionTreeClassifier(random_state=2)):
#     rand_clf = RandomizedSearchCV(clf, params, n_iter=runs, cv=5, n_jobs=-1, random_state=2)
#     rand_clf.fit(X_train,Y_train)
#     best_model = rand_clf.best_estimator_
#     best_score = rand_clf.best_score_

#     print("Trainig Score: {:.3f}".format(best_score))
#     y_pred3 = best_model.predict(X_test)
#     acc = accuracy_score(Y_test,y_pred3)
#     print("Test Score: {:.3f}".format(acc))

#     return best_model

# randomized_search(params={'criterion':['entropy','gini'],
#                         'splitter':['random','best'],
#                         'min_weight_fraction_leaf':[0.0, 0.0025, 0.005, 0.0075, 0.01],
#                         'min_samples_split':[2, 3, 4, 5, 6, 8, 10],
#                         'min_samples_leaf':[1, 0.01, 0.02, 0.03, 0.04],
#                         'min_impurity_decrease':[0.0, 0.0005, 0.005, 0.05, 0.10, 0.15, 0.2],
#                         'max_leaf_nodes':[10,15,20,25,30,40,45,50,None],
#                         'max_features':['auto', 0.95, 0.90, 0.85, 0.80, 0.75, 0.70],
#                         'max_depth':[None, 2, 4, 6, 8],
#                         'min_weight_fraction_leaf':[0.0, 0.0025, 0.005, 0.0075, 0.01, 0.05]
#                 })

ds_clf = DecisionTreeClassifier(max_depth=8, max_features=0.75, criterion="entropy",max_leaf_nodes=30,
                                min_impurity_decrease=0.05,min_samples_leaf=0.02,min_samples_split=10,
                                min_weight_fraction_leaf=0.005,random_state=2,splitter='random')
                            
ds_clf.fit(X_train,Y_train)
y_pred4 = ds_clf.predict(X_test)
# evaluating_model(Y_test,y_pred4)

#        Random Forest Classifier

rf_clf = RandomForestClassifier(max_depth=2, max_features=0.5, min_impurity_decrease=0.01, min_samples_leaf=10,random_state=2)
rf_clf.fit(X_train,Y_train)
y_pred5 = rf_clf.predict(X_test)
# evaluating_model(Y_test,y_pred5)

#        XGBoost Classifier

# xgb1 = XGBClassifier(colsample_bytree=0.1,learning_rate = 0.1, max_depth=4, n_estimators=400, subsamples = 1.0)
# eval_set =[(X_test,Y_test)]

# xgb1.fit(X_train,Y_train,
#         eval_set=eval_set,
#         # early_stopping_rounds=10, 
#         # eval_metric = "logloss", 
#         verbose=True)
# y_pred6 = xgb1.predict(X_test)
# evaluating_model(Y_test,y_pred6)

# # Feature Importance

# plot_importance(xgb1)
# plt.show()


#      Gradient Boosting Classifier

gb_clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=1, random_state=0)
gb_clf.fit(X_train,Y_train)

y_pred6 = gb_clf.predict(X_test)
evaluating_model(Y_test,y_pred6)