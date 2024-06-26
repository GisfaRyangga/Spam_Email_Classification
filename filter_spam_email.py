import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
os.getcwd()

from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.model_selection import train_test_split

print('-----==== STARTING PROGRAM ====-----')

df = pd.read_csv('SpamEmailClassificationDataset.csv')


df.isnull().sum()
df.dtypes
df['label'].value_counts()

X = df['text']
Y = df['label']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

print(X.shape)
print(X_train.shape)
print(X_test.shape)

print('-----==== FEATURE EXTRACTION ====-----')

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import time

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#DECISION TREE
print('-----==== STARTING DECISION TREE ====-----')

dtrees = DecisionTreeClassifier()
start_time = time.time()
dtrees.fit(X_train_features, Y_train)
end_time = time.time()
dt_training_time = end_time - start_time

dt_train = dtrees.predict(X_train_features)
dt_test = dtrees.predict(X_test_features)

dt_train_acc = accuracy_score(Y_train, dt_train)
dt_test_acc = accuracy_score(Y_test, dt_test)

dt_precision = precision_score(Y_test, dt_test)
dt_recall = recall_score(Y_test, dt_test)
dt_f1 = f1_score(Y_test, dt_test)

print("\nDecision Tress:\n")
print("Training Time          :", dt_training_time, "seconds")
print("Training Data Accuracy:", dt_train_acc)
print("Testing Data Accuracy :", dt_test_acc)

print("Precision             :", dt_precision)
print("Recall                :", dt_recall)
print("F1 Score              :", dt_f1)

#LOGISTIC REGRESSION
print('-----==== STARTING LOGISTIC REGRESSION ====-----')

lr = LogisticRegression()
start_time = time.time()
lr.fit(X_train_features, Y_train)
end_time = time.time()
lr_training_time = end_time - start_time

lr_train = lr.predict(X_train_features)
lr_test = lr.predict(X_test_features)

lr_train_acc = accuracy_score(Y_train, lr_train)
lr_test_acc = accuracy_score(Y_test, lr_test)

lr_precision = precision_score(Y_test, lr_test)
lr_recall = recall_score(Y_test, lr_test)
lr_f1 = f1_score(Y_test, lr_test)


print("\nLogistic Regression:\n")
print("Training Time          :", lr_training_time, "seconds")
print("Training Data Accuracy:", lr_train_acc)
print("Testing Data Accuracy :", lr_test_acc)

print("Precision             :", lr_precision)
print("Recall                :", lr_recall)
print("F1 Score              :", lr_f1)

#KNN
# from datetime import datetime

print('-----==== STARTING KNN ====-----')

knn = KNeighborsClassifier()
start_time = time.time()
knn.fit(X_train_features, Y_train)
end_time = time.time()
knn_training_time = end_time - start_time

knn_train = knn.predict(X_train_features)
knn_test = knn.predict(X_test_features)

knn_train_acc = accuracy_score(Y_train, knn_train)
knn_test_acc = accuracy_score(Y_test, knn_test)

knn_precision = precision_score(Y_test, knn_test)
knn_recall = recall_score(Y_test, knn_test)
knn_f1 = f1_score(Y_test, knn_test)
estimated_time = time.time() - start_time


print("\nK Nearest Neighbors:\n")
print("Training Time            :", knn_training_time, "seconds")
print('Training Time (estimated):', estimated_time, 'seconds')
print("Training Data Accuracy   :", knn_train_acc)
print("Testing Data Accuracy    :", knn_test_acc)

print("Precision                :", knn_precision)
print("Recall                   :", knn_recall)
print("F1 Score                 :", knn_f1)

#RANDOM FOREST
print('-----==== STARTING RANDOM FOREST ====-----')

rf = RandomForestClassifier()
start_time = time.time()
rf.fit(X_train_features, Y_train)
end_time = time.time()
rf_training_time = end_time - start_time

rf_train = rf.predict(X_train_features)
rf_test = rf.predict(X_test_features)

rf_train_acc = accuracy_score(Y_train, rf_train)
rf_test_acc = accuracy_score(Y_test, rf_test)

rf_precision = precision_score(Y_test, rf_test)
rf_recall = recall_score(Y_test, rf_test)
rf_f1 = f1_score(Y_test, rf_test)


print("\nRandom Forest:\n")
print("Training Time          :", rf_training_time, "seconds")
print("Training Data Accuracy:", rf_train_acc)
print("Testing Data Accuracy :", rf_test_acc)

print("Precision             :", rf_precision)
print("Recall                :", rf_recall)
print("F1 Score              :", rf_f1)

print('-----==== STACKING CLASSIFIER ====-----')

from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
import joblib

estimators = [ ('dtree', dtrees), ('lr', lr), ('knn', knn), ('rf', rf) ] 
stack = StackingClassifier(estimators, final_estimator = SVC(kernel='linear'))
start_time = time.time()
stack.fit(X_train_features, Y_train)
end_time = time.time()
stack_training_time = end_time - start_time

stack_train = stack.predict(X_train_features)
stack_test = stack.predict(X_test_features)

stack_train_acc = accuracy_score(Y_train, stack_train)
stack_test_acc = accuracy_score(Y_test, stack_test)

stack_precision = precision_score(Y_test, stack_test)
stack_recall = recall_score(Y_test, stack_test)
stack_f1 = f1_score(Y_test, stack_test)


print("\nStacking Classifier:\n")
print("Training Time          :", stack_training_time, "seconds")
print("Training Data Accuracy:", stack_train_acc)
print("Testing Data Accuracy :", stack_test_acc)

print("Precision             :", stack_precision)
print("Recall                :", stack_recall)
print("F1 Score              :", stack_f1)

print('-----==== SAVING MODELS ====-----')

joblib.dump(stack, 'stack.model')
joblib.dump(feature_extraction, 'feature_extraction.pkl')

print('-----==== END ====-----')

train_acc_list = {"LR":lr_train_acc,
                  "DT":dt_train_acc,
                  "KNN":knn_train_acc,
                  "RF":rf_train_acc,
                  "Stack":stack_train_acc
                 }

test_acc_list = {"LR":lr_test_acc,
                  "DT":dt_test_acc,
                  "KNN":knn_test_acc,
                  "RF":rf_test_acc,
                  "Stack":stack_test_acc
                }

precision_list = {"LR":lr_precision,
                  "DT":dt_precision,
                  "KNN":knn_precision,
                  "RF":rf_precision,
                  "Stack":stack_precision
                 }

recall_list = {"LR":lr_recall,
               "DT":dt_recall,
               "KNN":knn_recall,
               "RF":rf_recall,
               "Stack":stack_recall
              }

f1_list = {"LR":lr_f1,
           "DT":dt_f1,
           "KNN":knn_f1,
           "RF":rf_f1,
           "Stack":stack_f1   
          }

training_time_list = {"LR":lr_training_time,
           "DT":dt_training_time, 
           "KNN":knn_training_time, 
           "RF":rf_training_time, 
           "Stack":stack_training_time,
          }

a1 =  pd.DataFrame.from_dict(train_acc_list, orient = 'index', columns = ["Traning Accuracy"])
a2 =  pd.DataFrame.from_dict(test_acc_list, orient = 'index', columns = ["Testing Accuracy"])
a3 =  pd.DataFrame.from_dict(precision_list, orient = 'index', columns = ["Precision Score"])
a4 =  pd.DataFrame.from_dict(recall_list, orient = 'index', columns = ["Recall Score"])
a5 =  pd.DataFrame.from_dict(f1_list, orient = 'index', columns = ["F1 Score"])
a6 =  pd.DataFrame.from_dict(training_time_list, orient = 'index', columns = ["Training Time"]) 

org = pd.concat([a1, a2, a3, a4, a5, a6], axis = 1)
org


alg = ['LR','DT','KNN','RF', 'Stack']
plt.plot(alg,a1)
plt.plot(alg,a2)
plt.plot(alg,a3)
plt.plot(alg,a4)
plt.plot(alg,a5)
plt.plot(alg,a6)
legend = ['Traning Accuracy', 'Testing Accuracy', 'Precision Score', 'Recall Score', 'F1 Score']
plt.title("METRICS COMPARISION")
plt.legend(legend)
plt.show()

#predict
input_mail = ["You're receiving this email because you turned on Location History, a Google Account-level setting that creates Timeline, a personal map of your visited places, routes, and trips."]

input_mail_features = feature_extraction.transform(input_mail)

prediction = stack.predict(input_mail_features)

if(prediction == 0):
    print("SPAM MAIL")
else:
    print("HAM MAIL")


#predict
input_mail = ["date with me"]

input_mail_features = feature_extraction.transform(input_mail)

prediction = stack.predict(input_mail_features)

if(prediction == 0):
    print("SPAM MAIL")
else:
    print("HAM MAIL")