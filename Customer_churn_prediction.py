import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the dataset
data = pd.read_excel('customer_churn_large_dataset.xlsx')
df = pd.DataFrame(data)
df.head()
df.tail()

# Performing initial data exploration
print(df.shape)
print(df.info())
print(df.describe())
print(df.columns)
print(df.isnull().sum())

# Findind outliers
print(sns.boxplot(df['Monthly_Bill']))
print(sns.boxplot(df['Age']))
print(sns.boxplot(df['Gender']))
print(sns.boxplot(df['Subscription_Length_Months']))

# One hot Encoding and Spliting the dataset
df = pd.get_dummies(df, columns = ['Gender', 'Location'], drop_first=True)
y = df['Churn']
X = df.drop(['Churn','Name', 'CustomerID'], axis = 1)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
column_ss = ['Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']
df[column_ss] = ss.fit_transform(df[column_ss])

#  Building the different models for comparision 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

#Logistic Regression
logreg_model = LogisticRegression()
logreg_model.fit(X_train,y_train)
log_pred = logreg_model.predict(X_test)
logreg_accuracy = round(metrics.accuracy_score(y_test, log_pred) * 100, 2)
print(logreg_accuracy)

# K-Nearest_Neighbor
knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_model.fit(X_train, y_train) 
knn_pred = knn_model.predict(X_test)
knn_accuracy = round(metrics.accuracy_score(y_test, knn_pred) * 100, 2)
print(knn_accuracy)

# Decision Tree
dt_model = DecisionTreeClassifier(criterion = "gini", random_state = 50)
dt_model.fit(X_train, y_train) 
dt_pred = dt_model.predict(X_test)
dt_accuracy = round(metrics.accuracy_score(y_test, dt_pred) * 100, 2)
print(dt_accuracy)

# Random Forest
rf_model = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
rf_model.fit(X_train, y_train) 
rf_pred = rf_model.predict(X_test)
rf_accuracy = round(metrics.accuracy_score(y_test, rf_pred) * 100, 2)
print(rf_accuracy)

Model_Comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'K-Nearest Neighbor', 'Decision Tree', 'Random Forest'],
    'Accuracy': [logreg_accuracy, knn_accuracy, dt_accuracy, rf_accuracy]})
Model_Comparison_df = Model_Comparison.sort_values(by='Accuracy', ascending=False)
Model_Comparison_df = Model_Comparison_df.set_index('Accuracy')
print(Model_Comparison_df.reset_index())

# Logistic Regression Model Optimization

from sklearn.model_selection import GridSearchCV
param_grid = [    
    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000,2500, 5000]
    }
]
clf = GridSearchCV(logreg_model, param_grid = param_grid, cv = 3, verbose=True, n_jobs=-1)
best_clf = clf.fit(X,y)

logreg_model = LogisticRegression(random_state=None, C=3792.690190732246, max_iter=1000,  penalty='l2',  solver='newton-cg')
logreg_model.fit(X_train,y_train)
log_pred = logreg_model.predict(X_test)
logreg_accuracy = round(metrics.accuracy_score(y_test, log_pred) * 100, 2)
print(classification_report(y_test, log_pred))



# Pickling the model
import pickle
filename = 'model.sav'
pickle.dump(logreg_model, open(filename, 'wb'))
load_model = pickle.load(open(filename, 'rb'))
model_score_r1 = load_model.score(xr_test1, yr_test1)
print(prmodel_score_r1)

