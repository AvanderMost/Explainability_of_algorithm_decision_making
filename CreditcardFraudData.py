# Creditcard fraud Dataset

# run the dependencies
get_ipython().run_line_magic('run', 'Dependencies')

#Read and preview data
df = pd.read_csv(r"C:\Users\Alexandra.vanderMost\Afstuderen\Data\creditcard.csv")
print(df.shape)
print(df.dtypes)
df.head()

# Change plot size
plt.rcParams["figure.figsize"] = (20,10)

# Plot
df.hist()

# statisticsdf.isnull().sum()
df.describe().round(decimals=2)

df['Class'].value_counts()
#print(df.columns)

df.isnull().sum()

df.corr(method = 'pearson').round(decimals=3)

# Feature importance for regression problem
feature_cols = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
X = df[feature_cols].values # Features
y = df.Class # Target variable

# feature extraction
skb = SelectKBest(score_func=f_regression)
skb = skb.fit(X, y)

# List Scores
score = np.array([feature_cols, skb.scores_]).transpose()
scr = pd.DataFrame(data=score, columns=['Feature', 'Score'])
print(scr)

# Plot scores by feature
plt.bar(feature_cols, skb.scores_)
plt.xlabel('Feature')
plt.ylabel('Score')
plt.title('Feature Univariate linear regression test Score')

plt.show()

scr2 = scr.copy().[{'Score': 'int32'}]
scr2.sort_values(by=['Score'], inplace=True)

# Feature importance for classification problem
feature_cols = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
X = df[feature_cols].values # Features
y = df.Class.astype('category') # Target variable

# feature extraction
skb = SelectKBest(score_func=f_classif)
skb = skb.fit(X, y)

# List Scores
score = np.array([feature_cols, skb.scores_]).transpose()
scr = pd.DataFrame(data=score, columns=['Feature', 'Score'])
print(scr)

# Plot scores by feature
plt.bar(feature_cols, skb.scores_)
plt.xlabel('Feature')
plt.ylabel('Score')
plt.title('Feature ANOVA Score')

plt.show()

#function for True possitives, TN FP and FN.
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)


# ## Decision tree classification
# Feature importance for regression problem
feature_cols = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
X = df[feature_cols].values # Features
y = df.Class.astype('category') # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(random_state=1)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy using all available features:",metrics.accuracy_score(y_test, y_pred))

# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

# List feature importance
importance = np.array([feature_cols, clf.feature_importances_]).transpose()
imp = pd.DataFrame(data=importance, columns=['Feature', 'Importance'])
print(imp)

# Plot features importance
plt.bar(feature_cols, clf.feature_importances_)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Models feature importance')

plt.show()

# The "accuracy" scoring is proportional to the number of correct classifications
rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(2), scoring='accuracy')
rfecv.fit(X, y)
print("Optimal number of features : %d" % rfecv.n_features_)

#plot the RDE
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

# Feature importance for regression problem
feature_cols = ['V10','V14','V17']
X = df[feature_cols].values # Features
y = df.Class.astype('category') # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(random_state=1)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy using three features:",metrics.accuracy_score(y_test, y_pred))

# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

# Feature importance for regression problem
feature_cols = ['V10','V14','V17']
X = df[feature_cols].values # Features
y = df.Class.astype('category') # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=1)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy using three features:",metrics.accuracy_score(y_test, y_pred))

# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('BostonHousingDTC.png')
Image(graph.create_png())


# Decision tree regression
# Feature importance for regression problem
feature_cols = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
X = df[feature_cols].values # Features
y = df.Class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree regression object
rgs = DecisionTreeRegressor(random_state=1)

# Train Decision Tree Classifer
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with all available features in R^2:",rgs.score(X_test, y_test))

# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

# Get models feature importance
importance = np.array([feature_cols, rgs.feature_importances_]).transpose()
imp = pd.DataFrame(data=importance, columns=['Feature', 'Importance'])

plt.rcParams["figure.figsize"] = (10,5)
# Plot features importance
plt.bar(feature_cols, rgs.feature_importances_)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Models feature importance')
plt.show()

# The "accuracy" scoring is proportional to the number of correct classifications
rfecv = RFECV(estimator=rgs, step=1, cv=KFold(2))
rfecv.fit(X, y)
print("Optimal number of features : %d" % rfecv.n_features_)

#plot the RDE
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

# Feature for regression problem
feature_cols = ['V17','V10','V14', 'Time']
X = df[feature_cols].values # Features
y = df.Class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree regression object
rgs = DecisionTreeRegressor(random_state=1, max_depth=4)

# Train Decision Tree Classifer
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with all available features in R^2:",rgs.score(X_test, y_test))


dot_data = StringIO()
export_graphviz(rgs, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('CreditcardDTR.png')
Image(graph.create_png())


#Support Vector Machine classification
#new model
feature_cols = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
X = df[feature_cols].values # Features
y = df.Class.astype('category') # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = svm.SVC(random_state=1, kernel='linear')

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with all available features:",metrics.accuracy_score(y_test, y_pred))

# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

#Normalize the input features
feature_cols = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
X = df[feature_cols].values # Features
y = df.Class.astype('category') # Target variable

nrm = preprocessing.Normalizer(norm='l2')
X = nrm.transform(X) # Normalized features

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = svm.SVC(random_state=1, kernel='linear')

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with all available and normalized features:",metrics.accuracy_score(y_test, y_pred))

# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

#new model
feature_cols = ['V17', 'V14']
X = df[feature_cols].values # Features
y = df.Class.astype('category') # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = svm.SVC(random_state=1, kernel='linear')

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with two input features:",metrics.accuracy_score(y_test, y_pred))

# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

#new model
feature_cols = ['V17', 'V14', 'V12']
X = df[feature_cols].values # Features
y = df.Class.astype('category') # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = svm.SVC(random_state=1, kernel='linear')

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with three input features:",metrics.accuracy_score(y_test, y_pred))

# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

#new model
feature_cols = ['V17', 'V14', 'V12', 'V10']
X = df[feature_cols].values # Features
y = df.Class.astype('category') # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = svm.SVC(random_state=1, kernel='linear')

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with four input features:",metrics.accuracy_score(y_test, y_pred))

# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

#new model
feature_cols = ['V17', 'V14', 'V12', 'V10', 'V16']
X = df[feature_cols].values # Features
y = df.Class.astype('category') # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = svm.SVC(random_state=1, kernel='linear')

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with five input features:",metrics.accuracy_score(y_test, y_pred))

# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

#new model
feature_cols = ['V17', 'V14', 'V12', 'V10', 'V16', 'V3']
X = df[feature_cols].values # Features
y = df.Class.astype('category') # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = svm.SVC(random_state=1, kernel='linear')

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with six input features:",metrics.accuracy_score(y_test, y_pred))

# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

feature_cols = ['V17', 'V14', 'V12', 'V10', 'V16']
X = df[feature_cols].values # Features
y = df.Class.astype('category') # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object with kernel rbf
clf = svm.SVC(random_state=1, kernel='rbf', gamma='auto')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with rbf kernel:",metrics.accuracy_score(y_test, y_pred))
# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

# Create Support vector machine classifer object with kernel linear
clf = svm.SVC(random_state=1, kernel='linear')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with linear kernel:",metrics.accuracy_score(y_test, y_pred))
# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

# Create Support vector machine classifer object with kernel poly
clf = svm.SVC(random_state=1, kernel='poly', degree=2, gamma='auto')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with poly kernel:",metrics.accuracy_score(y_test, y_pred))
# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

#new model
feature_cols = ['V17', 'V14', 'V12', 'V10', 'V16']
X = df[feature_cols].values # Features
y = df.Class.astype('category') # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object with kernel rbf
clf = svm.SVC(random_state=1, kernel='rbf', gamma='auto')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with rbf kernel:",metrics.accuracy_score(y_test, y_pred))
# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)


# Support vector machine regression
#new model
feature_cols = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
X = df[feature_cols].values # Features
y = df.Class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = svm.SVR(max_iter = 4000)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with all available features in R^2:",rgs.score(X_test, y_test))
# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

#new model
feature_cols = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
X = df[feature_cols].values # Features
y = df.Class # Target variable

nrm = preprocessing.Normalizer(norm='l2')
X = nrm.transform(X) # Normalized features

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = svm.SVR(kernel='linear')

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with normalized features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['V17', 'V14']
X = df[feature_cols].values # Features
y = df.Class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = svm.SVR(kernel='linear')

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with two input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['V17', 'V14', 'V12']
X = df[feature_cols].values # Features
y = df.Class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = svm.SVR(kernel='linear')

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with three input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['V17', 'V14', 'V12', 'V10']
X = df[feature_cols].values # Features
y = df.Class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = svm.SVR(kernel='linear')

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with four input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['V17', 'V14', 'V12', 'V10', 'V16']
X = df[feature_cols].values # Features
y = df.Class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = svm.SVR(kernel='linear')

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with five input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['V17', 'V14', 'V12', 'V10', 'V16', 'V3']
X = df[feature_cols].values # Features
y = df.Class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = svm.SVR(kernel='linear')

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with six input features in R^2:",rgs.score(X_test, y_test))

#Decrease dataset size
df = df.sample(frac = 0.1, random_state = 1)
df['Class'].value_counts()
#print(df.columns)

feature_cols = ['V17', 'V14', 'V12', 'V10', 'V16', 'V3']
X = df[feature_cols].values # Features
y = df.Class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = svm.SVR(kernel='rbf', gamma='auto')
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with rbf kernel in R^2",rgs.score(X_test, y_test))

# Create Support vector machine regression object
rgs = svm.SVR(kernel='rbf', gamma='auto')
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with rbf kernel in R^2",rgs.score(X_test, y_test))

# Create Support vector machine regression object
rgs = svm.SVR(kernel='linear', gamma='auto')
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with linear kernel in R^2",rgs.score(X_test, y_test))

# Create Support vector machine regression object
rgs = svm.SVR(kernel='linear', gamma='scale')
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with linear kernel in R^2",rgs.score(X_test, y_test))

# Create Support vector machine regression object
rgs_poly = svm.SVR(kernel='poly', gamma='auto', degree=2)
rgs_poly = rgs_poly.fit(X_train,y_train)
y_pred = rgs_poly.predict(X_test)
print("Accuracy with poly kernel degree=2 in R^2",rgs_poly.score(X_test, y_test))

# Create Support vector machine regression object
rgs_poly = svm.SVR(kernel='poly', gamma='scale', degree=2)
rgs_poly = rgs_poly.fit(X_train,y_train)
y_pred = rgs_poly.predict(X_test)
print("Accuracy with poly kernel degree=2 in R^2",rgs_poly.score(X_test, y_test))

#new model
feature_cols = ['V17', 'V14', 'V12', 'V10', 'V16', 'V3']
X = df[feature_cols].values # Features
y = df.Class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = svm.SVR(kernel='sigmoid')
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with sigmoid kernel in R^2",rgs.score(X_test, y_test))

#new model
feature_cols = ['V17', 'V14', 'V12', 'V10', 'V16', 'V3']
X = df[feature_cols].values # Features
y = df.Class # Target variable

nrm = preprocessing.Normalizer(norm='l2')
X = nrm.transform(X) # Normalized features

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs_poly = svm.SVR(kernel='poly', C=30, gamma='auto', degree=2, epsilon=0.5, coef0=1)

# Train model
rgs_poly = rgs_poly.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs_poly.predict(X_test)
# Model Accuracy
print("Accuracy with normalized input features in R^2",rgs_poly.score(X_test, y_test))

plt.rcParams["figure.figsize"] = (7,7)
# Plot the results
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("actual")
plt.ylabel("predicted")
plt.title("support vector machine Regression")
plt.show()


# ## Neural Network Classification
#new model
feature_cols = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
X = df[feature_cols].values # Features
y = df.Class.astype('category') # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=100)

# Train  neural network Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with all available input features:",metrics.accuracy_score(y_test, y_pred))
# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

#new model
feature_cols = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
X = df[feature_cols].values # Features
y = df.Class.astype('category') # Target variable

nrm = preprocessing.Normalizer(norm='l2')
X = nrm.transform(X) # Normalized features

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=100)

# Train  neural network Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with all available normalized input features:",metrics.accuracy_score(y_test, y_pred))
# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

#new model
feature_cols = ['V17', 'V14']
X = df[feature_cols].values # Features
y = df.Class.astype('category') # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = MLPClassifier(random_state=1, max_iter=400)

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with two input features:",metrics.accuracy_score(y_test, y_pred))
# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

#new model
feature_cols = ['V17', 'V14', 'V12']
X = df[feature_cols].values # Features
y = df.Class.astype('category') # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = MLPClassifier(random_state=1, max_iter=400)

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with three input features:",metrics.accuracy_score(y_test, y_pred))
# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

#new model
feature_cols = ['V17', 'V14', 'V12', 'V10']
X = df[feature_cols].values # Features
y = df.Class.astype('category') # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = MLPClassifier(random_state=1, max_iter=400)

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with four input features:",metrics.accuracy_score(y_test, y_pred))
# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

#new model
feature_cols = ['V17', 'V14', 'V12', 'V10', 'V16']
X = df[feature_cols].values # Features
y = df.Class.astype('category') # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = MLPClassifier(random_state=1, max_iter=400)

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with five input features:",metrics.accuracy_score(y_test, y_pred))
# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

#new model
feature_cols = ['V17', 'V14', 'V12', 'V10', 'V16', 'V3']
X = df[feature_cols].values # Features
y = df.Class.astype('category') # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = MLPClassifier(random_state=1, max_iter=400)

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with six input features:",metrics.accuracy_score(y_test, y_pred))
# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

#new model
feature_cols = ['V17','V10','V12','V14']
X = df[feature_cols].values # Features
y = df.Class.astype('category') # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=4000, activation='relu', solver='adam')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with relu activation and solver='adam':",metrics.accuracy_score(y_test, y_pred))
# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

# Create Neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=4000, activation='relu', solver='lbfgs')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with relu activation and solver='lbfgs':",metrics.accuracy_score(y_test, y_pred))
# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

# Create Neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=4000, activation='relu', solver='sgd')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with relu activation and solver='sgd':",metrics.accuracy_score(y_test, y_pred))
# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

# Create Neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=4000, activation='identity', solver='adam')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with identity activation and solver='adam':",metrics.accuracy_score(y_test, y_pred))
# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

# Create Neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=40000, activation='identity', solver='lbfgs')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with identity activation and solver='lbfgs':",metrics.accuracy_score(y_test, y_pred))
# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

# Create Neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=4000, activation='identity', solver='sgd')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with identity activation feature and solver='sgd':",metrics.accuracy_score(y_test, y_pred))
# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

# Create Neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=4000, activation='logistic', solver='adam')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with logistic activation and solver='adam':",metrics.accuracy_score(y_test, y_pred))
# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

# Create Neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=4000, activation='logistic', solver='lbfgs')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with logistic activation and solver='lbfgs':",metrics.accuracy_score(y_test, y_pred))
# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

# Create Neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=4000, activation='logistic', solver='sgd')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with logistic activation and solver='sgd':",metrics.accuracy_score(y_test, y_pred))
# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

# Create Neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=4000, activation='tanh', solver ='adam')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with tangus hyperbolicus activation andsolver='adam' features:",metrics.accuracy_score(y_test, y_pred))
# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

# Create Neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=4000, activation='tanh', solver ='lbfgs')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with tangus hyperbolicus activation and solver='lbfgs':",metrics.accuracy_score(y_test, y_pred))
# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

# Create Neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=4000, activation='tanh', solver ='sgd')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with tangus hyperbolicus activation and solver='sgd':",metrics.accuracy_score(y_test, y_pred))
# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

#new model
feature_cols = ['V17', 'V14', 'V12', 'V10']
X = df[feature_cols].values # Features
y = df.Class.astype('category') # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=4000, hidden_layer_sizes=(120,80,80),  activation='identity', solver='lbfgs',batch_size=700, alpha=0.05)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with activation='identity', solver='lbfgs' and alpha=0.05:",metrics.accuracy_score(y_test, y_pred))
print("and the number of layers used:", clf.n_layers_)
# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)

# Create Neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=4000, hidden_layer_sizes=(),  activation='identity', solver='lbfgs',batch_size=200, alpha=0.001)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with activation='identity', solver='lbfgs' and alpha=0.001:",metrics.accuracy_score(y_test, y_pred))
print("and the number of layers used:", clf.n_layers_)
# Amount of true positives, false positive, true negative and false negative
TP, FP, TN, FN = perf_measure(y_test.values, y_pred)
print("TP:",TP," FP:", FP," TN:", TN," FN:", FN)


# Neural Network regression
#new model
feature_cols = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
X = df[feature_cols].values # Features
y = df.Class # Target variable

#nrm = preprocessing.Normalizer(norm='l2')
#X = nrm.transform(X) # Normalized features

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, max_iter = 100)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with all available features in R^2:",rgs.score(X_test, y_test))


plt.rcParams["figure.figsize"] = (7,7)
# Plot the results
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("actual")
plt.ylabel("predicted")
plt.title("neural network Regression")
plt.show()

#new model
feature_cols = ['V17', 'V14']
X = df[feature_cols].values # Features
y = df.Class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, max_iter = 2000)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with two input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['V17', 'V14', 'V12']
X = df[feature_cols].values # Features
y = df.Class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, max_iter = 2000)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with three input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['V17', 'V14', 'V12', 'V10']
X = df[feature_cols].values # Features
y = df.Class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, max_iter = 2000)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with four input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['V17', 'V14', 'V12', 'V10', 'V16']
X = df[feature_cols].values # Features
y = df.Class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, max_iter = 2000)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with five input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['V17', 'V14', 'V12', 'V10', 'V16', 'V3']
X = df[feature_cols].values # Features
y = df.Class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, max_iter = 2000)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with six input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['V17', 'V14', 'V12', 'V10']
X = df[feature_cols].values # Features
y = df.Class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, max_iter = 8000)
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with activation='relu', solver='adam'in R^2:",rgs.score(X_test, y_test))

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, max_iter = 8000, activation='logistic')
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with logistic activation, solver='adam'in R^2:",rgs.score(X_test, y_test))

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, hidden_layer_sizes=(100,), max_iter = 8000, activation='relu', solver='lbfgs', alpha=0.1)
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with activation='relu', solver='lbfgs' in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['V17', 'V14', 'V12', 'V10']
X = df[feature_cols].values # Features
y = df.Class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, max_iter = 8000)
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with five input features in R^2:",rgs.score(X_test, y_test))

plt.rcParams["figure.figsize"] = (7,7)
# Plot the results
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("actual")
plt.ylabel("predicted")
plt.title("neural network Regression")
plt.show()


# Linear regression
#new model
feature_cols = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
X = df[feature_cols].values # Features
y = df.Class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = linear_model.LinearRegression()

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with all available features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
X = df[feature_cols].values # Features
y = df.Class # Target variable

nrm = preprocessing.Normalizer(norm='l2')
X = nrm.transform(X) # Normalized features

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = linear_model.LinearRegression()

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with all available normalized features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
X = df[feature_cols].values # Features
y = df.Class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = linear_model.LinearRegression()

# The "accuracy" scoring is proportional to the number of correct classifications
rfecv = RFECV(estimator=rgs, step=1, cv=KFold(2))
rfecv.fit(X, y)
print("Optimal number of features : %d" % rfecv.n_features_)


plt.rcParams["figure.figsize"] = (10,5)
#plot the RDE
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

#new model
feature_cols = ['V17', 'V14', 'V12', 'V10', 'V16']
X = df[feature_cols].values # Features
y = df.Class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = linear_model.LinearRegression(fit_intercept=True)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with five input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['V17','V10', 'V12','V14','V16','V3','V7','V11','V4','V18','V1','V9','V5','V2','V6']
X = df[feature_cols].values # Features
y = df.Class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = linear_model.LinearRegression(fit_intercept=True)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with 15 input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['V17','V10', 'V12','V14','V16','V3','V7','V11','V4','V18','V1','V9','V5','V2','V6', 'V21']
X = df[feature_cols].values # Features
y = df.Class # Target variable


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = linear_model.LinearRegression(fit_intercept=True)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with 16 input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['V17','V10', 'V12','V14','V16','V3','V7','V11','V4','V18','V1','V9','V5','V2','V6','V21','V19']
X = df[feature_cols].values # Features
y = df.Class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = linear_model.LinearRegression(fit_intercept=True)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with 17 input features in R^2:",rgs.score(X_test, y_test))

