# Heart Disease Dataset

# run the dependencies
get_ipython().run_line_magic('run', 'Dependencies')

#Read and preview data
df = pd.read_csv(r"C:\Users\Alexandra.vanderMost\Afstuderen\Data\Heart.csv")
print(df.shape)

##print(df.dtypes)
df.head()

# Change plot size
plt.rcParams["figure.figsize"] = (20,10)

# Plot
df.hist()

# statistics
df.describe().round(decimals=2)
df.isnull().sum()

#measure column correlations
df.corr(method = 'pearson').round(decimals=3)

# Feature importance
# Feature importance for regression problem
feature_cols = ['age','sex','trestbps','chol','fbs','restecg','exang','oldpeak','slope','ca','thal']
X = df[feature_cols].values # Features
y = df.target # Target variable

# feature extraction
skb = SelectKBest(score_func=f_regression)
skb = skb.fit(X, y)

# List Scores
score = np.array([feature_cols, skb.scores_]).transpose()
scr = pd.DataFrame(data=score, columns=['Feature', 'Score'])
print(scr)

# Plot scores by feature
plt.rcParams["figure.figsize"] = (10,6)
plt.bar(feature_cols, skb.scores_)
plt.xlabel('Feature')
plt.ylabel('Score')
plt.title('Feature Univariate linear regression test Score')

plt.show()

# Feature importance for classification problem
feature_cols = ['age','sex','trestbps','chol','fbs','restecg','exang','oldpeak','slope','ca','thal']
X = df[feature_cols].values # Features
y = df.target.astype('category') # Target variable

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


# Decision Tree Classification
#new model
feature_cols = ['age','sex','trestbps','chol','fbs','restecg','exang','oldpeak','slope','ca','thal']
X = df[feature_cols].values # Features
y = df.target.astype('category') # Target variable

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

#new model
feature_cols = ['exang', 'age', 'ca', 'oldpeak', 'thal', 'trestbps', 'chol']
X = df[feature_cols].values # Features
y = df.target.astype('category') # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy using 'trestbps', 'chol' features:",metrics.accuracy_score(y_test, y_pred))

#new model
feature_cols = ['exang', 'age', 'ca', 'oldpeak', 'thal', 'trestbps', 'sex']
X = df[feature_cols].values # Features
y = df.target.astype('category') # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy using 'trestbps', 'sex' features:",metrics.accuracy_score(y_test, y_pred))

#new model
feature_cols = ['exang', 'age', 'ca', 'oldpeak', 'thal', 'sex', 'chol']
X = df[feature_cols].values # Features
y = df.target.astype('category') # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy using 'sex', 'chol' features:",metrics.accuracy_score(y_test, y_pred))

#new model
feature_cols = ['exang', 'age', 'ca', 'oldpeak', 'thal', 'trestbps', 'slope']
X = df[feature_cols].values # Features
y = df.target.astype('category') # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy using 'trestbps', 'slope' features:",metrics.accuracy_score(y_test, y_pred))

#new model
feature_cols = ['exang', 'age', 'ca', 'oldpeak', 'thal', 'slope', 'chol']
X = df[feature_cols].values # Features
y = df.target.astype('category') # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy using 'slope', 'chol' features:",metrics.accuracy_score(y_test, y_pred))

#new model
feature_cols = ['exang', 'age', 'ca', 'oldpeak', 'thal', 'slope', 'age']
X = df[feature_cols].values # Features
y = df.target.astype('category') # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy using 'slope', 'age' features:",metrics.accuracy_score(y_test, y_pred))

#new model
feature_cols = ['exang', 'age', 'ca', 'oldpeak', 'thal', 'trestbps', 'sex']
X = df[feature_cols].values # Features
y = df.target.astype('category') # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=1)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy after optimization:",metrics.accuracy_score(y_test, y_pred))

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


# Decision Tree Regression
#new model
feature_cols = ['age','sex','trestbps','chol','fbs','restecg','exang','oldpeak','slope','ca','thal']
X = df[feature_cols].values # Features
y = df.target # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create a Decision Tree Classifer 
rgs = DecisionTreeRegressor(criterion = 'mse',random_state=1)
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with criterion = 'mse' in R^2:",rgs.score(X_test, y_test))

# Create a Decision Tree Classifer 
rgs = DecisionTreeRegressor(criterion = 'friedman_mse',random_state=1)
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with criterion = 'friedman_mse' in R^2:",rgs.score(X_test, y_test))

# Create a Decision Tree Classifer 
rgs = DecisionTreeRegressor(criterion = 'mae',random_state=1)
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with criterion = 'mae' in R^2:",rgs.score(X_test, y_test))

# Create a Decision Tree Classifer 
rgs = DecisionTreeRegressor(criterion = 'mse', splitter = 'random',random_state=1)
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with criterion = 'mse', splitter = 'random' in R^2:",rgs.score(X_test, y_test))

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

#new model
feature_cols = ['exang']
X = df[feature_cols].values # Features
y = df.target # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

rgs = DecisionTreeRegressor(criterion = 'mse',max_depth=3,random_state=1)

# Train Decision Tree Classifer
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with one feature in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['exang', 'ca']
X = df[feature_cols].values # Features
y = df.target # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

rgs = DecisionTreeRegressor(criterion = 'mse',max_depth=4, random_state=1)

# Train Decision Tree Classifer
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with two features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['exang', 'ca', 'trestbps']
X = df[feature_cols].values # Features
y = df.target # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

rgs = DecisionTreeRegressor(criterion = 'mse',max_depth=5, random_state=1)

# Train Decision Tree Classifer
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with three features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['exang', 'ca']
X = df[feature_cols].values # Features
y = df.target # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

rgs = DecisionTreeRegressor(criterion = 'mse',max_depth=4, random_state=1)

# Train Decision Tree Classifer
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with two features in R^2:",rgs.score(X_test, y_test))

dot_data = StringIO()
export_graphviz(rgs, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('BostonHousingDTR.png')
Image(graph.create_png())


# Support vector machine classification
#new model
feature_cols = ['age','sex','trestbps','chol','fbs','restecg','exang','oldpeak','slope','ca','thal']
X = df[feature_cols].values # Features
y = df.target.astype('category') # Target variable

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

#new model
feature_cols = ['age','sex','trestbps','chol','fbs','restecg','exang','oldpeak','slope','ca','thal']
X = df[feature_cols].values # Features
y = df.target.astype('category') # Target variable

nrm = preprocessing.Normalizer(norm='l1')
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

#new model
feature_cols = ['exang', 'oldpeak']
X = df[feature_cols].values # Features
y = df.target.astype('category') # Target variable

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

#new model
feature_cols = ['exang', 'oldpeak', 'ca']
X = df[feature_cols].values # Features
y = df.target.astype('category') # Target variable

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

#new model
feature_cols = ['exang', 'oldpeak', 'ca', 'slope']
X = df[feature_cols].values # Features
y = df.target.astype('category') # Target variable

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

#new model
feature_cols = ['exang', 'oldpeak', 'ca', 'slope', 'thal']
X = df[feature_cols].values # Features
y = df.target.astype('category') # Target variable

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

#new model
feature_cols = ['exang', 'oldpeak', 'ca', 'slope', 'thal', 'sex']
X = df[feature_cols].values # Features
y = df.target.astype('category') # Target variable

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

#new model
feature_cols = ['exang', 'oldpeak', 'ca', 'slope', 'thal', 'sex', 'age']
X = df[feature_cols].values # Features
y = df.target.astype('category') # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = svm.SVC(random_state=1, kernel='linear')

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with seven input features:",metrics.accuracy_score(y_test, y_pred))

#new model
feature_cols = ['exang', 'oldpeak', 'ca', 'slope', 'thal', 'sex', 'age', 'trestbps']
X = df[feature_cols].values # Features
y = df.target.astype('category') # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = svm.SVC(random_state=1, kernel='linear')

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with eight input features:",metrics.accuracy_score(y_test, y_pred))

#new model
feature_cols = ['exang', 'oldpeak', 'ca', 'slope', 'thal', 'sex', 'age']
X = df[feature_cols].values # Features
y = df.target.astype('category') # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object with kernel rbf
clf = svm.SVC(random_state=1, kernel='rbf', gamma='auto')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with rbf kernel:",metrics.accuracy_score(y_test, y_pred))

# Create Support vector machine classifer object with kernel linear
clf = svm.SVC(random_state=1, kernel='linear')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with linear kernel:",metrics.accuracy_score(y_test, y_pred))

# Create Support vector machine classifer object with kernel poly
clf = svm.SVC(random_state=1, kernel='poly', degree=2, gamma='auto')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with poly kernel:",metrics.accuracy_score(y_test, y_pred))


# Support vector machine regression
#new model
feature_cols = ['age','sex','trestbps','chol','fbs','restecg','exang','oldpeak','slope','ca','thal']
X = df[feature_cols].values # Features
y = df.target # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = svm.SVR(kernel='linear')

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with all available features and rbf kernel in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['age','sex','trestbps','chol','fbs','restecg','exang','oldpeak','slope','ca','thal']
X = df[feature_cols].values # Features
y = df.target # Target variable

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
print("Accuracy with all available features and linear kernel in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['exang', 'oldpeak']
X = df[feature_cols].values # Features
y = df.target # Target variable

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
feature_cols = ['exang', 'oldpeak', 'ca']
X = df[feature_cols].values # Features
y = df.target # Target variable

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
feature_cols = ['exang', 'oldpeak', 'ca', 'slope']
X = df[feature_cols].values # Features
y = df.target # Target variable

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
feature_cols = ['exang', 'oldpeak', 'ca', 'slope', 'thal']
X = df[feature_cols].values # Features
y = df.target # Target variable

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
feature_cols = ['exang', 'oldpeak', 'ca', 'slope', 'thal', 'sex']
X = df[feature_cols].values # Features
y = df.target # Target variable

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

#new model
feature_cols = ['exang', 'oldpeak', 'ca', 'slope', 'thal', 'sex', 'age']
X = df[feature_cols].values # Features
y = df.target # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = svm.SVR(kernel='linear')

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with seven input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['exang', 'oldpeak', 'ca', 'slope', 'thal', 'sex', 'age']
X = df[feature_cols].values # Features
y = df.target # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = svm.SVR(kernel='rbf', C=10)
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with kernel rbf in R^2:",rgs.score(X_test, y_test))

# Create Support vector machine regression object
rgs = svm.SVR(kernel='linear', C=10)
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with kernel linear and C=10 in R^2:",rgs.score(X_test, y_test))

# Create Support vector machine regression object
rgs = svm.SVR(kernel='linear', C=15)
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with kernel linaer and C=15 in R^2:",rgs.score(X_test, y_test))

# Create Support vector machine regression object
rgs = svm.SVR(kernel='poly', degree=2)
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with kernel ploy of second degree in R^2:",rgs.score(X_test, y_test))

# Create Support vector machine regression object
rgs = svm.SVR(kernel='poly', degree=3)
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with kernel ploy of thirth degree and gamma='scale' in R^2:",rgs.score(X_test, y_test))

#The input features
feature_cols = ['exang', 'oldpeak', 'ca', 'slope', 'thal', 'sex', 'age']
X = df[feature_cols].values # Features
y = df.target # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs_poly = svm.SVR(kernel='poly', gamma='auto', degree=2, epsilon=0.05, coef0=1)

# Train model
rgs_poly = rgs_poly.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs_poly.predict(X_test)
# Model Accuracy
print("Accuracy with kernel='poly', gamma='auto', degree=2, epsilon=0.05, coef0=1 in R^2",rgs_poly.score(X_test, y_test))

plt.rcParams["figure.figsize"] = (7,7)
# Plot the results
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("actual")
plt.ylabel("predicted")
plt.title("support vector machine Regression")
plt.show()


# Neural Network Classification
#new model
feature_cols = ['age','sex','trestbps','chol','fbs','restecg','exang','oldpeak','slope','ca','thal']
X = df[feature_cols].values # Features
y = df.target.astype('category')# Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = MLPClassifier(random_state=1, max_iter=400)

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with all available input features:",metrics.accuracy_score(y_test, y_pred))


#Normalized input features
feature_cols = ['age','sex','trestbps','chol','fbs','restecg','exang','oldpeak','slope','ca','thal']
X = df[feature_cols].values # Features
y = df.target.astype('category')# Target variable

nrm = preprocessing.Normalizer(norm='l1')
X = nrm.transform(X) # Normalized features

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = MLPClassifier(random_state=1, max_iter=4000)

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with all available normalized input features:",metrics.accuracy_score(y_test, y_pred))

#new model
feature_cols = ['exang', 'oldpeak']
X = df[feature_cols].values # Features
y = df.target.astype('category')# Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = MLPClassifier(random_state=1, max_iter=4000)

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with two input features:",metrics.accuracy_score(y_test, y_pred))

#new model
feature_cols = ['exang', 'oldpeak', 'ca']
X = df[feature_cols].values # Features
y = df.target.astype('category')# Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = MLPClassifier(random_state=1, max_iter=4000)

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with three input features:",metrics.accuracy_score(y_test, y_pred))

#new model
feature_cols = ['exang', 'oldpeak', 'ca', 'slope']
X = df[feature_cols].values # Features
y = df.target.astype('category')# Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = MLPClassifier(random_state=1, max_iter=4000)

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with four input features:",metrics.accuracy_score(y_test, y_pred))

#new model
feature_cols = ['exang', 'oldpeak', 'ca', 'slope', 'thal']
X = df[feature_cols].values # Features
y = df.target.astype('category')# Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = MLPClassifier(random_state=1, max_iter=4000)

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with five input features:",metrics.accuracy_score(y_test, y_pred))

#new model
feature_cols = ['exang', 'oldpeak', 'ca', 'slope', 'thal', 'sex']
X = df[feature_cols].values # Features
y = df.target.astype('category')# Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = MLPClassifier(random_state=1, max_iter=4000)

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with six input features:",metrics.accuracy_score(y_test, y_pred))

#new model
feature_cols = ['exang', 'oldpeak', 'ca', 'slope', 'thal', 'sex', 'age']
X = df[feature_cols].values # Features
y = df.target.astype('category')# Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = MLPClassifier(random_state=1, max_iter=4000)

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with seven input features:",metrics.accuracy_score(y_test, y_pred))

#new model
feature_cols = ['exang', 'oldpeak', 'ca']
X = df[feature_cols].values # Features
y = df.target.astype('category')# Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = MLPClassifier(random_state=1, activation = 'identity')

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with activation identity and three features:",metrics.accuracy_score(y_test, y_pred))


# Create Support vector machine classifer object
clf = MLPClassifier(random_state=1, activation = 'logistic')

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with activation logistic and three features:",metrics.accuracy_score(y_test, y_pred))

# Create Support vector machine classifer object
clf = MLPClassifier(random_state=1, activation = 'tanh')

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with activation tanh and three features:",metrics.accuracy_score(y_test, y_pred))

# Create Support vector machine classifer object
clf = MLPClassifier(random_state=1, activation = 'relu')

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with activation relu and three features:",metrics.accuracy_score(y_test, y_pred))

#new model
feature_cols = ['exang', 'oldpeak', 'ca', 'slope', 'thal', 'sex']
X = df[feature_cols].values # Features
y = df.target.astype('category')# Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = MLPClassifier(random_state=1, activation = 'identity')

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with activation identity and six featuress:",metrics.accuracy_score(y_test, y_pred))

# Create Support vector machine classifer object
clf = MLPClassifier(random_state=1, activation = 'logistic')

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with activation logistic and six featuress:",metrics.accuracy_score(y_test, y_pred))

# Create Support vector machine classifer object
clf = MLPClassifier(random_state=1, activation = 'tanh')

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with activation tanh and six featuress:",metrics.accuracy_score(y_test, y_pred))

# Create Support vector machine classifer object
clf = MLPClassifier(random_state=1, activation = 'relu')

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with activation relu and six featuress:",metrics.accuracy_score(y_test, y_pred))


# Neural Network Regression
#new model
feature_cols = ['age','sex','trestbps','chol','fbs','restecg','exang','oldpeak','slope','ca','thal']
X = df[feature_cols].values # Features
y = df.target# Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, activation = 'logistic')

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
feature_cols = ['age','sex','trestbps','chol','fbs','restecg','exang','oldpeak','slope','ca','thal']
X = df[feature_cols].values # Features
y = df.target# Target variable

nrm = preprocessing.Normalizer(norm='l2')
X = nrm.transform(X) # Normalized features

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, activation = 'logistic')

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with all available features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['exang', 'oldpeak']
X = df[feature_cols].values # Features
y = df.target# Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, activation = 'logistic')

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with two input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['exang', 'oldpeak', 'ca']
X = df[feature_cols].values # Features
y = df.target# Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, activation = 'logistic')

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with three input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['exang', 'oldpeak', 'ca', 'slope']
X = df[feature_cols].values # Features
y = df.target# Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, activation = 'logistic')

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with four input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['exang', 'oldpeak', 'ca', 'slope', 'thal']
X = df[feature_cols].values # Features
y = df.target# Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, activation = 'logistic')

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with five input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['exang', 'oldpeak', 'ca', 'slope', 'thal', 'sex']
X = df[feature_cols].values # Features
y = df.target# Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, activation = 'logistic')

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with six input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['exang', 'oldpeak', 'ca', 'slope', 'thal', 'sex', 'age']
X = df[feature_cols].values # Features
y = df.target# Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, activation = 'logistic')

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with seven input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = feature_cols = ['exang', 'oldpeak', 'ca', 'slope', 'thal', 'sex']
X = df[feature_cols].values # Features
y = df.target# Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, activation = 'logistic', solver='lbfgs', alpha=0.1)
# Train model
rgs = rgs.fit(X_train,y_train)
# Predict the response for test dataset
y_pred = rgs.predict(X_test)
# Model Accuracy
print("Accuracy with solver='lbfgs' in R^2:",rgs.score(X_test, y_test))

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, activation = 'logistic', solver='sgd', alpha=0.1)
# Train model
rgs = rgs.fit(X_train,y_train)
# Predict the response for test dataset
y_pred = rgs.predict(X_test)
# Model Accuracy
print("Accuracy with solver='sgd' in R^2:",rgs.score(X_test, y_test))

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, activation = 'logistic', solver='adam', alpha=0.1)
# Train model
rgs = rgs.fit(X_train,y_train)
# Predict the response for test dataset
y_pred = rgs.predict(X_test)
# Model Accuracy
print("Accuracy with solver='adam' in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = feature_cols = ['exang', 'oldpeak', 'ca', 'slope', 'thal', 'sex']
X = df[feature_cols].values # Features
y = df.target# Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, activation = 'logistic', solver='lbfgs', alpha=0.1)
# Train model
rgs = rgs.fit(X_train,y_train)
# Predict the response for test dataset
y_pred = rgs.predict(X_test)
# Model Accuracy
print("Accuracy with six input features in R^2:",rgs.score(X_test, y_test))

plt.rcParams["figure.figsize"] = (7,7)
# Plot the results
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("actual")
plt.ylabel("predicted")
plt.title("neural network Regression")
plt.show()


# Linear Regression 
#new model
feature_cols = ['age','sex','trestbps','chol','fbs','restecg','exang','oldpeak','slope','ca','thal']
X = df[feature_cols].values # Features
y = df.target# Target variable

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

plt.rcParams["figure.figsize"] = (7,7)
# Plot the results
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("actual")
plt.ylabel("predicted")
plt.title("Linear Regression")
plt.show()

#new model
feature_cols = ['age','sex','trestbps','chol','fbs','restecg','exang','oldpeak','slope','ca','thal']
X = df[feature_cols].values # Features
y = df.target# Target variable

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
feature_cols = ['age','sex','trestbps','chol','fbs','restecg','exang','oldpeak','slope','ca','thal']
X = df[feature_cols].values # Features
y = df.target# Target variable


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
feature_cols = ['exang', 'oldpeak']
X = df[feature_cols].values # Features
y = df.target# Target variable


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = linear_model.LinearRegression(fit_intercept=True, normalize=True)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with four input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['exang', 'oldpeak', 'ca']
X = df[feature_cols].values # Features
y = df.target# Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = linear_model.LinearRegression(fit_intercept=True, normalize=True)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with three input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['exang', 'oldpeak', 'ca', 'slope']
X = df[feature_cols].values # Features
y = df.target# Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = linear_model.LinearRegression(fit_intercept=True, normalize=True)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with four input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['exang', 'oldpeak', 'ca', 'slope', 'thal']
X = df[feature_cols].values # Features
y = df.target# Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = linear_model.LinearRegression(fit_intercept=True, normalize=True)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with five input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['exang', 'oldpeak', 'ca', 'slope', 'thal', 'sex']
X = df[feature_cols].values # Features
y = df.target# Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = linear_model.LinearRegression(fit_intercept=True, normalize=True)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with six input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['exang', 'oldpeak', 'ca', 'slope', 'thal', 'sex', 'age']
X = df[feature_cols].values # Features
y = df.target# Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = linear_model.LinearRegression(fit_intercept=True, normalize=True)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with seven input features in R^2:",rgs.score(X_test, y_test))

