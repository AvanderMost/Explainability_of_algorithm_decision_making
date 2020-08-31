# Boston Housing Dataset

# run the dependencies
get_ipython().run_line_magic('run', 'Dependencies')

#Read and preview data
df = pd.read_csv(r"C:\Users\Alexandra.vanderMost\Afstuderen\Data\HousingData.csv")
print(df.shape)
print(df.dtypes)
print(df.head())

# Change plot size
plt.rcParams["figure.figsize"] = (20,10)

# Plot df
df.hist()
# statistics
df.describe().round(decimals=2)

df.isnull().sum()

df_dropna = df.dropna()
print(df_dropna.shape)

#replace missing values with median
df_replacena = df.fillna(df.median())

#calc column correlations
df_replacena.corr(method = 'pearson').round(decimals=3)

#create classes
bins = [0, 17, 21, 25, np.inf]
names = ['<17', '17-21', '21-25', '25+']
df['MEDVc'] = pd.cut(df['MEDV'], bins, labels=names)
df_replacena = df.fillna(df.median())
df_dropna = df.dropna()

# Change plot size
plt.rcParams["figure.figsize"] = (10,5)
# Plot
df_replacena['MEDVc'].value_counts().sort_index().plot(kind='bar')


# Feature importance

# Feature importance for regression problem
feature_cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

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

# Feature importance for classification problem
feature_cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDVc # Target variable

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

feature_cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDVc # Target variable

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


#make new model
feature_cols = ['NOX','RM','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDVc # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy using three features:",metrics.accuracy_score(y_test, y_pred))


#make new model
feature_cols = ['NOX','RM','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDVc # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=1)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy after optimization:",metrics.accuracy_score(y_test, y_pred))

#visualize model
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1','2','3'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('BostonHousingDTC.png')
Image(graph.create_png())

#analyse the effect of using dropna ipv replace with median values
feature_cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
X2 = df_dropna[feature_cols].values # Features
y2 = df_dropna.MEDVc # Target variable

# Split dataset into training set and test set
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=2) # 70% training and 30% test

# Create Decision Tree classifer object
clf2 = DecisionTreeClassifier(random_state=1)

# Train Decision Tree Classifer
clf2 = clf2.fit(X2_train,y2_train)

#Predict the response for test dataset
y2_pred = clf2.predict(X2_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy using all available features:",metrics.accuracy_score(y2_test, y2_pred))

# Get models feature importance
importance = np.array([feature_cols, clf2.feature_importances_]).transpose()
imp = pd.DataFrame(data=importance, columns=['Feature', 'Importance'])

# Plot features importance
plt.bar(feature_cols, clf2.feature_importances_)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Models feature importance')
plt.show()

# feature extraction
skb = SelectKBest(score_func=f_classif)
skb = skb.fit(X2, y2)

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

# The "accuracy" scoring is proportional to the number of correct classifications
rfecv = RFECV(estimator=clf2, step=1, cv=StratifiedKFold(2), scoring='accuracy')
rfecv.fit(X2, y2)
print("Optimal number of features : %d" % rfecv.n_features_)

#plot the RDE
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

#feature and target selection
feature_cols = ['TAX','RM','LSTAT']
X2 = df_dropna[feature_cols].values # Features
y2 = df_dropna.MEDVc # Target variable

# Split dataset into training set and test set
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=2) # 70% training and 30% test

# Train Decision Tree Classifer
clf2 = clf2.fit(X2_train,y2_train)

#Predict the response for test dataset
y2_pred = clf2.predict(X2_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with three features:",metrics.accuracy_score(y2_test, y2_pred))

# Redefine the Decision Tree classifer object - optimized
clf2 = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=1)

# Train Decision Tree Classifer
clf2 = clf2.fit(X2_train,y2_train)

# Predict the response for test dataset
y2_pred = clf2.predict(X2_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy after optimization:",metrics.accuracy_score(y2_test, y2_pred))


# Decision Tree Regression
feature_cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

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

plt.rcParams["figure.figsize"] = (7,7)
# Plot the results
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("actual")
plt.ylabel("predicted")
plt.title("Decision Tree Regression")
plt.show()

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
feature_cols = ['INDUS','RM','PTRATIO','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Train Decision Tree Classifer
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with four features LSTAT, RM, PTRATIO and INDUS in R^2:",rgs.score(X_test, y_test))

#parallel modeling with feature DIS
feature_cols = ['DIS','RM','PTRATIO','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Train Decision Tree Classifer
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with four features LSTAT, RM, PTRATIO and DIS R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['INDUS','RM','PTRATIO','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree regression object
rgs = DecisionTreeRegressor(random_state=1, max_depth=4, criterion="mae")

# Train Decision Tree Classifer
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy 
print("Accuracy after optimization with LSTAT, RM, PTRATIO and INDUS R^2:",rgs.score(X_test, y_test))

#parallel modeling with feature DIS
feature_cols = ['DIS','RM','PTRATIO','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree regression object
rgs = DecisionTreeRegressor(random_state=1, max_depth=8, criterion="mae")

# Train Decision Tree Classifer
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy, --Not finished yet--
print("Accuracy after optimization with LSTAT, RM, PTRATIO and DIS R^2:",rgs.score(X_test, y_test))

#visualize the model
dot_data = StringIO()
export_graphviz(rgs, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1','2','3'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('BostonHousingDTR.png')
Image(graph.create_png())


# Support vector machine classification
feature_cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDVc # Target variable

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

#Normalize the input features
feature_cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
X = df_dropna[feature_cols].values # Features
y = df_dropna.MEDVc # Target variable

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

#new model
feature_cols = ['RM','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDVc # Target variable

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
feature_cols = ['NOX','RM','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDVc # Target variable

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
feature_cols = ['NOX','RM','TAX','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDVc # Target variable

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
feature_cols = ['INDUS','NOX','RM','TAX','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDVc # Target variable

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
feature_cols = ['INDUS','NOX','RM','AGE','TAX','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDVc # Target variable

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
feature_cols = ['INDUS','NOX','RM','TAX','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDVc # Target variable

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

#new model
feature_cols = ['INDUS','NOX','RM','TAX','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDVc # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = svm.LinearSVC(random_state=1, max_iter=999999999, tol=1e-05)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with linearSVM function:",metrics.accuracy_score(y_test, y_pred))


# Support vector machine regression
#new model
feature_cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = svm.SVR(kernel='rbf', gamma='auto')

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with all available features and rbf kernel in R^2:",rgs.score(X_test, y_test))

plt.rcParams["figure.figsize"] = (7,7)
# Plot the results
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("actual")
plt.ylabel("predicted")
plt.title("support vector machine Regression")
plt.show()

feature_cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

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

plt.rcParams["figure.figsize"] = (7,7)
# Plot the results
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("actual")
plt.ylabel("predicted")
plt.title("support vector machine Regression")
plt.show()

#new model with nomalized input features
feature_cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

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

plt.rcParams["figure.figsize"] = (7,7)
# Plot the results
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("actual")
plt.ylabel("predicted")
plt.title("support vector machine Regression")
plt.show()

#new model
feature_cols = ['RM','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

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
feature_cols = ['RM','PTRATIO','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

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
feature_cols = ['RM','PTRATIO','INDUS','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

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
feature_cols = ['RM','TAX','PTRATIO','LSTAT', 'INDUS']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

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
feature_cols = ['RM','TAX','PTRATIO','LSTAT', 'INDUS', 'NOX']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

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

#new models
feature_cols = ['RM','TAX','PTRATIO','LSTAT', 'INDUS']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = svm.SVR(kernel='rbf', C=10, gamma='auto', epsilon=0.5)
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with kernel rbf and additional parameters specified in R^2:",rgs.score(X_test, y_test))

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
rgs = svm.SVR(kernel='linear', C=10, tol=1e-5, epsilon=1.5)#0.7377038678579562
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with kernel linear and other parameters in R^2:",rgs.score(X_test, y_test))

# Create Support vector machine regression object
rgs = svm.SVR(kernel='linear', C=10, shrinking=False)
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with kernel linaer and shrinking off in R^2:",rgs.score(X_test, y_test))

# Create Support vector machine regression object
rgs = svm.SVR(kernel='poly', C=10, gamma='auto', degree=2, epsilon=0.5, coef0=1)
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with kernel ploy of second degree in R^2:",rgs.score(X_test, y_test))

# Create Support vector machine regression object
rgs = svm.SVR(kernel='poly', C=10, gamma='scale', degree=2, epsilon=0.5, coef0=1)
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with kernel ploy of second degree and gamma='scale' in R^2:",rgs.score(X_test, y_test))


#The input features
feature_cols = ['RM', 'LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs_poly = svm.SVR(kernel='poly', C=30, gamma='auto', degree=2, epsilon=0.5, coef0=1)

# Train model
rgs_poly = rgs_poly.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs_poly.predict(X_test)
# Model Accuracy
print("Accuracy with two features in R^2:",rgs_poly.score(X_test, y_test))


#The input features
feature_cols = ['RM','PTRATIO','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs_poly = svm.SVR(kernel='poly', C=30, gamma='auto', degree=2, epsilon=0.5, coef0=1)

# Train model
rgs_poly = rgs_poly.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs_poly.predict(X_test)
# Model Accuracy
print("Accuracy with three in R^2:",rgs_poly.score(X_test, y_test))


#The input features
feature_cols = ['RM','PTRATIO','INDUS','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs_poly = svm.SVR(kernel='poly', C=30, gamma='auto', degree=2, epsilon=0.5, coef0=1)

# Train model
rgs_poly = rgs_poly.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs_poly.predict(X_test)
# Model Accuracy
print("Accuracy with four features in R^2:",rgs_poly.score(X_test, y_test))


#The input features
feature_cols = ['RM','TAX','PTRATIO','LSTAT', 'INDUS']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs_poly = svm.SVR(kernel='poly', C=30, gamma='auto', degree=2, epsilon=0.5, coef0=1)

# Train model
rgs_poly = rgs_poly.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs_poly.predict(X_test)
# Model Accuracy
print("Accuracy with five features in R^2:",rgs_poly.score(X_test, y_test))


#The input features
feature_cols = ['RM','TAX','PTRATIO','LSTAT', 'INDUS', 'NOX']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

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
print("Accuracy with six features in R^2:",rgs_poly.score(X_test, y_test))

#The input features
feature_cols = ['RM','PTRATIO','INDUS','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

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


# Neural Network Classification
feature_cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDVc # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=400)

# Train  neural network Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with all available input features:",metrics.accuracy_score(y_test, y_pred))


#Normalized input features
feature_cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDVc # Target variable

nrm = preprocessing.Normalizer(norm='l2')
X = nrm.transform(X) # Normalized features

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=4000)

# Train  neural network Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with all available normalized input features:",metrics.accuracy_score(y_test, y_pred))


#new model
feature_cols = ['RM','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDVc # Target variable

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
feature_cols = ['RM','NOX','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDVc # Target variable

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
feature_cols = ['RM','TAX','NOX','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDVc # Target variable

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
feature_cols = ['NOX','RM','INDUS','TAX','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDVc # Target variable

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
feature_cols = ['NOX','RM','INDUS','TAX','LSTAT', 'AGE']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDVc # Target variable

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


#Normalized input features
feature_cols = ['RM','NOX','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDVc # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=4000, activation='relu', solver='adam')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with relu activation and solver='adam':",metrics.accuracy_score(y_test, y_pred))

# Create Neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=4000, activation='relu', solver='lbfgs')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with relu activation and solver='lbfgs':",metrics.accuracy_score(y_test, y_pred))

# Create Neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=4000, activation='relu', solver='sgd')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with relu activation and solver='sgd':",metrics.accuracy_score(y_test, y_pred))


# Create Neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=4000, activation='identity', solver='adam')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with identity activation and solver='adam':",metrics.accuracy_score(y_test, y_pred))

# Create Neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=40000, activation='identity', solver='lbfgs')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with identity activation and solver='lbfgs':",metrics.accuracy_score(y_test, y_pred))

# Create Neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=4000, activation='identity', solver='sgd')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with identity activation feature and solver='sgd':",metrics.accuracy_score(y_test, y_pred))


# Create Neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=4000, activation='logistic', solver='adam')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with logistic activation and solver='adam':",metrics.accuracy_score(y_test, y_pred))

# Create Neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=4000, activation='logistic', solver='lbfgs')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with logistic activation and solver='lbfgs':",metrics.accuracy_score(y_test, y_pred))

# Create Neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=4000, activation='logistic', solver='sgd')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with logistic activation and solver='sgd':",metrics.accuracy_score(y_test, y_pred))


# Create Neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=4000, activation='tanh', solver ='adam')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with tangus hyperbolicus activation andsolver='adam' features:",metrics.accuracy_score(y_test, y_pred))

# Create Neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=4000, activation='tanh', solver ='lbfgs')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with tangus hyperbolicus activation and solver='lbfgs':",metrics.accuracy_score(y_test, y_pred))

# Create Neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=4000, activation='tanh', solver ='sgd')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with tangus hyperbolicus activation and solver='sgd':",metrics.accuracy_score(y_test, y_pred))

nrm = preprocessing.Normalizer(norm='l2')
X = nrm.transform(X) # Normalized features

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=4000, activation='identity', solver='lbfgs')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with identity activation and normalized features:",metrics.accuracy_score(y_test, y_pred))

#new model
feature_cols = ['RM','NOX','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDVc # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=4000, hidden_layer_sizes=(120,80,80),  activation='identity', solver='lbfgs',batch_size=700, alpha=0.05)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with relu activation and normalized features:",metrics.accuracy_score(y_test, y_pred))
print("and the number of layers used:", clf.n_layers_)

# Create Neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=4000, hidden_layer_sizes=(),  activation='identity', solver='lbfgs',batch_size=200, alpha=0.001)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with relu activation and normalized features:",metrics.accuracy_score(y_test, y_pred))
print("and the number of layers used:", clf.n_layers_)


# Neural Network Regression

#new model
feature_cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, max_iter = 800)

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
feature_cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

nrm = preprocessing.Normalizer(norm='l2')
X = nrm.transform(X) # Normalized features

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, max_iter = 10000)

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
feature_cols = ['RM','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

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
feature_cols = ['RM','LSTAT','PTRATIO']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

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
feature_cols = ['RM','LSTAT','PTRATIO','INDUS']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

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
feature_cols = ['RM','LSTAT','PTRATIO', 'INDUS','TAX']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

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
feature_cols = ['RM','LSTAT','PTRATIO', 'INDUS','TAX', 'NOX']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

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
feature_cols = ['RM','LSTAT','PTRATIO','INDUS']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, max_iter = 8000, activation='logistic')
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with four input features and logistic activation in R^2:",rgs.score(X_test, y_test))

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, max_iter = 8000, activation='relu', solver='lbfgs', alpha=0.1)
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with four input features and activation='relu' in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['RM','LSTAT','PTRATIO']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, max_iter = 8000, activation='logistic')
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with three input features and logistic activation in R^2:",rgs.score(X_test, y_test))

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, max_iter = 8000, activation='relu', solver='lbfgs', alpha=0.1)
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with three input features and activation='relu' in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['RM','LSTAT','PTRATIO','INDUS']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

nrm = preprocessing.Normalizer(norm='l2')
X = nrm.transform(X) # Normalized features

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, max_iter = 8000, hidden_layer_sizes=(100,), activation='relu', solver='lbfgs', alpha=0.1,)
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with four normalized input features and activation='relu' in R^2:",rgs.score(X_test, y_test))


#new model
feature_cols = ['RM','LSTAT','PTRATIO','INDUS']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, max_iter = 8000, hidden_layer_sizes=(100,), activation='relu', solver='lbfgs', alpha=0.1)
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with four input features and activation='relu' in R^2:",rgs.score(X_test, y_test))

plt.rcParams["figure.figsize"] = (7,7)
# Plot the results
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("actual")
plt.ylabel("predicted")
plt.title("neural network Regression")
plt.show()


# Linear Regression 
feature_cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

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
feature_cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

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

plt.rcParams["figure.figsize"] = (7,7)
# Plot the results
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("actual")
plt.ylabel("predicted")
plt.title("Linear Regression")
plt.show()


#new model
feature_cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

nrm = preprocessing.Normalizer(norm='l2')
X = nrm.transform(X) # Normalized features

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
feature_cols = ['RM','LSTAT','PTRATIO','INDUS']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

nrm = preprocessing.Normalizer(norm='l2')
X = nrm.transform(X) # Normalized features

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

feature_cols = ['RM','LSTAT','PTRATIO']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable

nrm = preprocessing.Normalizer(norm='l2')
X = nrm.transform(X) # Normalized features

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

plt.rcParams["figure.figsize"] = (7,7)
# Plot the results
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("actual")
plt.ylabel("predicted")
plt.title("Linear Regression")
plt.show()

