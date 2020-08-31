# Motor Temperature Dataset

# run the dependencies
get_ipython().run_line_magic('run', 'Dependencies')

#Read and preview data
df = pd.read_csv(r"C:\Users\Alexandra.vanderMost\Afstuderen\Data\pmsm_temperature_data.csv")
print(df.shape)
print(df.dtypes)
df.head().round(decimals=3)

# Change plot size
plt.rcParams["figure.figsize"] = (20,10)

# Plot
df.hist()

# statistics
df.describe().round(decimals=2)
df.isnull().sum()

#measure column correlations
df.corr(method = 'pearson').round(decimals=3)

#create bins
bins = [-np.inf,-1.8, -1.3, -1, -0.8, -0.5, -0.2, 0, 0.2, 0.4, 0.7, 1, 1.3, 1.8, np.inf]
names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
df['pm_class'] = pd.cut(df['pm'], bins, labels=names)

# Change plot size
plt.rcParams["figure.figsize"] = (10,5)

# Plot
df['pm_class'].value_counts().sort_index().plot(kind='bar')

# Feature importance for regression problem
feature_cols = ['ambient','coolant','u_d','u_q','motor_speed','torque','i_d','i_q','stator_yoke','stator_tooth','stator_winding','profile_id']
X = df[feature_cols].values # Features
y = df.pm # Target variable

# feature extraction
skb = SelectKBest(score_func=f_regression)
skb = skb.fit(X, y)

# List Scores
score = np.array([feature_cols, skb.scores_]).transpose()
scr = pd.DataFrame(data=score, columns=['Feature', 'Score'])
print(scr)
plt.rcParams["figure.figsize"] = (15,8)
# Plot scores by feature
plt.bar(feature_cols, skb.scores_)
plt.xlabel('Feature')
plt.ylabel('Score')
plt.title('Feature Univariate linear regression test Score')

plt.show()

# Feature importance for classification problem
feature_cols = ['ambient','coolant','u_d','u_q','motor_speed','torque','i_d','i_q','stator_yoke','stator_tooth','stator_winding','profile_id']
X = df[feature_cols].values # Features
y = df.pm_class # Target variable

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

#decrease datasize
df = df.sample(frac = 0.1, random_state = 1)
df.shape

# Plot
df['pm_class'].value_counts().sort_index().plot(kind='bar')


# Decision Tree Classification
#new model
feature_cols = ['ambient','coolant','u_d','u_q','motor_speed','torque','i_d','i_q','stator_yoke','stator_tooth','stator_winding','profile_id']
X = df[feature_cols].values # Features
y = df.pm_class # Target variable

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

plt.rcParams["figure.figsize"] = (15,8)
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
plt.rcParams["figure.figsize"] = (10,5)
#plot the RDE
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

#new model
feature_cols = ['stator_yoke','profile_id','ambient', 'coolant']
X = df[feature_cols].values # Features
y = df.pm_class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(random_state=1)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy using four features:",metrics.accuracy_score(y_test, y_pred))

#new model
feature_cols = ['stator_yoke','profile_id','ambient', 'coolant','u_q']
X = df[feature_cols].values # Features
y = df.pm_class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(random_state=1)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy using three features:",metrics.accuracy_score(y_test, y_pred))

#new model
feature_cols = ['stator_yoke','profile_id','ambient', 'coolant','u_q']
X = df[feature_cols].values # Features
y = df.pm_class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", random_state=1)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy after optimization:",metrics.accuracy_score(y_test, y_pred))

clf.get_depth()

#new model
feature_cols = ['stator_tooth','profile_id','ambient', 'coolant','u_q']
X = df[feature_cols].values # Features
y = df.pm_class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth = 20, random_state=1)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy after optimization:",metrics.accuracy_score(y_test, y_pred))


# Decision Tree Regression
#new model
feature_cols = ['ambient','coolant','u_d','u_q','motor_speed','torque','i_d','i_q','stator_yoke','stator_tooth','stator_winding','profile_id']
X = df[feature_cols].values # Features
y = df.pm # Target variable

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

# Get models feature importance
importance = np.array([feature_cols, rgs.feature_importances_]).transpose()
imp = pd.DataFrame(data=importance, columns=['Feature', 'Importance'])

plt.rcParams["figure.figsize"] = (15,8)
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
plt.rcParams["figure.figsize"] = (10,5)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

#new model
feature_cols = ['stator_tooth','ambient','coolant']
X = df[feature_cols].values # Features
y = df.pm # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree regression object
rgs = DecisionTreeRegressor(random_state=1)

# Train Decision Tree Classifer
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with three features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['stator_tooth','ambient','coolant','profile_id']
X = df[feature_cols].values # Features
y = df.pm # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Train Decision Tree Classifer
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with four features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['stator_tooth','ambient','coolant','profile_id','motor_speed']
X = df[feature_cols].values # Features
y = df.pm # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Train Decision Tree Classifer
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with five features in R^2:",rgs.score(X_test, y_test))

print
(rgs.get_depth())

#new model
feature_cols = ['stator_tooth','ambient','coolant','profile_id','motor_speed']
X = df[feature_cols].values # Features
y = df.pm # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree regression object
rgs = DecisionTreeRegressor(random_state=1, max_depth=18)

# Train Decision Tree Classifer
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy 
print("Accuracy after optimization in R^2:",rgs.score(X_test, y_test))


# Support vector machine classification
#new model
feature_cols = ['ambient','coolant','u_d','u_q','motor_speed','torque','i_d','i_q','stator_yoke','stator_tooth','stator_winding','profile_id']
X = df[feature_cols].values # Features
y = df.pm_class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = svm.SVC(random_state=1, max_iter = 4000)

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with all available features:",metrics.accuracy_score(y_test, y_pred))

#new model
feature_cols = ['stator_tooth','ambient']
X = df[feature_cols].values # Features
y = df.pm_class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = svm.SVC(random_state=1)

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with two input features:",metrics.accuracy_score(y_test, y_pred))

#new model
feature_cols = ['stator_tooth','ambient', 'coolant']
X = df[feature_cols].values # Features
y = df.pm_class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = svm.SVC(random_state=1)

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with three input features:",metrics.accuracy_score(y_test, y_pred))

#new model
feature_cols = ['stator_tooth','ambient', 'coolant', 'motor_speed']
X = df[feature_cols].values # Features
y = df.pm_class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = svm.SVC(random_state=1)

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with four input features:",metrics.accuracy_score(y_test, y_pred))

#new model
feature_cols = ['stator_tooth','ambient', 'coolant', 'motor_speed', 'i_d']
X = df[feature_cols].values # Features
y = df.pm_class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = svm.SVC(random_state=1)

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with five input features:",metrics.accuracy_score(y_test, y_pred))

#new model
feature_cols = ['stator_tooth','ambient', 'coolant', 'motor_speed', 'i_d','profile_id']
X = df[feature_cols].values # Features
y = df.pm_class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = svm.SVC(random_state=1)

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with six input features:",metrics.accuracy_score(y_test, y_pred))

#new model
feature_cols = ['stator_tooth','ambient', 'coolant', 'motor_speed', 'i_d', 'profile_id']
X = df[feature_cols].values # Features
y = df.pm_class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object with kernel rbf
clf = svm.SVC(random_state=1, kernel='rbf', gamma='scale', max_iter = 4000)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with rbf kernel:",metrics.accuracy_score(y_test, y_pred))

# Create Support vector machine classifer object with kernel linear
clf = svm.SVC(random_state=1, kernel='linear', max_iter = 4000)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with linear kernel:",metrics.accuracy_score(y_test, y_pred))

# Create Support vector machine classifer object with kernel poly
clf = svm.SVC(random_state=1, kernel='poly', degree=2, gamma='auto', max_iter = 4000)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with poly kernel:",metrics.accuracy_score(y_test, y_pred))

#new model
feature_cols = ['stator_tooth','ambient', 'coolant', 'motor_speed', 'i_d','profile_id', 'u_d']
X = df[feature_cols].values # Features
y = df.pm_class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine classifer object
clf = svm.SVC(random_state=1, max_iter = 4000)

# Train  Support vector machine Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with seven input features:",metrics.accuracy_score(y_test, y_pred))


# Support vector machine regression
#new model
feature_cols = ['ambient','coolant','u_d','u_q','motor_speed','torque','i_d','i_q','stator_yoke','stator_tooth','stator_winding','profile_id']
X = df[feature_cols].values # Features
y = df.pm # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = svm.SVR(kernel='rbf', gamma='auto',max_iter = 4000)

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

#new model
feature_cols = ['ambient','coolant','u_d','u_q','motor_speed','torque','i_d','i_q','stator_yoke','stator_tooth','stator_winding','profile_id']
X = df[feature_cols].values # Features
y = df.pm # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = svm.SVR(kernel='linear', max_iter = 4000)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with all available features and linear kernel in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['ambient','coolant','u_d','u_q','motor_speed','torque','i_d','i_q','stator_yoke','stator_tooth','stator_winding','profile_id']
X = df[feature_cols].values # Features
y = df.pm # Target variable

nrm = preprocessing.Normalizer(norm='l2')
X = nrm.transform(X) # Normalized features

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = svm.SVR(kernel='rbf', gamma='auto',max_iter = 4000)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with all available features and linear kernel in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['stator_tooth','ambient', 'coolant', 'motor_speed']
X = df[feature_cols].values # Features
y = df.pm # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = svm.SVR(kernel='rbf', gamma='auto',max_iter = 4000)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with four input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['stator_tooth','ambient', 'coolant', 'motor_speed', 'i_d']
X = df[feature_cols].values # Features
y = df.pm # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = svm.SVR(kernel='rbf', gamma='auto',max_iter = 4000)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with five input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['stator_tooth','ambient', 'coolant', 'motor_speed', 'i_d','profile_id']
X = df[feature_cols].values # Features
y = df.pm # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = svm.SVR(kernel='rbf', gamma='auto',max_iter = 4000)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with six input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['stator_tooth','ambient', 'coolant', 'motor_speed', 'i_d','profile_id']
X = df[feature_cols].values # Features
y = df.pm # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = svm.SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.5, max_iter = 4000)
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with kernel rbf and additional parameters specified in R^2:",rgs.score(X_test, y_test))

# Create Support vector machine regression object
rgs = svm.SVR(kernel='poly', C=10, gamma='auto', degree=2, epsilon=0.5, coef0=1, max_iter = 4000)
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with kernel ploy of second degree in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['stator_tooth','ambient', 'coolant', 'motor_speed', 'i_d','profile_id']
X = df[feature_cols].values # Features
y = df.pm # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs_poly = svm.SVR(kernel='rbf', C=30, gamma='auto', epsilon=0.5, coef0=1)

# Train model
rgs_poly = rgs_poly.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs_poly.predict(X_test)
# Model Accuracy
print("Accuracy in R^2",rgs_poly.score(X_test, y_test))

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
feature_cols = ['ambient','coolant','u_d','u_q','motor_speed','torque','i_d','i_q','stator_yoke','stator_tooth','stator_winding','profile_id']
X = df[feature_cols].values # Features
y = df.pm_class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create neural network classifer object
clf = MLPClassifier(random_state=1, max_iter=4000)

# Train  neural network Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy with all available input features:",metrics.accuracy_score(y_test, y_pred))

#new model
#Normalized input features
feature_cols = ['ambient','coolant','u_d','u_q','motor_speed','torque','i_d','i_q','stator_yoke','stator_tooth','stator_winding','profile_id']
X = df[feature_cols].values # Features
y = df.pm_class # Target variable

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
feature_cols = ['stator_tooth','ambient']
X = df[feature_cols].values # Features
y = df.pm_class # Target variable

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
feature_cols = ['stator_tooth','ambient', 'coolant']
X = df[feature_cols].values # Features
y = df.pm_class # Target variable

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
feature_cols = ['stator_tooth','ambient', 'coolant', 'motor_speed']
X = df[feature_cols].values # Features
y = df.pm_class # Target variable

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
feature_cols = ['stator_tooth','ambient', 'coolant', 'motor_speed', 'i_d']
X = df[feature_cols].values # Features
y = df.pm_class # Target variable

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
feature_cols = ['stator_tooth','ambient', 'coolant', 'motor_speed', 'i_d','profile_id']
X = df[feature_cols].values # Features
y = df.pm_class # Target variable

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
feature_cols = ['stator_tooth','ambient', 'coolant', 'motor_speed', 'i_d','profile_id']
X = df[feature_cols].values # Features
y = df.pm_class # Target variable

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

#new model
feature_cols = ['stator_tooth','ambient', 'coolant', 'motor_speed', 'i_d','profile_id', 'u_d']
X = df[feature_cols].values # Features
y = df.pm_class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

clf = MLPClassifier(random_state=1, max_iter=4000, activation='logistic', solver='adam')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with logistic activation and solver='adam':",metrics.accuracy_score(y_test, y_pred))


# Neural Network Regression
#new model
feature_cols = ['ambient','coolant','u_d','u_q','motor_speed','torque','i_d','i_q','stator_yoke','stator_tooth','stator_winding','profile_id']
X = df[feature_cols].values # Features
y = df.pm # Target variable

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
feature_cols = ['ambient','coolant','u_d','u_q','motor_speed','torque','i_d','i_q','stator_yoke','stator_tooth','stator_winding','profile_id']
X = df[feature_cols].values # Features
y = df.pm # Target variable

nrm = preprocessing.Normalizer(norm='l2')
X = nrm.transform(X) # Normalized features

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, max_iter = 4000)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with all available features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['stator_tooth','ambient']
X = df[feature_cols].values # Features
y = df.pm # Target variable

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
feature_cols = ['stator_tooth','ambient', 'coolant']
X = df[feature_cols].values # Features
y = df.pm # Target variable

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
feature_cols = ['stator_tooth','ambient', 'coolant', 'motor_speed']
X = df[feature_cols].values # Features
y = df.pm # Target variable

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
feature_cols = ['stator_tooth','ambient', 'coolant', 'motor_speed', 'i_d']
X = df[feature_cols].values # Features
y = df.pm # Target variable

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
feature_cols = ['stator_tooth','ambient', 'coolant', 'motor_speed', 'i_d','profile_id']
X = df[feature_cols].values # Features
y = df.pm # Target variable

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
feature_cols = ['stator_tooth','ambient', 'coolant', 'motor_speed', 'i_d']
X = df[feature_cols].values # Features
y = df.pm # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, max_iter = 8000, activation='logistic')
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with five input features and logistic activation in R^2:",rgs.score(X_test, y_test))

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, max_iter = 8000, activation='relu', solver='lbfgs', alpha=0.1)
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with five input features and activation='relu' in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['stator_tooth','ambient', 'coolant', 'motor_speed', 'i_d','profile_id']
X = df[feature_cols].values # Features
y = df.pm # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, max_iter = 8000, activation='logistic')
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with six input features and logistic activation in R^2:",rgs.score(X_test, y_test))

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, max_iter = 8000, activation='relu', solver='lbfgs', alpha=0.1)
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with six input features and activation='relu' in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['stator_tooth','ambient', 'coolant', 'motor_speed', 'i_d']
X = df[feature_cols].values # Features
y = df.pm # Target variable

nrm = preprocessing.Normalizer(norm='l2')
X = nrm.transform(X) # Normalized features

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, max_iter = 8000, activation='relu', solver='lbfgs', alpha=0.1)
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with five normalized input features and activation='relu' in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['stator_tooth','ambient', 'coolant', 'motor_speed', 'i_d']
X = df[feature_cols].values # Features
y = df.pm # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Neural Network regression object
rgs = MLPRegressor(random_state = 1, max_iter = 8000, hidden_layer_sizes=(100,), activation='relu', solver='lbfgs', alpha=0.1)
rgs = rgs.fit(X_train,y_train)
y_pred = rgs.predict(X_test)
print("Accuracy with five input features and activation='relu' in R^2:",rgs.score(X_test, y_test))

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
feature_cols = ['ambient','coolant','u_d','u_q','motor_speed','torque','i_d','i_q','stator_yoke','stator_tooth','stator_winding','profile_id']
X = df[feature_cols].values # Features
y = df.pm # Target variable

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
feature_cols = ['ambient','coolant','u_d','u_q','motor_speed','torque','i_d','i_q','stator_yoke','stator_tooth','stator_winding','profile_id']
X = df[feature_cols].values # Features
y = df.pm # Target variable

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
feature_cols = ['ambient','coolant','u_d','u_q','motor_speed','torque','i_d','i_q','stator_yoke','stator_tooth','stator_winding','profile_id']
X = df[feature_cols].values # Features
y = df.pm # Target variable

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
feature_cols = ['stator_tooth','ambient','coolant', 'motor_speed', 'profile_id', 'i_d', 'torque','i_q','u_d','u_q']
X = df[feature_cols].values # Features
y = df.pm # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = linear_model.LinearRegression(fit_intercept=True)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with ten input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['stator_tooth','ambient','coolant', 'motor_speed', 'i_d', 'profile_id', 'torque']
X = df[feature_cols].values # Features
y = df.pm # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = linear_model.LinearRegression(fit_intercept=True)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with four input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['stator_tooth','ambient','coolant']
X = df[feature_cols].values # Features
y = df.pm # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = linear_model.LinearRegression(fit_intercept=True)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with three input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['stator_yoke','stator_tooth','stator_winding','ambient','coolant', 'motor_speed', 'profile_id', 'i_d','u_q','i_q','u_d', 'torque' ]
X = df[feature_cols].values # Features
y = df.pm # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = linear_model.LinearRegression(fit_intercept=True)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with twelve input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['stator_tooth','ambient','coolant', 'motor_speed', 'profile_id', 'i_d','u_q','i_q','u_d', 'torque' ]
X = df[feature_cols].values # Features
y = df.pm # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = linear_model.LinearRegression(fit_intercept=True)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with ten input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['stator_tooth','ambient','coolant', 'motor_speed', 'profile_id', 'i_d','u_q','i_q','u_d']
X = df[feature_cols].values # Features
y = df.pm # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = linear_model.LinearRegression(fit_intercept=True)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with nine input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['stator_tooth','ambient','coolant', 'motor_speed', 'profile_id', 'i_d','u_q','i_q']
X = df[feature_cols].values # Features
y = df.pm # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = linear_model.LinearRegression(fit_intercept=True)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with eight input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['stator_tooth','ambient','coolant', 'motor_speed', 'profile_id', 'i_d','u_q' ]
X = df[feature_cols].values # Features
y = df.pm # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = linear_model.LinearRegression(fit_intercept=True)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with seven input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['stator_tooth','ambient','coolant', 'motor_speed', 'profile_id', 'i_d' ]
X = df[feature_cols].values # Features
y = df.pm # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Support vector machine regression object
rgs = linear_model.LinearRegression(fit_intercept=True)

# Train model
rgs = rgs.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = rgs.predict(X_test)

# Model Accuracy
print("Accuracy with six input features in R^2:",rgs.score(X_test, y_test))

#new model
feature_cols = ['stator_tooth','ambient','coolant', 'motor_speed',  'i_d' ]
X = df[feature_cols].values # Features
y = df.pm # Target variable

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