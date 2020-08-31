# Experiment

# run the dependencies
get_ipython().run_line_magic('run', 'Dependencies')
# run the complexity measurements
get_ipython().run_line_magic('run', 'ComplexityMeasurements')

#create dataframe to save results
results = pd.DataFrame(columns=["Dataset", "Technique", "Model", "Accuracy", "NF", "IAS", "MEC"])

# Boston Housing models
Dataset = "Boston Housing"

#read data
df = pd.read_csv(r"C:\Users\Alexandra.vanderMost\Afstuderen\Data\HousingData.csv")
#replace null values
df_replacena = df.fillna(df.median())
#create classes
bins = [0, 17, 21, 25, np.inf]
names = ['1', '2', '3', '4']
df_replacena['MEDVc'] = pd.cut(df_replacena['MEDV'], bins, labels=names)

#Decision tree classification
print("start BH_DT_clf")
Technique = "DecisionTree"
Model = "Classification"
feature_cols = ['NOX','RM','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDVc # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
BH_DT_clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=1) # Create Decision Tree classifer object
BH_DT_clf = BH_DT_clf.fit(X_train,y_train)
y_pred = BH_DT_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Technique, Model, metrics.accuracy_score(y_test, y_pred), feature_used(BH_DT_clf, X, 100), interaction_strength_class(BH_DT_clf, X, ['1', '2', '3', '4'], 200), main_effect_complexity_class(BH_DT_clf, X, ['1', '2', '3', '4'], 200, 5, 0.05)]], columns=["Dataset", "Technique", "Model", "Accuracy", "NF", "IAS", "MEC"])])

#Decision tree regression
print("start BH_DT_rgs")
Model = "Regression"
feature_cols = ['DIS','RM','PTRATIO','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
BH_DT_rgs = DecisionTreeRegressor(random_state=1, max_depth=8, criterion="mae") # Create Decision Tree regression object
BH_DT_rgs = BH_DT_rgs.fit(X_train,y_train)
y_pred = BH_DT_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Technique, Model, BH_DT_rgs.score(X_test, y_test), feature_used(BH_DT_rgs, X, 100), interaction_strength(BH_DT_rgs, X, 200), main_effect_complexity(BH_DT_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Technique", "Model", "Accuracy", "NF", "IAS", "MEC"])])

#Support Vector Machine classification
print("start BH_SVM_clf")
Technique = "Support Vector Machine"
Model = "Classification"
feature_cols = ['INDUS','NOX','RM','TAX','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDVc # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
BH_SVM_clf = svm.SVC(random_state=1, kernel='linear')
BH_SVM_clf = BH_SVM_clf.fit(X_train,y_train)
y_pred = BH_SVM_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Technique, Model, metrics.accuracy_score(y_test, y_pred), feature_used(BH_SVM_clf, X, 100), interaction_strength_class(BH_SVM_clf, X, ['1', '2', '3', '4'], 200), main_effect_complexity_class(BH_SVM_clf, X, ['1', '2', '3', '4'], 200, 5, 0.05)]], columns=["Dataset", "Technique", "Model", "Accuracy", "NF", "IAS", "MEC"])])

#Support Vector Machine regression
print("start BH_SVM_rgs")
Model = "Regression"
feature_cols = ['RM','PTRATIO','INDUS','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
BH_SVM_rgs = svm.SVR(kernel='poly', C=30, gamma='auto', degree=2, epsilon=0.5, coef0=1)
BH_SVM_rgs = BH_SVM_rgs.fit(X_train,y_train)
y_pred = BH_SVM_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Technique, Model, BH_SVM_rgs.score(X_test, y_test), feature_used(BH_SVM_rgs, X, 100), interaction_strength(BH_SVM_rgs, X, 200), main_effect_complexity(BH_SVM_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Technique", "Model", "Accuracy", "NF", "IAS", "MEC"])])

#Neural network classification
print("start BH_NN_clf")
Technique = "Neural Network"
Model = "Classification"
feature_cols = ['RM','NOX','LSTAT']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDVc # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
BH_NN_clf = MLPClassifier(random_state=1, max_iter=4000, hidden_layer_sizes=(),  activation='identity', solver='lbfgs',batch_size=200, alpha=0.001)
BH_NN_clf = BH_NN_clf.fit(X_train,y_train)
y_pred = BH_NN_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Technique, Model, metrics.accuracy_score(y_test, y_pred), feature_used(BH_NN_clf, X, 100), interaction_strength_class(BH_NN_clf, X, ['1', '2', '3', '4'], 200), main_effect_complexity_class(BH_NN_clf, X, ['1', '2', '3', '4'], 200, 5, 0.05)]], columns=["Dataset", "Technique", "Model", "Accuracy", "NF", "IAS", "MEC"])])

#Neural network regression
print("start BH_NN_rgs")
Model = "Regression"
feature_cols = ['RM','LSTAT','PTRATIO','INDUS']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
BH_NN_rgs = MLPRegressor(random_state = 1, max_iter = 8000, activation='relu', solver='lbfgs', alpha=0.1)
BH_NN_rgs = BH_NN_rgs.fit(X_train,y_train)
y_pred = BH_NN_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Technique, Model, BH_NN_rgs.score(X_test, y_test), feature_used(BH_NN_rgs, X, 100), interaction_strength(BH_NN_rgs, X, 200), main_effect_complexity(BH_NN_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Technique", "Model", "Accuracy", "NF", "IAS", "MEC"])])

#Linear regression
print("start BH_LR_rgs")
Technique = "Linear regression"
Model = "Regression"
feature_cols = ['RM','LSTAT','PTRATIO']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable
nrm = preprocessing.Normalizer(norm='l2')
X = nrm.transform(X) # Normalized features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
BH_LR_rgs = linear_model.LinearRegression(fit_intercept=True, normalize=True)
BH_LR_rgs = BH_LR_rgs.fit(X_train,y_train)
y_pred = BH_LR_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Technique, Model, BH_LR_rgs.score(X_test, y_test), feature_used(BH_LR_rgs, X, 100), interaction_strength(BH_LR_rgs, X, 200), main_effect_complexity(BH_LR_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Technique", "Model", "Accuracy", "NF", "IAS", "MEC"])])

print(results)

# Creditcard fraud models
Dataset = "Creditcard Fraud"

#read data
df = pd.read_csv(r"C:\Users\Alexandra.vanderMost\Afstuderen\Data\creditcard.csv")

#Decision tree classification
print("start CCF_DT_clf")
Technique = "DecisionTree"
Model = "Classification"
feature_cols = ['V10','V14','V17']
X = df[feature_cols].values # Features
y = df.Class.astype('category') # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
CCF_DT_clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=1)
CCF_DT_clf = CCF_DT_clf.fit(X_train,y_train)
y_pred = CCF_DT_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Technique, Model, metrics.accuracy_score(y_test, y_pred), feature_used(CCF_DT_clf, X, 10000), interaction_strength_class(CCF_DT_clf, X, [0,1], 200, 0.01), main_effect_complexity_class(CCF_DT_clf, X, [0, 1], 200, 5, 0.05, 0.01)]], columns=["Dataset", "Technique", "Model", "Accuracy", "NF", "IAS", "MEC"])])

#Decision tree regression
print("start CCF_DT_rgs")
Model = "Regression"
feature_cols = ['V17','V10','V14', 'Time']
X = df[feature_cols].values # Features
y = df.Class # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
CCF_DT_rgs = DecisionTreeRegressor(random_state=1, max_depth=4)
CCF_DT_rgs = CCF_DT_rgs.fit(X_train,y_train)
y_pred = CCF_DT_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Technique, Model, CCF_DT_rgs.score(X_test, y_test), feature_used(CCF_DT_rgs, X, 10000), interaction_strength(CCF_DT_rgs, X, 200, 0.01), main_effect_complexity(CCF_DT_rgs, X, 200, 5, 0.05, 0.01)]], columns=["Dataset", "Technique", "Model", "Accuracy", "NF", "IAS", "MEC"])])

#Support Vector Machine classification
print("start CCF_SVM_clf")
Technique = "Support Vector Machine"
Model = "Classification"
feature_cols = ['V17', 'V14', 'V12', 'V10', 'V16']
X = df[feature_cols].values # Features
y = df.Class.astype('category') # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
CCF_SVM_clf = svm.SVC(random_state=1, kernel='rbf', gamma='auto')
CCF_SVM_clf = CCF_SVM_clf.fit(X_train,y_train)
y_pred = CCF_SVM_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Technique, Model, metrics.accuracy_score(y_test, y_pred), feature_used(CCF_SVM_clf, X, 10000), interaction_strength_class(CCF_SVM_clf, X, [0, 1], 200, 0.01), main_effect_complexity_class(CCF_SVM_clf, X, [0, 1], 200, 5, 0.05, 0.01)]], columns=["Dataset", "Technique", "Model", "Accuracy", "NF", "IAS", "MEC"])])

#Support Vector Machine regression
Model = "Regression"
results = pd.concat([results, pd.DataFrame([[Dataset, Technique, Model, 0, 0, 0, 0]], columns=["Dataset", "Technique", "Model", "Accuracy", "NF", "IAS", "MEC"])])

#Neural network classification
print("start CCF_NN_clf")
Technique = "Neural Network"
Model = "Classification"
feature_cols = ['V17', 'V14', 'V12', 'V10']
X = df[feature_cols].values # Features
y = df.Class.astype('category') # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
CCF_NN_clf = MLPClassifier(random_state=1, max_iter=4000, hidden_layer_sizes=(120,80,80),  activation='identity', solver='lbfgs',batch_size=700, alpha=0.05)
CCF_NN_clf = CCF_NN_clf.fit(X_train,y_train)
y_pred = CCF_NN_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Technique, Model, metrics.accuracy_score(y_test, y_pred), feature_used(CCF_NN_clf, X, 10000), interaction_strength_class(CCF_NN_clf, X, [0, 1], 200, 0.01), main_effect_complexity_class(CCF_NN_clf, X, [0, 1], 200, 5, 0.05, 0.01)]], columns=["Dataset", "Technique", "Model", "Accuracy", "NF", "IAS", "MEC"])])

#Neural network regression
print("start CCF_NN_rgs")
Model = "Regression"
feature_cols = ['V17', 'V14', 'V12', 'V10']
X = df[feature_cols].values # Features
y = df.Class # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
CCF_NN_rgs = MLPRegressor(random_state = 1, max_iter = 8000)
CCF_NN_rgs = CCF_NN_rgs.fit(X_train,y_train)
y_pred = CCF_NN_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Technique, Model, CCF_NN_rgs.score(X_test, y_test), feature_used(CCF_NN_rgs, X, 10000), interaction_strength(CCF_NN_rgs, X, 200, 0.01), main_effect_complexity(CCF_NN_rgs, X, 200, 5, 0.05, 0.01)]], columns=["Dataset", "Technique", "Model", "Accuracy", "NF", "IAS", "MEC"])])

#Linear regression
print("start CCF_LR_rgs")
Technique = "Linear regression"
Model = "Regression"
feature_cols = ['V17','V10', 'V12','V14','V16','V3','V7','V11','V4','V18','V1','V9','V5','V2','V6','V21','V19']
X = df[feature_cols].values # Features
y = df.Class # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
CCF_LR_rgs = linear_model.LinearRegression(fit_intercept=True)
CCF_LR_rgs = CCF_LR_rgs.fit(X_train,y_train)
y_pred = CCF_LR_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Technique, Model, CCF_LR_rgs.score(X_test, y_test), feature_used(CCF_LR_rgs, X, 10000), interaction_strength(CCF_LR_rgs, X, 200, 0.01), main_effect_complexity(CCF_LR_rgs, X, 200, 5, 0.05, 0.01)]], columns=["Dataset", "Technique", "Model", "Accuracy", "NF", "IAS", "MEC"])])

print(results)

# Heart Diseases models
Dataset = "Heart Diseases"

#read data
df = pd.read_csv(r"C:\Users\Alexandra.vanderMost\Afstuderen\Data\Heart.csv")

#Decision tree classification
print("start HD_DT_clf")
Technique = "DecisionTree"
Model = "Classification"
feature_cols = ['exang', 'age', 'ca', 'oldpeak', 'thal', 'trestbps', 'sex']
X = df[feature_cols].values # Features
y = df.target.astype('category') # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
HD_DT_clf = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=1)
HD_DT_clf = HD_DT_clf.fit(X_train,y_train)
y_pred = HD_DT_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Technique, Model, metrics.accuracy_score(y_test, y_pred), feature_used(HD_DT_clf, X, 100), interaction_strength_class(HD_DT_clf, X, [0,1], 200), main_effect_complexity_class(HD_DT_clf, X, [0,1], 200, 5, 0.05)]], columns=["Dataset", "Technique", "Model", "Accuracy", "NF", "IAS", "MEC"])])

#Decision tree regression
print("start HD_DT_rgs")
Model = "Regression"
feature_cols = ['exang', 'ca']
X = df[feature_cols].values # Features
y = df.target # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
HD_DT_rgs = DecisionTreeRegressor(criterion = 'mse',max_depth=4, random_state=1)
HD_DT_rgs = HD_DT_rgs.fit(X_train,y_train)
y_pred = HD_DT_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Technique, Model, HD_DT_rgs.score(X_test, y_test), feature_used(HD_DT_rgs, X, 100), interaction_strength(HD_DT_rgs, X, 200), main_effect_complexity(HD_DT_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Technique", "Model", "Accuracy", "NF", "IAS", "MEC"])])

#Support Vector Machine classification
print("start HD_SVM_clf")
Technique = "Support Vector Machine"
Model = "Classification"
feature_cols = ['exang', 'oldpeak', 'ca', 'slope', 'thal', 'sex', 'age']
X = df[feature_cols].values # Features
y = df.target.astype('category') # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
HD_SVM_clf = svm.SVC(random_state=1, kernel='linear')
HD_SVM_clf = HD_SVM_clf.fit(X_train,y_train)
y_pred = HD_SVM_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Technique, Model, metrics.accuracy_score(y_test, y_pred), feature_used(HD_SVM_clf, X, 100), interaction_strength_class(HD_SVM_clf, X, [0,1], 200), main_effect_complexity_class(HD_SVM_clf, X, [0,1], 200, 5, 0.05)]], columns=["Dataset", "Technique", "Model", "Accuracy", "NF", "IAS", "MEC"])])

#Support Vector Machine regression
print("start HD_SVM_rgs")
Model = "Regression"
feature_cols = ['exang', 'oldpeak', 'ca', 'slope', 'thal', 'sex', 'age']
X = df[feature_cols].values # Features
y = df.target # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
HD_SVM_rgs = svm.SVR(kernel='poly', gamma='auto', degree=2, epsilon=0.05, coef0=1)
HD_SVM_rgs = HD_SVM_rgs.fit(X_train,y_train)
y_pred = HD_SVM_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Technique, Model, HD_SVM_rgs.score(X_test, y_test), feature_used(HD_SVM_rgs, X, 100), interaction_strength(HD_SVM_rgs, X, 200), main_effect_complexity(HD_SVM_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Technique", "Model", "Accuracy", "NF", "IAS", "MEC"])])

#Neural network classification
print("start HD_NN_clf")
Technique = "Neural Network"
Model = "Classification"
feature_cols = ['exang', 'oldpeak', 'ca', 'slope', 'thal', 'sex']
X = df[feature_cols].values # Features
y = df.target.astype('category')# Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
HD_NN_clf = MLPClassifier(random_state=1, activation = 'relu')
HD_NN_clf = HD_NN_clf.fit(X_train,y_train)
y_pred = HD_NN_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Technique, Model, metrics.accuracy_score(y_test, y_pred), feature_used(HD_NN_clf, X, 100), interaction_strength_class(HD_NN_clf, X, [0,1], 200), main_effect_complexity_class(HD_NN_clf, X, [0,1], 200, 5, 0.05)]], columns=["Dataset", "Technique", "Model", "Accuracy", "NF", "IAS", "MEC"])])

#Neural network regression
print("start HD_NN_rgs")
Model = "Regression"
feature_cols = feature_cols = ['exang', 'oldpeak', 'ca', 'slope', 'thal', 'sex']
X = df[feature_cols].values # Features
y = df.target# Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
HD_NN_rgs = MLPRegressor(random_state = 1, activation = 'logistic', solver='lbfgs', alpha=0.1)
HD_NN_rgs = HD_NN_rgs.fit(X_train,y_train)
y_pred = HD_NN_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Technique, Model, HD_NN_rgs.score(X_test, y_test), feature_used(HD_NN_rgs, X, 100), interaction_strength(HD_NN_rgs, X, 200), main_effect_complexity(HD_NN_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Technique", "Model", "Accuracy", "NF", "IAS", "MEC"])])

#Linear regression
print("start HD_LR_rgs")
Technique = "Linear regression"
Model = "Regression"
feature_cols = ['exang', 'oldpeak', 'ca', 'slope', 'thal', 'sex']
X = df[feature_cols].values # Features
y = df.target# Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
HD_LR_rgs = linear_model.LinearRegression(fit_intercept=True, normalize=True)
HD_LR_rgs = HD_LR_rgs.fit(X_train,y_train)
y_pred = HD_LR_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Technique, Model, HD_LR_rgs.score(X_test, y_test), feature_used(HD_LR_rgs, X, 100), interaction_strength(HD_LR_rgs, X, 200), main_effect_complexity(HD_LR_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Technique", "Model", "Accuracy", "NF", "IAS", "MEC"])])

print(results)

# Motor Temperature models
Dataset = "Motor Temperature"

#read data
df = pd.read_csv(r"C:\Users\Alexandra.vanderMost\Afstuderen\Data\pmsm_temperature_data.csv")
#create classes
bins = [-np.inf,-1.8, -1.3, -1, -0.8, -0.5, -0.2, 0, 0.2, 0.4, 0.7, 1, 1.3, 1.8, np.inf]
names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
df['pm_class'] = pd.cut(df['pm'], bins, labels=names)
#decrease dataset size 
df = df.sample(frac = 0.1, random_state = 1)

#Decision tree classification
print("start MT_DT_clf")
Technique = "DecisionTree"
Model = "Classification"
feature_cols = ['stator_tooth','profile_id','ambient', 'coolant','u_q']
X = df[feature_cols].values # Features
y = df.pm_class # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
MT_DT_clf = DecisionTreeClassifier(criterion="entropy", max_depth = 20, random_state=1)
MT_DT_clf = MT_DT_clf.fit(X_train,y_train)
y_pred = MT_DT_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Technique, Model, metrics.accuracy_score(y_test, y_pred), feature_used(MT_DT_clf, X, 100), interaction_strength_class(MT_DT_clf, X, ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'], 200, 0.01), main_effect_complexity_class(MT_DT_clf, X, ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'], 200, 5, 0.05, 0.01)]], columns=["Dataset", "Technique", "Model", "Accuracy", "NF", "IAS", "MEC"])])

#Decision tree regression
print("start MT_DT_rgs")
Model = "Regression"
feature_cols = ['stator_tooth','ambient','coolant','profile_id','motor_speed']
X = df[feature_cols].values # Features
y = df.pm # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
MT_DT_rgs = DecisionTreeRegressor(random_state=1, max_depth=18)
MT_DT_rgs = MT_DT_rgs.fit(X_train,y_train)
y_pred = MT_DT_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Technique, Model, MT_DT_rgs.score(X_test, y_test), feature_used(MT_DT_rgs, X, 100), interaction_strength(MT_DT_rgs, X, 200, 0.01), main_effect_complexity(MT_DT_rgs, X, 200, 5, 0.05, 0.01)]], columns=["Dataset", "Technique", "Model", "Accuracy", "NF", "IAS", "MEC"])])

#Support Vector Machine classification
print("start MT_SVM_clf")
Technique = "Support Vector Machine"
Model = "Classification"
feature_cols = ['stator_tooth','ambient', 'coolant', 'motor_speed', 'i_d','profile_id']
X = df[feature_cols].values # Features
y = df.pm_class # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
MT_SVM_clf = svm.SVC(random_state=1)
MT_SVM_clf = MT_SVM_clf.fit(X_train,y_train)
y_pred = MT_SVM_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Technique, Model, metrics.accuracy_score(y_test, y_pred), feature_used(MT_SVM_clf, X, 100), interaction_strength_class(MT_SVM_clf, X, ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'], 200, 0.01), main_effect_complexity_class(MT_SVM_clf, X, ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'], 200, 5, 0.05, 0.01)]], columns=["Dataset", "Technique", "Model", "Accuracy", "NF", "IAS", "MEC"])])

#Support Vector Machine regression
print("start MT_SVM_rgs")
Model = "Regression"
feature_cols = ['stator_tooth','ambient', 'coolant', 'motor_speed', 'i_d','profile_id']
X = df[feature_cols].values # Features
y = df.pm # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
MT_SVM_rgs = svm.SVR(kernel='rbf', C=30, gamma='auto', epsilon=0.5, coef0=1)
MT_SVM_rgs = MT_SVM_rgs.fit(X_train,y_train)
y_pred = MT_SVM_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Technique, Model, MT_SVM_rgs.score(X_test, y_test), feature_used(MT_SVM_rgs, X, 100), interaction_strength(MT_SVM_rgs, X, 200, 0.01), main_effect_complexity(MT_SVM_rgs, X, 200, 5, 0.05, 0.01)]], columns=["Dataset", "Technique", "Model", "Accuracy", "NF", "IAS", "MEC"])])

#Neural network classification
print("start MT_NN_clf")
Technique = "Neural Network"
Model = "Classification"
feature_cols = ['stator_tooth','ambient', 'coolant', 'motor_speed', 'i_d','profile_id', 'u_d']
X = df[feature_cols].values # Features
y = df.pm_class # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
MT_NN_clf = MLPClassifier(random_state=1, max_iter=4000, activation='logistic', solver='adam')
MT_NN_clf = MT_NN_clf.fit(X_train,y_train)
y_pred = MT_NN_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Technique, Model, metrics.accuracy_score(y_test, y_pred), feature_used(MT_NN_clf, X, 100), interaction_strength_class(MT_NN_clf, X, ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'], 200, 0.01), main_effect_complexity_class(MT_NN_clf, X, ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'], 200, 5, 0.05, 0.01)]], columns=["Dataset", "Technique", "Model", "Accuracy", "NF", "IAS", "MEC"])])

#Neural network regression
print("start MT_NN_rgs")
Model = "Regression"
feature_cols = ['stator_tooth','ambient', 'coolant', 'motor_speed', 'i_d']
X = df[feature_cols].values # Features
y = df.pm # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
MT_NN_rgs = MLPRegressor(random_state = 1, max_iter = 8000, hidden_layer_sizes=(100,), activation='relu', solver='lbfgs', alpha=0.1)
MT_NN_rgs = MT_NN_rgs.fit(X_train,y_train)
y_pred = MT_NN_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Technique, Model, MT_NN_rgs.score(X_test, y_test), feature_used(MT_NN_rgs, X, 100), interaction_strength(MT_NN_rgs, X, 200, 0.01), main_effect_complexity(MT_NN_rgs, X, 200, 5, 0.05, 0.01)]], columns=["Dataset", "Technique", "Model", "Accuracy", "NF", "IAS", "MEC"])])

#Linear regression
print("start MT_LR_rgs")
Technique = "Linear regression"
Model = "Regression"
feature_cols = ['stator_tooth','ambient','coolant', 'motor_speed', 'profile_id', 'i_d','u_q' ]
X = df[feature_cols].values # Features
y = df.pm # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
MT_LR_rgs = linear_model.LinearRegression(fit_intercept=True)
MT_LR_rgs = MT_LR_rgs.fit(X_train,y_train)
y_pred = MT_LR_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Technique, Model, MT_LR_rgs.score(X_test, y_test), feature_used(MT_LR_rgs, X, 100), interaction_strength(MT_LR_rgs, X, 200, 0.01), main_effect_complexity(MT_LR_rgs, X, 200, 5, 0.05, 0.01)]], columns=["Dataset", "Technique", "Model", "Accuracy", "NF", "IAS", "MEC"])])

print(results)

#reset index
results = resulta.reset_index()
results = results.drop(columns = ['index'])  
print(results)


df = results.copy()

df_clf = df[df.Model == "Classification"].pivot(index='Dataset', columns='Technique', values='Accuracy')
df_rgs = df[df.Model == "Regression"].pivot(index='Dataset', columns='Technique', values='Accuracy')
df_clf.plot(kind = 'bar', rot = 90, sort_columns = True, figsize = (10,8), color = list(['tab:blue','tab:orange','tab:green']))
df_rgs.plot(kind = 'bar', rot = 90, sort_columns = True, figsize = (10,8), color = list(['tab:blue','tab:red','tab:orange','tab:green']))

df1 = df.sort_values(['Dataset','Technique' ,'Model'])
fig, axs = plt.subplots(4, 2, figsize=(10,16), sharex=False)
axs[0, 0].plot(['DT', 'LR', 'NN', 'SVM'], df1[(df1.Model == "Regression") & (df1.Dataset == "Boston Housing")].values[:,5])
axs[0, 0].set(title = 'Boston Housing', ylabel='IAS')
axs[1, 0].plot(['DT', 'LR', 'NN'], df1[(df1.Model == "Regression") & (df1.Dataset == "Creditcard Fraud")].values[:,5])
axs[1, 0].set(title = 'Creditcard Fraud', ylabel='IAS')
axs[2, 0].plot(['DT', 'LR', 'NN', 'SVM'], df1[(df1.Model == "Regression") & (df1.Dataset == "Heart Diseases")].values[:,5])
axs[2, 0].set(title = 'Heart Diseases', ylabel='IAS')
axs[3, 0].plot(['DT', 'LR', 'NN', 'SVM'], df1[(df1.Model == "Regression") & (df1.Dataset == "Motor Temperature")].values[:,5])
axs[3, 0].set(title='Motor Temperature', ylabel='IAS')
axs[0, 1].plot(['DT', 'LR', 'NN', 'SVM'], df1[(df1.Model == "Regression") & (df1.Dataset == "Boston Housing")].values[:,6])
axs[0, 1].set(title='Boston Housing', ylabel='MEC')
axs[1, 1].plot(['DT', 'LR', 'NN'], df1[(df1.Model == "Regression") & (df1.Dataset == "Creditcard Fraud")].values[:,6])
axs[1, 1].set(title ='Creditcard Fraud', ylabel='MEC')
axs[2, 1].plot(['DT', 'LR', 'NN', 'SVM'], df1[(df1.Model == "Regression") & (df1.Dataset == "Heart Diseases")].values[:,6])
axs[2, 1].set(title='Heart Diseases', ylabel='MEC')
axs[3, 1].plot(['DT', 'LR', 'NN', 'SVM'], df1[(df1.Model == "Regression") & (df1.Dataset == "Motor Temperature")].values[:,6])
axs[3, 1].set(title='Motor Temperature', ylabel='MEC')

df1 = df.sort_values(['Dataset','Technique' ,'Model'])
fig, axs = plt.subplots(4, 2, figsize=(10,16), sharex=False)
axs[0, 0].plot(['DT', 'NN', 'SVM'], df1[(df1.Model == "Classification") & (df1.Dataset == "Boston Housing")].values[:,5])
axs[0, 0].set(title = 'Boston Housing', ylabel='IAS')
axs[1, 0].plot(['DT', 'NN', 'SVM'], df1[(df1.Model == "Classification") & (df1.Dataset == "Creditcard Fraud")].values[:,5])
axs[1, 0].set(title = 'Creditcard Fraud', ylabel='IAS')
axs[2, 0].plot(['DT', 'NN', 'SVM'], df1[(df1.Model == "Classification") & (df1.Dataset == "Heart Diseases")].values[:,5])
axs[2, 0].set(title = 'Heart Diseases', ylabel='IAS')
axs[3, 0].plot(['DT', 'NN', 'SVM'], df1[(df1.Model == "Classification") & (df1.Dataset == "Motor Temperature")].values[:,5])
axs[3, 0].set(title='Motor Temperature', ylabel='IAS')
axs[0, 1].plot(['DT', 'NN', 'SVM'], df1[(df1.Model == "Classification") & (df1.Dataset == "Boston Housing")].values[:,6])
axs[0, 1].set(title='Boston Housing', ylabel='MEC')
axs[1, 1].plot(['DT', 'NN', 'SVM'], df1[(df1.Model == "Classification") & (df1.Dataset == "Creditcard Fraud")].values[:,6])
axs[1, 1].set(title ='Creditcard Fraud', ylabel='MEC')
axs[2, 1].plot(['DT', 'NN', 'SVM'], df1[(df1.Model == "Classification") & (df1.Dataset == "Heart Diseases")].values[:,6])
axs[2, 1].set(title='Heart Diseases', ylabel='MEC')
axs[3, 1].plot(['DT', 'NN', 'SVM'], df1[(df1.Model == "Classification") & (df1.Dataset == "Motor Temperature")].values[:,6])
axs[3, 1].set(title='Motor Temperature', ylabel='MEC')

df1 = df.sort_values(['Dataset','Technique' ,'Model'])
fig, axs = plt.subplots(4, 2, figsize=(10,16), sharex=False)
axs[0, 0].plot(['DT', 'NN', 'SVM'], df1[(df1.Model == "Classification") & (df1.Dataset == "Boston Housing")].values[:,6])
axs[0, 0].set(title = 'Boston Housing', ylabel='IAS')
axs[1, 0].plot(['DT', 'NN', 'SVM'], df1[(df1.Model == "Classification") & (df1.Dataset == "Creditcard Fraud")].values[:,6])
axs[1, 0].set(title = 'Creditcard Fraud', ylabel='IAS')
axs[2, 0].plot(['DT', 'NN', 'SVM'], df1[(df1.Model == "Classification") & (df1.Dataset == "Heart Diseases")].values[:,6])
axs[2, 0].set(title = 'Heart Diseases', ylabel='IAS')
axs[3, 0].plot(['DT', 'NN', 'SVM'], df1[(df1.Model == "Classification") & (df1.Dataset == "Motor Temperature")].values[:,6])
axs[3, 0].set(title='Motor Temperature', ylabel='IAS')
axs[0, 1].plot(['DT', 'NN', 'SVM'], df1[(df1.Model == "Classification") & (df1.Dataset == "Boston Housing")].values[:,4])
axs[0, 1].set(title='Boston Housing', ylabel='MEC')
axs[1, 1].plot(['DT', 'NN', 'SVM'], df1[(df1.Model == "Classification") & (df1.Dataset == "Creditcard Fraud")].values[:,4])
axs[1, 1].set(title ='Creditcard Fraud', ylabel='MEC')
axs[2, 1].plot(['DT', 'NN', 'SVM'], df1[(df1.Model == "Classification") & (df1.Dataset == "Heart Diseases")].values[:,4])
axs[2, 1].set(title='Heart Diseases', ylabel='MEC')
axs[3, 1].plot(['DT', 'NN', 'SVM'], df1[(df1.Model == "Classification") & (df1.Dataset == "Motor Temperature")].values[:,4])
axs[3, 1].set(title='Motor Temperature', ylabel='MEC')


plt.rcParams["figure.figsize"] = (20,10)
fig, axs = plt.subplots(2, 2, figsize=(6,3))
axs[0, 0].plot(['DT', 'SVM', 'NN', 'LR'], df[(df.Model == "Regression") & (df.Dataset == "Boston Housing")].values[:,6])
axs[0, 0].set_title('Boston Housing')
axs[0, 1].plot(['DT', 'SVM', 'NN'], df[(df.Model == "Regression") & (df.Dataset == "Creditcard Fraud")].values[:,6])
axs[0, 1].set_title('Creditcard Fraud')
axs[1, 0].plot(['DT', 'SVM', 'NN', 'LR'], df[(df.Model == "Regression") & (df.Dataset == "Heart Diseases")].values[:,6])
axs[1, 0].set_title('Heart Diseases')
axs[1, 1].plot(['DT', 'NN', 'LR', 'SVM'], df[(df.Model == "Regression") & (df.Dataset == "Motor Temperature")].values[:,6])
axs[1, 1].set_title('Motor Temperature')
for ax in axs.flat:
    ax.set(xlabel='Machine learning method', ylabel='MEC')


#plt.rcParams["figure.figsize"] = (200,10)
fig, axs = plt.subplots(4, 2, figsize=(10,8))
axs[0, 0].plot(['DT', 'SVM', 'NN',], df[(df.Model == "Classification") & (df.Dataset == "Boston Housing")].values[:,5])
axs[0, 0].set_title('Boston Housing')
axs[1, 0].plot(['DT', 'SVM', 'NN'], df[(df.Model == "Classification") & (df.Dataset == "Creditcard Fraud")].values[:,5])
axs[1, 0].set_title('Creditcard Fraud')
axs[2, 0].plot(['DT', 'SVM', 'NN'], df[(df.Model == "Classification") & (df.Dataset == "Heart Diseases")].values[:,5])
axs[2, 0].set_title('Heart Diseases')
axs[3, 0].plot(['DT', 'NN', 'SVM'], df[(df.Model == "Classification") & (df.Dataset == "Motor Temperature")].values[:,5])
axs[3, 0].set_title('Motor Temperature')
axs[0, 1].plot(['DT', 'SVM', 'NN',], df[(df.Model == "Classification") & (df.Dataset == "Boston Housing")].values[:,5])
axs[0, 1].set_title('Boston Housing')
axs[1, 1].plot(['DT', 'SVM', 'NN'], df[(df.Model == "Classification") & (df.Dataset == "Creditcard Fraud")].values[:,5])
axs[1, 1].set_title('Creditcard Fraud')
axs[2, 1].plot(['DT', 'SVM', 'NN'], df[(df.Model == "Classification") & (df.Dataset == "Heart Diseases")].values[:,5])
axs[2, 1].set_title('Heart Diseases')
axs[3, 1].plot(['DT', 'NN', 'SVM'], df[(df.Model == "Classification") & (df.Dataset == "Motor Temperature")].values[:,5])
axs[3, 1].set_title('Motor Temperature')
for ax in axs.flat:
    ax.set(xlabel = '.', ylabel='IAS')
print(axs.flat)    
 
plt.rcParams["figure.figsize"] = (20,10)
fig, axs = plt.subplots(2, 2, figsize=(6,3))
axs[0, 0].plot(['DT', 'SVM', 'NN'], df[(df.Model == "Classification") & (df.Dataset == "Boston Housing")].values[:,6])
axs[0, 0].set_title('Boston Housing')
axs[0, 1].plot(['DT', 'SVM', 'NN'], df[(df.Model == "Classification") & (df.Dataset == "Creditcard Fraud")].values[:,6])
axs[0, 1].set_title('Creditcard Fraud')
axs[1, 0].plot(['DT', 'SVM', 'NN'], df[(df.Model == "Classification") & (df.Dataset == "Heart Diseases")].values[:,6])
axs[1, 0].set_title('Heart Diseases')
axs[1, 1].plot(['DT', 'NN', 'SVM'], df[(df.Model == "Classification") & (df.Dataset == "Motor Temperature")].values[:,6])
axs[1, 1].set_title('Motor Temperature')
for ax in axs.flat:
    ax.set(xlabel='Machine learning method', ylabel='MEC')

df_clf = df[(df.Model == "Classification") & (df.Dataset != "Creditcard Fraud") & ((df.Technique == "Neural Network") | (df.Technique == "Support Vector Machine"))].pivot(index='Dataset', columns='Technique', values='IAS')
df_rgs = df[(df.Model == "Regression") & (df.Dataset != "Creditcard Fraud") & ((df.Technique == "Neural Network") | (df.Technique == "Support Vector Machine"))].pivot(index='Dataset', columns='Technique', values='IAS')
df_clf.plot(kind = 'line', rot = 90, sort_columns = True, figsize = (10,8), color = list(['tab:orange','tab:green']))
df_rgs.plot(kind = 'line', rot = 90, sort_columns = True, figsize = (10,8), color = list(['tab:orange','tab:green']))

df_clf = df[(df.Model == "Classification") & (df.Dataset != "Creditcard Fraud")].pivot(index='Dataset', columns='Technique', values='IAS')
df_rgs = df[(df.Model == "Regression") & (df.Dataset != "Creditcard Fraud")].pivot(index='Dataset', columns='Technique', values='IAS')
df_clf.plot(kind = 'bar', rot = 90, sort_columns = True, figsize = (10,8), color = list(['tab:blue','tab:orange','tab:green']))
df_rgs.plot(kind = 'bar', rot = 90, sort_columns = True, figsize = (10,8), color = list(['tab:blue','tab:red','tab:orange','tab:green']))

print(df[(df.Model == "Classification") & (df.Dataset != "Creditcard Fraud")].corr(method = 'pearson').round(decimals=3))
print(df[(df.Model == "Regression") & (df.Dataset != "Creditcard Fraud")].corr(method = 'pearson').round(decimals=3))
print(df[(df.Dataset != "Creditcard Fraud")].corr(method = 'pearson').round(decimals=3))


