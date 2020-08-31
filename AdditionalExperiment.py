# Additional Experiment

# run the dependencies
get_ipython().run_line_magic('run', 'Dependencies')
# run the complexity measurements
get_ipython().run_line_magic('run', 'ComplexityMeasurements')

#create result dataframe
results = pd.DataFrame(columns=["Dataset", "Model", "Depth", "NF", "IAS", "MEC"])


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

#Define train and test data
feature_cols = ['LSTAT','RM', 'NOX', 'TAX', 'INDUS']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDVc # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

#Decision tree classification
Technique = "DecisionTree"
Model = "Classification"
BH_DT_clf = DecisionTreeClassifier(max_depth=3, random_state=1) # Create Decision Tree classifer object
BH_DT_clf = BH_DT_clf.fit(X_train,y_train)
y_pred = BH_DT_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, BH_DT_clf.get_depth(), feature_used(BH_DT_clf, X, 100), interaction_strength_class(BH_DT_clf, X, ['1', '2', '3', '4'], 200), main_effect_complexity_class(BH_DT_clf, X, ['1', '2', '3', '4'], 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF", "IAS", "MEC"])])

BH_DT_clf = DecisionTreeClassifier(max_depth=4, random_state=1) # Create Decision Tree classifer object
BH_DT_clf = BH_DT_clf.fit(X_train,y_train)
y_pred = BH_DT_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, BH_DT_clf.get_depth(), feature_used(BH_DT_clf, X, 100), interaction_strength_class(BH_DT_clf, X, ['1', '2', '3', '4'], 200), main_effect_complexity_class(BH_DT_clf, X, ['1', '2', '3', '4'], 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF",  "IAS", "MEC"])])

BH_DT_clf = DecisionTreeClassifier(max_depth=5, random_state=1) # Create Decision Tree classifer object
BH_DT_clf = BH_DT_clf.fit(X_train,y_train)
y_pred = BH_DT_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, BH_DT_clf.get_depth(), feature_used(BH_DT_clf, X, 100), interaction_strength_class(BH_DT_clf, X, ['1', '2', '3', '4'], 200), main_effect_complexity_class(BH_DT_clf, X, ['1', '2', '3', '4'], 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF",  "IAS", "MEC"])])

BH_DT_clf = DecisionTreeClassifier(max_depth=6, random_state=1) # Create Decision Tree classifer object
BH_DT_clf = BH_DT_clf.fit(X_train,y_train)
y_pred = BH_DT_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, BH_DT_clf.get_depth(), feature_used(BH_DT_clf, X, 100), interaction_strength_class(BH_DT_clf, X, ['1', '2', '3', '4'], 200), main_effect_complexity_class(BH_DT_clf, X, ['1', '2', '3', '4'], 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF",  "IAS", "MEC"])])

BH_DT_clf = DecisionTreeClassifier(max_depth=7, random_state=1) # Create Decision Tree classifer object
BH_DT_clf = BH_DT_clf.fit(X_train,y_train)
y_pred = BH_DT_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, BH_DT_clf.get_depth(), feature_used(BH_DT_clf, X, 100), interaction_strength_class(BH_DT_clf, X, ['1', '2', '3', '4'], 200), main_effect_complexity_class(BH_DT_clf, X, ['1', '2', '3', '4'], 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF", "IAS", "MEC"])])

BH_DT_clf = DecisionTreeClassifier(max_depth=8, random_state=1) # Create Decision Tree classifer object
BH_DT_clf = BH_DT_clf.fit(X_train,y_train)
y_pred = BH_DT_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, BH_DT_clf.get_depth(), feature_used(BH_DT_clf, X, 100), interaction_strength_class(BH_DT_clf, X, ['1', '2', '3', '4'], 200), main_effect_complexity_class(BH_DT_clf, X, ['1', '2', '3', '4'], 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF",  "IAS", "MEC"])])

BH_DT_clf = DecisionTreeClassifier(max_depth=9, random_state=1) # Create Decision Tree classifer object
BH_DT_clf = BH_DT_clf.fit(X_train,y_train)
y_pred = BH_DT_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, BH_DT_clf.get_depth(), feature_used(BH_DT_clf, X, 100), interaction_strength_class(BH_DT_clf, X, ['1', '2', '3', '4'], 200), main_effect_complexity_class(BH_DT_clf, X, ['1', '2', '3', '4'], 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF",  "IAS", "MEC"])])

BH_DT_clf = DecisionTreeClassifier(max_depth=10, random_state=1) # Create Decision Tree classifer object
BH_DT_clf = BH_DT_clf.fit(X_train,y_train)
y_pred = BH_DT_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, BH_DT_clf.get_depth(), feature_used(BH_DT_clf, X, 100), interaction_strength_class(BH_DT_clf, X, ['1', '2', '3', '4'], 200), main_effect_complexity_class(BH_DT_clf, X, ['1', '2', '3', '4'], 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF",  "IAS", "MEC"])])

BH_DT_clf = DecisionTreeClassifier(max_depth=11, random_state=1) # Create Decision Tree classifer object
BH_DT_clf = BH_DT_clf.fit(X_train,y_train)
y_pred = BH_DT_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, BH_DT_clf.get_depth(), feature_used(BH_DT_clf, X, 100), interaction_strength_class(BH_DT_clf, X, ['1', '2', '3', '4'], 200), main_effect_complexity_class(BH_DT_clf, X, ['1', '2', '3', '4'], 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF",  "IAS", "MEC"])])

BH_DT_clf = DecisionTreeClassifier(max_depth=12, random_state=1) # Create Decision Tree classifer object
BH_DT_clf = BH_DT_clf.fit(X_train,y_train)
y_pred = BH_DT_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, BH_DT_clf.get_depth(), feature_used(BH_DT_clf, X, 100), interaction_strength_class(BH_DT_clf, X, ['1', '2', '3', '4'], 200), main_effect_complexity_class(BH_DT_clf, X, ['1', '2', '3', '4'], 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF", "IAS", "MEC"])])

BH_DT_clf = DecisionTreeClassifier(max_depth=13, random_state=1) # Create Decision Tree classifer object
BH_DT_clf = BH_DT_clf.fit(X_train,y_train)
y_pred = BH_DT_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, BH_DT_clf.get_depth(), feature_used(BH_DT_clf, X, 100), interaction_strength_class(BH_DT_clf, X, ['1', '2', '3', '4'], 200), main_effect_complexity_class(BH_DT_clf, X, ['1', '2', '3', '4'], 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF",  "IAS", "MEC"])])

BH_DT_clf = DecisionTreeClassifier(max_depth=14, random_state=1) # Create Decision Tree classifer object
BH_DT_clf = BH_DT_clf.fit(X_train,y_train)
y_pred = BH_DT_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, BH_DT_clf.get_depth(), feature_used(BH_DT_clf, X, 100), interaction_strength_class(BH_DT_clf, X, ['1', '2', '3', '4'], 200), main_effect_complexity_class(BH_DT_clf, X, ['1', '2', '3', '4'], 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF",  "IAS", "MEC"])])

BH_DT_clf = DecisionTreeClassifier(max_depth=15, random_state=1) # Create Decision Tree classifer object
BH_DT_clf = BH_DT_clf.fit(X_train,y_train)
y_pred = BH_DT_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, BH_DT_clf.get_depth(), feature_used(BH_DT_clf, X, 100), interaction_strength_class(BH_DT_clf, X, ['1', '2', '3', '4'], 200), main_effect_complexity_class(BH_DT_clf, X, ['1', '2', '3', '4'], 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF",  "IAS", "MEC"])])


#Define train and test data
feature_cols = ['LSTAT','RM','PTRATIO','INDUS', 'TAX']
X = df_replacena[feature_cols].values # Features
y = df_replacena.MEDV # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

#Decision tree regression
Model = "Regression"
BH_DT_rgs = DecisionTreeRegressor(random_state=1, max_depth=3) # Create Decision Tree regression object
BH_DT_rgs = BH_DT_rgs.fit(X_train,y_train)
y_pred = BH_DT_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, BH_DT_rgs.get_depth(), feature_used(BH_DT_rgs, X, 100), interaction_strength(BH_DT_rgs, X, 200), main_effect_complexity(BH_DT_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF", "IAS", "MEC"])])

BH_DT_rgs = DecisionTreeRegressor(random_state=1, max_depth=4) # Create Decision Tree regression object
BH_DT_rgs = BH_DT_rgs.fit(X_train,y_train)
y_pred = BH_DT_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, BH_DT_rgs.get_depth(), feature_used(BH_DT_rgs, X, 100), interaction_strength(BH_DT_rgs, X, 200), main_effect_complexity(BH_DT_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF", "IAS", "MEC"])])

BH_DT_rgs = DecisionTreeRegressor(random_state=1, max_depth=5) # Create Decision Tree regression object
BH_DT_rgs = BH_DT_rgs.fit(X_train,y_train)
y_pred = BH_DT_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, BH_DT_rgs.get_depth(), feature_used(BH_DT_rgs, X, 100), interaction_strength(BH_DT_rgs, X, 200), main_effect_complexity(BH_DT_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF", "IAS", "MEC"])])

BH_DT_rgs = DecisionTreeRegressor(random_state=1, max_depth=6) # Create Decision Tree regression object
BH_DT_rgs = BH_DT_rgs.fit(X_train,y_train)
y_pred = BH_DT_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, BH_DT_rgs.get_depth(), feature_used(BH_DT_rgs, X, 100), interaction_strength(BH_DT_rgs, X, 200), main_effect_complexity(BH_DT_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF", "IAS", "MEC"])])

BH_DT_rgs = DecisionTreeRegressor(random_state=1, max_depth=7) # Create Decision Tree regression object
BH_DT_rgs = BH_DT_rgs.fit(X_train,y_train)
y_pred = BH_DT_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, BH_DT_rgs.get_depth(), feature_used(BH_DT_rgs, X, 100), interaction_strength(BH_DT_rgs, X, 200), main_effect_complexity(BH_DT_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF", "IAS", "MEC"])])

BH_DT_rgs = DecisionTreeRegressor(random_state=1, max_depth=8) # Create Decision Tree regression object
BH_DT_rgs = BH_DT_rgs.fit(X_train,y_train)
y_pred = BH_DT_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, BH_DT_rgs.get_depth(), feature_used(BH_DT_rgs, X, 100), interaction_strength(BH_DT_rgs, X, 200), main_effect_complexity(BH_DT_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF", "IAS", "MEC"])])

BH_DT_rgs = DecisionTreeRegressor(random_state=1, max_depth=9) # Create Decision Tree regression object
BH_DT_rgs = BH_DT_rgs.fit(X_train,y_train)
y_pred = BH_DT_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, BH_DT_rgs.get_depth(), feature_used(BH_DT_rgs, X, 100), interaction_strength(BH_DT_rgs, X, 200), main_effect_complexity(BH_DT_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF", "IAS", "MEC"])])

BH_DT_rgs = DecisionTreeRegressor(random_state=1, max_depth=10) # Create Decision Tree regression object
BH_DT_rgs = BH_DT_rgs.fit(X_train,y_train)
y_pred = BH_DT_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, BH_DT_rgs.get_depth(), feature_used(BH_DT_rgs, X, 100), interaction_strength(BH_DT_rgs, X, 200), main_effect_complexity(BH_DT_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF", "IAS", "MEC"])])

BH_DT_rgs = DecisionTreeRegressor(random_state=1, max_depth=11) # Create Decision Tree regression object
BH_DT_rgs = BH_DT_rgs.fit(X_train,y_train)
y_pred = BH_DT_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, BH_DT_rgs.get_depth(), feature_used(BH_DT_rgs, X, 100), interaction_strength(BH_DT_rgs, X, 200), main_effect_complexity(BH_DT_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF", "IAS", "MEC"])])

BH_DT_rgs = DecisionTreeRegressor(random_state=1, max_depth=12) # Create Decision Tree regression object
BH_DT_rgs = BH_DT_rgs.fit(X_train,y_train)
y_pred = BH_DT_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, BH_DT_rgs.get_depth(), feature_used(BH_DT_rgs, X, 100), interaction_strength(BH_DT_rgs, X, 200), main_effect_complexity(BH_DT_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF", "IAS", "MEC"])])

BH_DT_rgs = DecisionTreeRegressor(random_state=1, max_depth=13) # Create Decision Tree regression object
BH_DT_rgs = BH_DT_rgs.fit(X_train,y_train)
y_pred = BH_DT_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, BH_DT_rgs.get_depth(), feature_used(BH_DT_rgs, X, 100), interaction_strength(BH_DT_rgs, X, 200), main_effect_complexity(BH_DT_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF", "IAS", "MEC"])])

BH_DT_rgs = DecisionTreeRegressor(random_state=1, max_depth=14) # Create Decision Tree regression object
BH_DT_rgs = BH_DT_rgs.fit(X_train,y_train)
y_pred = BH_DT_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, BH_DT_rgs.get_depth(), feature_used(BH_DT_rgs, X, 100), interaction_strength(BH_DT_rgs, X, 200), main_effect_complexity(BH_DT_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF", "IAS", "MEC"])])

BH_DT_rgs = DecisionTreeRegressor(random_state=1, max_depth=15) # Create Decision Tree regression object
BH_DT_rgs = BH_DT_rgs.fit(X_train,y_train)
y_pred = BH_DT_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, BH_DT_rgs.get_depth(), feature_used(BH_DT_rgs, X, 100), interaction_strength(BH_DT_rgs, X, 200), main_effect_complexity(BH_DT_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF", "IAS", "MEC"])])


# Heart Diseases models
Dataset = "Heart Diseases"

#read data
df = pd.read_csv(r"C:\Users\Alexandra.vanderMost\Afstuderen\Data\Heart.csv")

#Define train and test data
feature_cols = ['exang', 'age', 'ca', 'oldpeak', 'thal', 'trestbps']
X = df[feature_cols].values # Features
y = df.target.astype('category') # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

#Decision tree classification
Model = "Classification"
HD_DT_clf = DecisionTreeClassifier(max_depth=3, random_state=1)
HD_DT_clf = HD_DT_clf.fit(X_train,y_train)
y_pred = HD_DT_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, HD_DT_clf.get_depth(), feature_used(HD_DT_clf, X, 100), interaction_strength_class(HD_DT_clf, X, [0,1], 200), main_effect_complexity_class(HD_DT_clf, X, [0,1], 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF",  "IAS", "MEC"])])

HD_DT_clf = DecisionTreeClassifier(max_depth=4, random_state=1)
HD_DT_clf = HD_DT_clf.fit(X_train,y_train)
y_pred = HD_DT_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, HD_DT_clf.get_depth(), feature_used(HD_DT_clf, X, 100), interaction_strength_class(HD_DT_clf, X, [0,1], 200), main_effect_complexity_class(HD_DT_clf, X, [0,1], 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF",  "IAS", "MEC"])])

HD_DT_clf = DecisionTreeClassifier(max_depth=5, random_state=1)
HD_DT_clf = HD_DT_clf.fit(X_train,y_train)
y_pred = HD_DT_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, HD_DT_clf.get_depth(), feature_used(HD_DT_clf, X, 100), interaction_strength_class(HD_DT_clf, X, [0,1], 200), main_effect_complexity_class(HD_DT_clf, X, [0,1], 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF",  "IAS", "MEC"])])

HD_DT_clf = DecisionTreeClassifier(max_depth=6, random_state=1)
HD_DT_clf = HD_DT_clf.fit(X_train,y_train)
y_pred = HD_DT_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, HD_DT_clf.get_depth(), feature_used(HD_DT_clf, X, 100), interaction_strength_class(HD_DT_clf, X, [0,1], 200), main_effect_complexity_class(HD_DT_clf, X, [0,1], 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF",  "IAS", "MEC"])])

HD_DT_clf = DecisionTreeClassifier(max_depth=7, random_state=1)
HD_DT_clf = HD_DT_clf.fit(X_train,y_train)
y_pred = HD_DT_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, HD_DT_clf.get_depth(), feature_used(HD_DT_clf, X, 100), interaction_strength_class(HD_DT_clf, X, [0,1], 200), main_effect_complexity_class(HD_DT_clf, X, [0,1], 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF",  "IAS", "MEC"])])

HD_DT_clf = DecisionTreeClassifier(max_depth=8, random_state=1)
HD_DT_clf = HD_DT_clf.fit(X_train,y_train)
y_pred = HD_DT_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, HD_DT_clf.get_depth(), feature_used(HD_DT_clf, X, 100), interaction_strength_class(HD_DT_clf, X, [0,1], 200), main_effect_complexity_class(HD_DT_clf, X, [0,1], 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF",  "IAS", "MEC"])])

HD_DT_clf = DecisionTreeClassifier(max_depth=9, random_state=1)
HD_DT_clf = HD_DT_clf.fit(X_train,y_train)
y_pred = HD_DT_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, HD_DT_clf.get_depth(), feature_used(HD_DT_clf, X, 100), interaction_strength_class(HD_DT_clf, X, [0,1], 200), main_effect_complexity_class(HD_DT_clf, X, [0,1], 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF",  "IAS", "MEC"])])

HD_DT_clf = DecisionTreeClassifier(max_depth=10, random_state=1)
HD_DT_clf = HD_DT_clf.fit(X_train,y_train)
y_pred = HD_DT_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, HD_DT_clf.get_depth(), feature_used(HD_DT_clf, X, 100), interaction_strength_class(HD_DT_clf, X, [0,1], 200), main_effect_complexity_class(HD_DT_clf, X, [0,1], 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF",  "IAS", "MEC"])])

HD_DT_clf = DecisionTreeClassifier(max_depth=11, random_state=1)
HD_DT_clf = HD_DT_clf.fit(X_train,y_train)
y_pred = HD_DT_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, HD_DT_clf.get_depth(), feature_used(HD_DT_clf, X, 100), interaction_strength_class(HD_DT_clf, X, [0,1], 200), main_effect_complexity_class(HD_DT_clf, X, [0,1], 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF",  "IAS", "MEC"])])

HD_DT_clf = DecisionTreeClassifier(max_depth=12, random_state=1)
HD_DT_clf = HD_DT_clf.fit(X_train,y_train)
y_pred = HD_DT_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, HD_DT_clf.get_depth(), feature_used(HD_DT_clf, X, 100), interaction_strength_class(HD_DT_clf, X, [0,1], 200), main_effect_complexity_class(HD_DT_clf, X, [0,1], 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF",  "IAS", "MEC"])])

HD_DT_clf = DecisionTreeClassifier(max_depth=13, random_state=1)
HD_DT_clf = HD_DT_clf.fit(X_train,y_train)
y_pred = HD_DT_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, HD_DT_clf.get_depth(), feature_used(HD_DT_clf, X, 100), interaction_strength_class(HD_DT_clf, X, [0,1], 200), main_effect_complexity_class(HD_DT_clf, X, [0,1], 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF",  "IAS", "MEC"])])

HD_DT_clf = DecisionTreeClassifier(max_depth=14, random_state=1)
HD_DT_clf = HD_DT_clf.fit(X_train,y_train)
y_pred = HD_DT_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, HD_DT_clf.get_depth(), feature_used(HD_DT_clf, X, 100), interaction_strength_class(HD_DT_clf, X, [0,1], 200), main_effect_complexity_class(HD_DT_clf, X, [0,1], 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF",  "IAS", "MEC"])])

HD_DT_clf = DecisionTreeClassifier(max_depth=15, random_state=1)
HD_DT_clf = HD_DT_clf.fit(X_train,y_train)
y_pred = HD_DT_clf.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, HD_DT_clf.get_depth(), feature_used(HD_DT_clf, X, 100), interaction_strength_class(HD_DT_clf, X, [0,1], 200), main_effect_complexity_class(HD_DT_clf, X, [0,1], 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF",  "IAS", "MEC"])])


#Define train and test data
feature_cols = ['exang', 'age', 'ca', 'oldpeak', 'thal', 'trestbps']
X = df[feature_cols].values # Features
y = df.target # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

#Decision tree regression
Model = "Regression"
HD_DT_rgs = DecisionTreeRegressor(max_depth=3, random_state=1)
HD_DT_rgs = HD_DT_rgs.fit(X_train,y_train)
y_pred = HD_DT_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, HD_DT_rgs.get_depth(), feature_used(HD_DT_rgs, X, 100), interaction_strength(HD_DT_rgs, X, 200), main_effect_complexity(HD_DT_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF", "IAS", "MEC"])])

HD_DT_rgs = DecisionTreeRegressor(max_depth=4, random_state=1)
HD_DT_rgs = HD_DT_rgs.fit(X_train,y_train)
y_pred = HD_DT_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, HD_DT_rgs.get_depth(), feature_used(HD_DT_rgs, X, 100), interaction_strength(HD_DT_rgs, X, 200), main_effect_complexity(HD_DT_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF", "IAS", "MEC"])])

HD_DT_rgs = DecisionTreeRegressor(max_depth=5, random_state=1)
HD_DT_rgs = HD_DT_rgs.fit(X_train,y_train)
y_pred = HD_DT_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, HD_DT_rgs.get_depth(), feature_used(HD_DT_rgs, X, 100), interaction_strength(HD_DT_rgs, X, 200), main_effect_complexity(HD_DT_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF", "IAS", "MEC"])])

HD_DT_rgs = DecisionTreeRegressor(max_depth=6, random_state=1)
HD_DT_rgs = HD_DT_rgs.fit(X_train,y_train)
y_pred = HD_DT_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, HD_DT_rgs.get_depth(), feature_used(HD_DT_rgs, X, 100), interaction_strength(HD_DT_rgs, X, 200), main_effect_complexity(HD_DT_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF", "IAS", "MEC"])])

HD_DT_rgs = DecisionTreeRegressor(max_depth=7, random_state=1)
HD_DT_rgs = HD_DT_rgs.fit(X_train,y_train)
y_pred = HD_DT_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, HD_DT_rgs.get_depth(), feature_used(HD_DT_rgs, X, 100), interaction_strength(HD_DT_rgs, X, 200), main_effect_complexity(HD_DT_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF", "IAS", "MEC"])])

HD_DT_rgs = DecisionTreeRegressor(max_depth=8, random_state=1)
HD_DT_rgs = HD_DT_rgs.fit(X_train,y_train)
y_pred = HD_DT_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, HD_DT_rgs.get_depth(), feature_used(HD_DT_rgs, X, 100), interaction_strength(HD_DT_rgs, X, 200), main_effect_complexity(HD_DT_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF", "IAS", "MEC"])])

HD_DT_rgs = DecisionTreeRegressor(max_depth=9, random_state=1)
HD_DT_rgs = HD_DT_rgs.fit(X_train,y_train)
y_pred = HD_DT_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, HD_DT_rgs.get_depth(), feature_used(HD_DT_rgs, X, 100), interaction_strength(HD_DT_rgs, X, 200), main_effect_complexity(HD_DT_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF", "IAS", "MEC"])])

HD_DT_rgs = DecisionTreeRegressor(max_depth=10, random_state=1)
HD_DT_rgs = HD_DT_rgs.fit(X_train,y_train)
y_pred = HD_DT_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, HD_DT_rgs.get_depth(), feature_used(HD_DT_rgs, X, 100), interaction_strength(HD_DT_rgs, X, 200), main_effect_complexity(HD_DT_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF", "IAS", "MEC"])])

HD_DT_rgs = DecisionTreeRegressor(max_depth=11, random_state=1)
HD_DT_rgs = HD_DT_rgs.fit(X_train,y_train)
y_pred = HD_DT_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, HD_DT_rgs.get_depth(), feature_used(HD_DT_rgs, X, 100), interaction_strength(HD_DT_rgs, X, 200), main_effect_complexity(HD_DT_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF", "IAS", "MEC"])])

HD_DT_rgs = DecisionTreeRegressor(max_depth=12, random_state=1)
HD_DT_rgs = HD_DT_rgs.fit(X_train,y_train)
y_pred = HD_DT_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, HD_DT_rgs.get_depth(), feature_used(HD_DT_rgs, X, 100), interaction_strength(HD_DT_rgs, X, 200), main_effect_complexity(HD_DT_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF", "IAS", "MEC"])])

HD_DT_rgs = DecisionTreeRegressor(max_depth=13, random_state=1)
HD_DT_rgs = HD_DT_rgs.fit(X_train,y_train)
y_pred = HD_DT_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, HD_DT_rgs.get_depth(), feature_used(HD_DT_rgs, X, 100), interaction_strength(HD_DT_rgs, X, 200), main_effect_complexity(HD_DT_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF", "IAS", "MEC"])])

HD_DT_rgs = DecisionTreeRegressor(max_depth=14, random_state=1)
HD_DT_rgs = HD_DT_rgs.fit(X_train,y_train)
y_pred = HD_DT_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, HD_DT_rgs.get_depth(), feature_used(HD_DT_rgs, X, 100), interaction_strength(HD_DT_rgs, X, 200), main_effect_complexity(HD_DT_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF", "IAS", "MEC"])])

HD_DT_rgs = DecisionTreeRegressor(max_depth=15, random_state=1)
HD_DT_rgs = HD_DT_rgs.fit(X_train,y_train)
y_pred = HD_DT_rgs.predict(X_test)
results = pd.concat([results, pd.DataFrame([[Dataset, Model, HD_DT_rgs.get_depth(), feature_used(HD_DT_rgs, X, 100), interaction_strength(HD_DT_rgs, X, 200), main_effect_complexity(HD_DT_rgs, X, 200, 5, 0.05)]], columns=["Dataset", "Model", "Depth", "NF", "IAS", "MEC"])])


pd.set_option('display.max_rows', None)
print(results)

results = results.reset_index()
results = results.drop(columns = ['index', 'MEC'])  
print(results)

df = results.copy()
df = df.drop(df.index[0]).reset_index().drop(columns = ['index'])
df = df.drop(df.index[10:14]).reset_index().drop(columns = ['index'])
df = df.drop(df.index[21]).reset_index().drop(columns = ['index'])
df = df.drop(df.index[31:33]).reset_index().drop(columns = ['index'])
df = df.drop(df.index[41:43]).reset_index().drop(columns = ['index'])
print(df)


df1 = df[(df.Model == "Classification") & (df.Dataset == "Boston Housing")]
df2 = df[(df.Model == "Regression") & (df.Dataset == "Boston Housing")]
fig, axs = plt.subplots(2, 1, figsize=(8,12), sharex=False)
axs[0].plot(df1['Depth'].values, df1['IAS'].values)
axs[0].set(title = 'Classification', ylabel='IAS')
axs[1].plot(df2['Depth'].values, df2['IAS'].values)
axs[1].set(title = 'Regression', xlabel = 'actual tree depth', ylabel='IAS')


df1 = df[(df.Model == "Classification") & (df.Dataset == "Heart Diseases")]
df2 = df[(df.Model == "Regression") & (df.Dataset == "Heart Diseases")]
fig, axs = plt.subplots(2, 1, figsize=(8,12), sharex=False)
axs[0].plot(df1['Depth'].values, df1['IAS'].values)
axs[0].set(title = 'Classification', ylabel='IAS')
axs[1].plot(df2['Depth'].values, df2['IAS'].values)
axs[1].set(title = 'Regression', xlabel = 'actual tree depth', ylabel='IAS')

