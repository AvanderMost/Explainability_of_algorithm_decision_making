# # Complexity measures
import numpy as np # linear algebra
import pandas as pd # use pandas dataframes for data processing
import statistics as st 
import os

from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model


# ## Number of Features used (NF)

def feature_used(model, X, sample_size = 500): 
    NF = 0
    dat1 = X[np.random.choice(X.shape[0], sample_size, replace=False), :]
    for j in range(0,X.shape[1]):
        dat2 = dat1.copy()
        for m in range(0,sample_size):
            while dat2[m,j] == dat1[m,j]:
                dat2[m,j] = X[np.random.choice(X.shape[0], 1, replace=False), j]
        dat1_pred = model.predict(dat1)
        dat2_pred = model.predict(dat2)
        if np.any(dat1_pred != dat2_pred):
            NF = NF + 1   
    return(NF)


# ## ALE approximation 
# Source code is used from the ALEpython package that based ALE function on the the book interpretable ml, written by Christoph Molnar. \citep{https://christophm.github.io/interpretable-ml-book/}. 

def first_order_ale_approximation(model, X, bins, sample_frac=1):
    #effects = pd.DataFrame()
    print("start ALE approximation")
    ale_approximation = []
    feature_bin_bounderies = []

    ale_feature_values = pd.DataFrame()

    
    for j in range(0, X.shape[1]):
        print("feature :", str(j))
        # An ordered array of unique feature values, representing the bins bounderies
        bin_bounderies = np.linspace(min(X[:,j]), max(X[:,j]), bins + 1)
        
        # Determine the effect of each bin
        effects = pd.DataFrame()
        for i in range(0, bins):
            print(i)
            predictions = []
            mod_train_set = pd.DataFrame(X).sample(frac = sample_frac, random_state = 1).values
            for offset in range(2):
                mod_train_set[:,j] = bin_bounderies[i + offset]
                predictions.append(model.predict(mod_train_set))           
            # The individual effects.
            effect_for_bin_i = pd.DataFrame({"index": i, "effects": predictions[1] - predictions[0]})
            effects = pd.concat([effects, effect_for_bin_i], ignore_index=True)
        
        # Average these differences within each bin.
        index_groupby = effects.groupby("index")
        mean_effects_by_index = index_groupby.mean().to_numpy().flatten()
                
        ale = np.array([0, *np.cumsum(mean_effects_by_index)])
        
        # The uncentred mean main effects at the bin centres.
        ale = (ale[1:] + ale[:-1]) / 2    
        
        # Centre the effects by subtracting the mean (the mean of the individual `effects`, 
        # which is equivalently calculated using `mean_effects` and the number of samples in each bin).
        bin_indices = pd.DataFrame({"bin_index": np.arange(0,bins)})
        X_bin_indices = pd.DataFrame({"bin_index": np.clip(np.digitize(X[:,j], bin_bounderies, right=True) -1, 0, None)}).groupby("bin_index").size()
        X_bins_size = (pd.concat([bin_indices, X_bin_indices], axis=1, join='outer', ignore_index=True, keys="bin_index").fillna(0))[1].to_numpy().flatten()
        
        ale -= np.sum(ale * X_bins_size / X.shape[0])
           
        
        ale_approximation.append(ale)
        feature_bin_bounderies.append(bin_bounderies)
    
    #Predict y with the ale approximation
    for j in range(0,X.shape[1]):
        ale_approx_array = np.asarray(ale_approximation[j])
        index = np.clip(np.digitize(X[:,j], feature_bin_bounderies[j], right=True) -1, 0, None)
        ale_feature_values[j] = ale_approx_array[index]
    
    y_pred_ale = np.mean(model.predict(X)) + np.sum(ale_feature_values, axis=1)  
    #print(pd.DataFrame({"model":model.predict(X), "ale":y_pred_ale}))
    print("done")
    return y_pred_ale, ale_approximation, feature_bin_bounderies


# ## Interaction Strength (IAS)
def interaction_strength(model, X, bins, sample_frac=1):
    #create an array of prediction outcomes established with the ALE apporximation
    y_pred_ale, ale_approximation, feature_bin_bounderies  = first_order_ale_approximation(model, X, bins, sample_frac)
    #create an array of the models prediction outcomes
    y_pred = model.predict(X)
    # define the mean value of the  models prediction outcomes
    f0 = np.mean(model.predict(X))
    
    numerator = np.sum((y_pred - y_pred_ale)**2)
    denominator = np.sum((y_pred - f0)**2)
    
    IAS = numerator/denominator
    
    return(IAS)   


# ## Main Eﬀect Complexity (MEC)
def main_effect_complexity(model, X, bins, max_segments, approximation_error, sample_frac=1):

    # get ALE approximation
    y_pred_ale, ale_approximation, feature_bin_bounderies = first_order_ale_approximation(model, X, bins, sample_frac)
    
    # define empty variance array
    Vj = []
    MECj = []

    # for each features we define the segmented linear function gj(xj)
    for j in range(0, X.shape[1]):
        #print("")
        #print("featurenumber is : " + str(j))
        feature_values = X[:,j].reshape(-1,1)
        #determine the fjALE(xj)
        ale_approx_array = np.asarray(ale_approximation[j])
        index = np.clip(np.digitize(X[:,j], feature_bin_bounderies[j], right=True) -1, 0, None)
        target_values = ale_approx_array[index]
        #print(feature_values)
        #print(target_values)
    
        # amount of segments K = 1
        K = 1
        breakpoints = [min(X[:,j]), max(X[:,j])]
        
        # initial gj(xj) with only 1 segment
        g = linear_model.LinearRegression()
        g = g.fit(feature_values, target_values)
        y_pred = g.predict(feature_values)
    
        # calculate the accuracy in R2 for the segmented linear function gj(xj) 
        R2_variance = np.asarray(np.sum((y_pred - target_values)**2))
        R2 = 1 - R2_variance/np.sum((target_values)**2)
       
        # define Beta values
        Beta = np.array([[round(g.intercept_, 2), g.coef_[0]]])
            
        while K < max_segments and R2 < (1-approximation_error):
            #print("K : ", str(K))
            optional_breakpoints = np.setdiff1d(np.unique(np.quantile(X[:,j], np.linspace(0, 1, bins + 1), interpolation="lower")), breakpoints)
            #print(optional_breakpoints)
            max_R_square = 0
            original_breakpoints = breakpoints
            for i in optional_breakpoints:
                #print(i)
                R = 0
                optional_Beta = []
                segment_variance = []
                indices = np.clip(np.digitize(X[:,j], np.sort(np.append(original_breakpoints, i)), right=True) - 1, 0, None) 
                for index in np.unique(indices):
                    mod_train_features = feature_values[(indices == index)]
                    mod_train_target = target_values[(indices == index)]
                    g.fit(mod_train_features,mod_train_target)
                    y_prediction = g.predict(mod_train_features)
                    beta_i = [round(g.intercept_, 2), g.coef_[0]]
                    optional_Beta.append(beta_i)
                    variance = np.sum((y_prediction - mod_train_target)**2)
                    segment_variance.append(variance)
                R_square = 1 - np.sum(segment_variance)/np.sum((target_values)**2)
                if R_square > max_R_square:
                    max_R_square = R_square
                    breakpoints = np.sort(np.append(original_breakpoints, i))
                    Beta = np.array(optional_Beta)
                    R2_variance = segment_variance
            R2 = max_R_square
            K += 1
        K = Beta.shape[0]
        #print("K = ", str(K))
        #print("breakpoints = ", str(breakpoints))
        #print("Beta = ", str(Beta))
        #print("R2 = ", str(R2))
   
        #Greedily set slopes to zero while R2 > 1−epsilon
        indices = np.clip(np.digitize(X[:,j], breakpoints, right=True) - 1, 0, None)
        #starting with the smallest slope
        for i in np.argsort(abs(Beta[:,1])):
            #Mean of ale values in segment
            Ale_values_in_segment = target_values[(indices == i)]
            R_square = 1 - np.sum((Ale_values_in_segment - np.mean(Ale_values_in_segment, axis = 0))**2)/np.sum((Ale_values_in_segment)**2)
            if R_square > (1-approximation_error):
                Beta[i,1] = 0
                Beta[i,0] = np.mean(Ale_values_in_segment, axis = 0)
                R2_variance[i] = np.sum((Ale_values_in_segment - np.mean(Ale_values_in_segment, axis = 0))**2)
 
        MEC_for_j = K + Beta[(Beta[:,1] != 0),1].shape[0] - 1
        MECj.append(MEC_for_j)
        #print("Beta with slopes to 0 = ", str(Beta))
        
        
        
        #average ALEj approximation variance
        V_for_j = np.sum((target_values)**2)/X.shape[0]       
        Vj.append(V_for_j)
    #print("")
    #print("MECj = ", str(MECj))
    #print("Vj = ", str(Vj))    
 
    MEC = np.sum(np.multiply(MECj, Vj))/np.sum(Vj)
    return(MEC)


# ## ALE Approximation for classification models
def first_order_ale_approximation_class(model, X, array_of_classes, bins, sample_frac=1 ):
    print("start ALE approximation")
    ale_approximation = []
    feature_bin_bounderies = []

    ale_feature_values = pd.DataFrame()
    
    #labelEncoder for model_outcome classes
    le = LabelEncoder()
    le.fit(array_of_classes)   
    #print(le.classes_)
    
    for j in range(0, X.shape[1]):
        print("feature :", str(j))
        # An ordered array of unique feature values, representing the bins bounderies
        bin_bounderies = np.linspace(min(X[:,j]), max(X[:,j]), bins + 1)
        
        # Determine the effect of each bin
        effects = pd.DataFrame()
        for i in range(0, bins):
            predictions = []
            mod_train_set = pd.DataFrame(X).sample(frac = sample_frac, random_state = 1).values
            for offset in range(2):
                mod_train_set[:,j] = bin_bounderies[i + offset]
                predictions.append(le.transform(model.predict(mod_train_set)))
                     
            # The individual effects.
            effect_for_bin_i = pd.DataFrame({"index": i, "effect": predictions[1] - predictions[0]})
            effects = pd.concat([effects, effect_for_bin_i], ignore_index=True)
        
        # Average these differences within each bin.
        index_groupby = effects.groupby("index")
        mean_effects_by_index = index_groupby.mean().to_numpy().flatten()
                
        ale = np.array([0, *np.cumsum(mean_effects_by_index)])
        
        # The uncentred mean main effects at the bin centres.
        ale = (ale[1:] + ale[:-1]) / 2    
        
        # Centre the effects by subtracting the mean (the mean of the individual `effects`, 
        # which is equivalently calculated using `mean_effects` and the number of samples in each bin).
        bin_indices = pd.DataFrame({"bin_index": np.arange(0,bins)})
        X_bin_indices = pd.DataFrame({"bin_index": np.clip(np.digitize(X[:,j], bin_bounderies, right=True) -1, 0, None)}).groupby("bin_index").size()
        X_bins_size = (pd.concat([bin_indices, X_bin_indices], axis=1, join='outer', ignore_index=True, keys="bin_index").fillna(0))[1].to_numpy().flatten()
        
        ale -= np.sum(ale * X_bins_size / X.shape[0])
        
        ale_approximation.append(ale)
        feature_bin_bounderies.append(bin_bounderies)
    
    #Predict y with the ale approximation
    for j in range(0,X.shape[1]):
        ale_approx_array = np.asarray(ale_approximation[j])
        index = np.clip(np.digitize(X[:,j], feature_bin_bounderies[j], right=True) -1, 0, None)
        ale_feature_values[j] = ale_approx_array[index]
    
    y_pred_ale_num = np.mean(le.transform(model.predict(X))) + np.sum(ale_feature_values, axis=1)   
    y_pred_ale_index = np.clip(round(y_pred_ale_num).astype(int), 0, len(array_of_classes)-1)
    y_pred_ale_cat = le.inverse_transform(y_pred_ale_index)

    mean = np.mean(le.transform(model.predict(X)))
    model_outcome_index = le.transform(model.predict(X))
    print("done")
    return y_pred_ale_num, y_pred_ale_index, y_pred_ale_cat, ale_approximation, feature_bin_bounderies, model_outcome_index, mean


def interaction_strength_class(model, X, array_of_classes, bins, sample_frac=1 ):
    #return outcomes established with the ALE apporximation
    y_pred_ale_num, y_pred_ale_index, y_pred_ale_cat, ale_approximation, feature_bin_bounderies, model_outcome_index, mean = first_order_ale_approximation_class(model, X, array_of_classes, bins, sample_frac)

    f0 = mean
    
    numerator = np.sum((y_pred_ale_index - model_outcome_index)**2)
    denominator = np.sum((model_outcome_index - f0)**2)
    
    IAS = numerator/denominator
    
    return(IAS)   


def main_effect_complexity_class(model, X, array_of_classes, bins, max_segments, approximation_error, sample_frac=1 ):
    
    # get ALE approximation
    y_pred_ale_num, y_pred_ale_index, y_pred_ale_cat, ale_approximation, feature_bin_bounderies, model_outcome_index, mean = first_order_ale_approximation_class(model, X, array_of_classes, bins, sample_frac)
    
    # define empty variance array
    Vj = []
    MECj = []

    # for each features we define the segmented linear function gj(xj)
    for j in range(0, X.shape[1]):
        #print("")
        #print("featurenumber is : " + str(j))
        feature_values = X[:,j].reshape(-1,1)
        #determine the fjALE(xj)
        ale_approx_array = np.asarray(ale_approximation[j])
        index = np.clip(np.digitize(X[:,j], feature_bin_bounderies[j], right=True) -1, 0, None)
        target_values = ale_approx_array[index]
        #print(feature_values)
        #print(target_values)
    
        # amount of segments K = 1
        K = 1
        breakpoints = [min(X[:,j]), max(X[:,j])]
        
        # initial gj(xj) with only 1 segment
        g = linear_model.LinearRegression()
        g = g.fit(feature_values, target_values)
        y_pred = g.predict(feature_values)
    
        # calculate the accuracy in R2 for the segmented linear function gj(xj) 
        R2_variance = np.asarray(np.sum((y_pred - target_values)**2))
        R2 = 1 - R2_variance/np.sum((target_values)**2)
       
        # define Beta values
        Beta = np.array([[round(g.intercept_, 2), g.coef_[0]]])
            
        while K < max_segments and R2 < (1-approximation_error):
            print("K : ", str(K))
            optional_breakpoints = np.setdiff1d(np.unique(np.quantile(X[:,j], np.linspace(0, 1, bins + 1), interpolation="lower")), breakpoints)
            #print(optional_breakpoints)
            max_R_square = 0
            original_breakpoints = breakpoints
            for i in optional_breakpoints:
                R = 0
                optional_Beta = []
                segment_variance = []
                indices = np.clip(np.digitize(X[:,j], np.sort(np.append(original_breakpoints, i)), right=True) - 1, 0, None) 
                for index in np.unique(indices):
                    mod_train_features = feature_values[(indices == index)]
                    mod_train_target = target_values[(indices == index)]
                    g.fit(mod_train_features,mod_train_target)
                    y_prediction = g.predict(mod_train_features)
                    beta_i = [round(g.intercept_, 2), g.coef_[0]]
                    optional_Beta.append(beta_i)
                    variance = np.sum((y_prediction - mod_train_target)**2)
                    segment_variance.append(variance)
                R_square = 1 - np.sum(segment_variance)/np.sum((target_values)**2)
                if R_square > max_R_square:
                    max_R_square = R_square
                    breakpoints = np.sort(np.append(original_breakpoints, i))
                    Beta = np.array(optional_Beta)
                    R2_variance = segment_variance
            R2 = max_R_square
            K += 1
        K = Beta.shape[0]
        #print("K = ", str(K))
        #print("breakpoints = ", str(breakpoints))
        #print("Beta = ", str(Beta))
        #print("R2 = ", str(R2))
        print("done")
        #Greedily set slopes to zero while R2 > 1−epsilon
        indices = np.clip(np.digitize(X[:,j], breakpoints, right=True) - 1, 0, None)
        #starting with the smallest slope
        for i in np.argsort(abs(Beta[:,1])):
            #Mean of ale values in segment
            Ale_values_in_segment = target_values[(indices == i)]
            R_square = 1 - np.sum((Ale_values_in_segment - np.mean(Ale_values_in_segment, axis = 0))**2)/np.sum((Ale_values_in_segment)**2)
            if R_square > (1-approximation_error):
                Beta[i,1] = 0
                Beta[i,0] = np.mean(Ale_values_in_segment, axis = 0)
                R2_variance[i] = np.sum((Ale_values_in_segment - np.mean(Ale_values_in_segment, axis = 0))**2)
 
        MEC_for_j = K + Beta[(Beta[:,1] != 0),1].shape[0] - 1
        MECj.append(MEC_for_j)
        #print("Beta with slopes to 0 = ", str(Beta))
        
        
        
        #average ALEj approximation variance
        V_for_j = np.sum((target_values)**2)/X.shape[0]       
        Vj.append(V_for_j)
    #print("")
    #print("MECj = ", str(MECj))
    #print("Vj = ", str(Vj))    
 
    MEC = np.sum(np.multiply(MECj, Vj))/np.sum(Vj)
    return(MEC)

