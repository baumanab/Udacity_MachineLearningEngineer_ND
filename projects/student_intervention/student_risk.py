
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pickle

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import pandas as pd
import numpy as np

    
def load_pkl(filename):
    
    """
    Function accepts a .pkl filename and loads (de-serializes) object
    """
    
    with open(filename, "r") as infile:
        obj = pickle.load(infile)
        return obj
    
def generate_all_features(data_filename):
    
    """
    Function accepts a csv (data_filename), reads it into pandas, splits out features, and returns
    a feature array.
    """
    
    student_data = pd.read_csv(data_filename)
    
    # Extract feature (X) and target (y) columns
    feature_cols = list(student_data.columns[:-1])  # all columns but last are features
    target_col = student_data.columns[-1]  # last column is the target/label

    X_all = student_data[feature_cols]  # feature values for all students
    
    return X_all
    

def preprocess_features(X):
    
    """
    Function accepts a feature array (X) and returns pre-processed features (dummie variables etc.)
    """
    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty

    # Check each column
    for col, col_data in X.iteritems():
        # If data type is non-numeric, try to replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        # Note: This should change the data type for yes/no columns to int

        # If still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'school' => 'school_GP', 'school_MS'

        outX = outX.join(col_data)  # collect column(s) in output dataframe

    return outX
    

def student_pass_probs(classifier, features):
    """
    Returns probability of student passing given a classifier and feature set.
    inputs: features, classifier
    outputs: array of probabilities per student
    """
    
    probs= np.round(classifier.predict_proba(features) * 100, decimals=2)    
    probs_df= pd.DataFrame(probs, columns= ['Yes', 'No'])
    
    return probs_df


def return_at_risk_students(dataframe, cut_level, threshold):
    
    """
    Function accepts a student pass probability dataframe, a cut level (quartiles are default but deciles,
    can also be specified) and a threshold value (as a percent), and returns several dataframes
    and corresponding exports to csv.
    
    Outputs:
    DataFrame 1: Pass probability dataframe (input) with added column for decile or quarile bins
    DataFrame 2: DataFrame one selected for rows <= threshold
    Remaining 4 or 10 DataFrames: DataFrame 1 split by quartile or decile
    
    csv files corresponding to each DataFrame
    """
    # defend against treshold argument as string when using CLI
    threshold= float(threshold)
    # set up cuts
    if cut_level== 'quartiles':
        bins= bins = [0, 25, 50, 75, 101]
        label_list= ['lowest', 'low', 'mid', 'high']
    elif cut_level == 'deciles':
        bins= range(0,110,10)
        label_list= [str(num) for num in range(1,11)]
        
    # assign bins into new column and export to csv
    dataframe['bin']= pd.cut(dataframe.Yes, bins= bins, labels= label_list, include_lowest= True)
    dataframe.to_csv('all_student_probabilites_and_bins.csv')
    
    
    # select by threshold and export to csv
    threshold_df= dataframe[dataframe.Yes <= threshold].sort_values(by= 'Yes')
    threshold_df.to_csv('threshold.csv')
    
    # split by bins and export to csv
    df_list= []
    for index, label in enumerate(label_list): 
        
        df_index= dataframe[dataframe['bin'] == label]  
        df_index.name= label # name the dataframe as an attribute for use in export to csv
        df_list.append(df_index)
        
        # export to csv
        name= df_index.name + '.csv'
        df_index.to_csv(name)
        
        
    return_tuple= (threshold_df, dataframe) + tuple(df_list)
    
    print "Unpack {} variables.\
    First is threshold dataframe, second is dataframe labeled with quartiles or deciles \
    the remainder are either the individual quartile or decile dataframes.".format(len(return_tuple))
    
    return(return_tuple)  


def student_at_risk_info(classifier_filename, data_filename, cut_level, threshold): 
    
    """
    Function accepts a classifier .pkl filename, a student data filename, a cut level (quartiles or deciles)
    and a threshold value (eg 60% would be 60), and returns several dataframes and corresponding exports to csv.
    
    Outputs:
    
    DataFrame 1: Pass probability dataframe (input) with added column for decile or quarile bins
    DataFrame 2: DataFrame one selected for rows <= threshold
    Remaining 4 or 10 DataFrames: DataFrame 1 split by quartile or decile
    
    csv files corresponding to each DataFrame
    
    How to Run:  on command line enter python student_risk.py and 4 arguments separated by spaces 
    example: python student_risk.py tuned_classifier.pkl student-data.csv 'quartiles' 60
    """
    
    # laod classifier from .pkl
    print('Loading classifier')
    classifier= load_pkl(classifier_filename)
    
    # extract and pre-process all features
    print('Reading data, extracting features, pre-processing features')
    X_all= generate_all_features(data_filename)
    all_features = preprocess_features(X_all)

    
    # extract top 10 features determined in classifier development
    print('Extracting top 10 features')
    feature_list= ['failures', 'absences', 'paid', 'goout', 'reason_reputation',
                   'Mjob_health', 'guardian_other', 'Medu', 'reason_course', 'Fedu']
    
    features= all_features[feature_list]
    
    
    # create student probability DataFrame
    print('Generating DataFrames and .csv files')
    prob_dataframe= student_pass_probs(classifier, features)
    
    # make cuts and bin, returning corresponding DataFrames and creating .csv
    return_at_risk_students(prob_dataframe, cut_level, threshold)
    

student_at_risk_info(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
