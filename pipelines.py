import pickle
import pandas as pd
import json
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_union, Pipeline


class DataFrameSelector(BaseEstimator, TransformerMixin):
    '''
    Selects columns from a Pandas DataFrame using attr
    Must implement fit() and transform(), as well as get_feature_names() as a passthrough
    '''
    def __init__(self, attr: list):
        self.attr = attr

    def fit(self, X, y=None):
        '''No fit action required for this selector class'''
        return self

    def transform(self, X):
        '''
        Selects from X the columns listed in self.attr
        If any of the self.attr are not in the target dataframe,
            will print a warning and skip those attrs
        '''
        # X must be a pd.DataFrame
        existing_attr_only = list(set(X.columns.tolist()).intersection(set(self.attr)))
        n_exluded_attrs = len(self.attr) - len(existing_attr_only)
        if n_exluded_attrs > 0:
            print(f'WARNING: exluded {n_exluded_attrs} columns which were not found in the passed dataframe')
        return X[existing_attr_only]
    
    def get_feature_names(self):
        # return df.columns.tolist()
        return self.attr


class customPipeline(Pipeline):
    '''
    Custom modified Pipeline which implements get_feature_names()
        - specifically works with OneHotEncoder Transformer
        - may need additional modification to work with other transformers
    '''
    def get_feature_names(self):
        labels = []
        for name, step in self.steps:
            try:
                if isinstance(step, OneHotEncoder):
                    labels = step.get_feature_names(labels)
                else:
                    labels = step.get_feature_names()
            except AttributeError:
                pass
            if isinstance(labels,np.ndarray):
                labels = labels.tolist()
            elif isinstance(labels,str):
                labels = [labels]
        return labels


def get_pipeline_selectors(no_op_path=None,
                            num_attrs_path=None,
                            lognum_attrs_path=None,
                            cat_attrs_path=None,
                            encoded_attrs_path=None) -> dict:
    '''
    Returns lists of column names from target jsons to use with get_pipeline()
    INPUT:
        (str | [str]): Each input is either a string representing a json file, or a list of such paths
    RETURNS:
        (dict): dictionary of <Op_category>:[<column_names>]
    '''

    # Helper
    def read_json_list(path) -> list:
        # path may be a single string, or a list of strings
        if path is not None:
            if type(path) == list:
                # list of paths
                result = []
                for path_i in path:
                    with open(path_i, 'r') as f:
                        result.extend(json.load(f))
            else:
                # single path
                with open(path, 'r') as f:
                    result = json.load(f)
            return result
        return list()

    # Process with no_op_pipeline (select)
    no_op_attrs = read_json_list(no_op_path)

    # Process with num_pipeline (select, impute, standardscale)
    num_attrs = read_json_list(num_attrs_path)

    # Process with num_pipeline (select, exp, impute, standardscale)
    lognum_attrs = read_json_list(lognum_attrs_path)
    
    # Process with cat_pipeline (select, one-hot-encode)
    cat_attrs = read_json_list(cat_attrs_path)

    # Process with encoded_pipeline (select, impute)
    encoded_attrs = read_json_list(encoded_attrs_path)

    return {
        'no_op_attrs':no_op_attrs,
        'num_attrs':num_attrs,
        'lognum_attrs':lognum_attrs,
        'cat_attrs':cat_attrs,
        'encoded_attrs':encoded_attrs
    }


def load_trained_pipeline(path:str):
    '''
    Returns a loaded pipeline from a pickle file
    '''
    print(f'Loading pipeline from {path}')
    loaded_pipeline = pickle.load(open(path, 'rb'))
    return loaded_pipeline


def save_trained_pipeline(pipeline, path:str):
    '''
    Saves the pipeline to the target path as a pickle binary
    '''
    print(f'Saving pipeline to {path}')
    pickle.dump(pipeline, open(path, 'wb'))


def log_linear(x):
    '''
    Function used by log-linear transformer in get_custom_pipeline
    MUST be defined out of local scope in order to pickle pipeline model
    Convert log-scale measures to linear-scale
    '''
    return 10**(x/10)


def get_custom_pipeline(no_op_attrs, num_attrs, lognum_attrs, cat_attrs, encoded_attrs):
        '''
        Returns a post-processing pipeline for a unified DF
        The use of customPipeline is to enable retrival of feature labels. Note that
        editing or creating new pipelines may require modifying 
        customPipeline.get_feature_names()
        '''
        
        pipeline_ops = []
        
        # Define the no-operation pipeline
        if len(no_op_attrs) > 0:
            no_op_pipeline = customPipeline([
                ('selector', DataFrameSelector(no_op_attrs)),
            ])
            pipeline_ops.append(no_op_pipeline)

        # Define the numerical pipeline
        #   standardscaler scales to zero centered unit variance
        #   z = (x - u) / stddev
        if len(num_attrs) > 0:
            num_pipeline = customPipeline([
                ('selector', DataFrameSelector(num_attrs)),
                ('imputer', SimpleImputer(strategy="median")),
                ('std_scaler', StandardScaler()),
            ]) 
            pipeline_ops.append(num_pipeline)

        # Define the numerical pipeline (logarithmic pcodes)
        #   standardscaler scales to zero centered unit variance
        #   z = (x - u) / stddev
        if len(lognum_attrs) > 0:

            lognum_pipeline = customPipeline([
                ('selector', DataFrameSelector(lognum_attrs)),
                ('imputer', SimpleImputer(strategy="median")),
                ('log-linear', FunctionTransformer(log_linear)),
                ('std_scaler', StandardScaler()),
            ]) 
            pipeline_ops.append(lognum_pipeline)

        # Define the categorical pipeline
        if len(cat_attrs) > 0:
            cat_pipeline = customPipeline([
                ('selector', DataFrameSelector(cat_attrs)),
                ('one_hot_encoder', OneHotEncoder()),
            ])
            pipeline_ops.append(cat_pipeline)

        # Define the encoded CT column pipeline
        if len(encoded_attrs) > 0:
            encoded_pipeline = customPipeline([
                ('selector', DataFrameSelector(encoded_attrs)),
                ('imputer', SimpleImputer(strategy="median")),
            ])
            pipeline_ops.append(encoded_pipeline)

        return make_union(*pipeline_ops)
