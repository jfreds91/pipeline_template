# core
from typing import Tuple
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import re
import logging
from configparser import ConfigParser
from tqdm import tqdm

# model eval
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, make_scorer
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
import pickle

# models
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# custom modules
import pipelines
import utils
tqdm.pandas()


def load_and_filter(PREDICTION_COLUMN:str,
                    PARQUET_PATH:str,
                    load_and_filter_dict:dict=None
                    ) -> Tuple[pd.DataFrame, pd.Series]:
    """
    This function loads a dataset (from parquet, for now), and runs preprocessing/filtering on it,
        not including pipeline transforms. This should be custom for each dataset.
    INPUTS:
	PREDICTION_COLUMN (str): name of target prediction column in the training data
        PARQUET_PATH (str): Path to parquet file which has contains dataframe 
        load_and_filter_dict (dict): dict which contains any custom info needed by this function.
                                     Used to pass multiple objects through preprocess_data()
    RETURNS:
        X (pd.DataFrame): Feature dataframe ready to run through pipeline
        y (pd.Series): Feature labels
    """

    ### load parquet file into dataframe
    logging.info(f'Loading parquet data from {PARQUET_PATH}')
    data = pd.read_parquet(PARQUET_PATH)

    ### get x and y
    # drop NaN predictions...
    data.dropna(subset=[PREDICTION_COLUMN], inplace=True)

    ### format df as output 
    y = data[PREDICTION_COLUMN].astype(int)
    X_preproc = data.drop(columns=[PREDICTION_COLUMN])

    return X_preproc, y


def preprocess_data(PREDICTION_COLUMN:str,
                    PARQUET_PATH:str,
                    NO_OP_PATH:str,
                    NUM_ATTRS_PATH:str,
                    LOG_ATTRS_PATH:str,
                    CAT_ATTRS_PATH:str,
                    ENCODED_ATTRS_PATH:str,
                    TRAIN_PIPELINE:bool,
                    SAVE_PIPELINE:bool,
                    PIPELINE_PATH:bool,
                    TEST_TRAIN_SPLIT:float,
                    load_and_filter_dict:dict=None,
                    ) -> Tuple[np.array, np.array, pd.Series, pd.Series, list]:
    '''
    This function encapsulates all the data loading, preprocessing, and pipeline ops, and returns training and testing datasets
    INPUTS:
        parquet_path (str): parquet file holding raw parametric and header data parsed from pdels
        sfc_plt_count_path (str): csv file holding a count of PLT tests per SFC, to be added after as an engineered feature
        no_op_path (str): path to json file list of features to use in the no_op pipeline. May be None
        num_attrs_path (str): path to json file list of features to use in the num pipeline. May be None
        cat_attrs_path (str): path to json file list of features to use in the cat pipeline. May be None
        encoded_attrs_path (str): path to json file list of features to use in the encoded pipeline. May be None
        train_pipeline (bool): If true, create and fit a new pipeline object. If false, must specify pipeline path
        save_pipeline (bool): If true, save fitted pipeline
        pipeline_path (bool): location of pickled binary file to load or save pipeline
        TEST_TRAIN_SPLIT (float): what percent of the data shall be test data
        load_and_filter_dict (dict): dict which contains any custom info needed by load_and_filter().
                                     Used to pass multiple objects through preprocess_data()
    RETURNS
        X_train (np.array)
        X_test (np.array)
        y_train (pd.Series)
        y_test (pd.Series)
        labels (list): List of feature labels exported from the pipeline
    '''

    #### load and preprocess training data
    X, y = load_and_filter(PREDICTION_COLUMN=PREDICTION_COLUMN,
        PARQUET_PATH=PARQUET_PATH,
        load_and_filter_dict=load_and_filter_dict
    )

    #### run training data through pipeline
    # get feature selectors
    pipe_attrs = pipelines.get_pipeline_selectors(
        no_op_path=NO_OP_PATH,
        num_attrs_path=NUM_ATTRS_PATH,
        lognum_attrs_path=LOG_ATTRS_PATH,
        cat_attrs_path=CAT_ATTRS_PATH,
        encoded_attrs_path=ENCODED_ATTRS_PATH
    )
    no_op_attrs = pipe_attrs['no_op_attrs']
    num_attrs = pipe_attrs['num_attrs']
    lognum_attrs = pipe_attrs['lognum_attrs']
    cat_attrs = pipe_attrs['cat_attrs']
    encoded_attrs = pipe_attrs['encoded_attrs']

    # load pipeline
    if TRAIN_PIPELINE:
        pipeline = pipelines.get_custom_pipeline(no_op_attrs, num_attrs, lognum_attrs, cat_attrs, encoded_attrs)
        pipeline.fit(X)
    else:
        pipeline = pipelines.load_trained_pipeline(PIPELINE_PATH)
    if SAVE_PIPELINE:
        pipelines.save_trained_pipeline(pipeline, PIPELINE_PATH)

    # transform X    
    X = pipeline.transform(X) #.toarray()

    # get feature labels
    labels = pipeline.get_feature_names()
    
    # test train split. stratify on prediction class
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_TRAIN_SPLIT, stratify=y)

    return X_train, X_test, y_train, y_test, labels


def build_model():
    '''
    Return a model that implements fit() and predict()
    '''

    # https://machinelearningmastery.com/cost-sensitive-svm-for-imbalanced-classification/#:~:text=Weighted%20SVM%20With%20Scikit%2DLearn,-The%20scikit%2Dlearn&text=The%20class_weight%20is%20a%20dictionary,calculation%20of%20the%20soft%20margin.&text=The%20class%20weighing%20can%20be,talking%20to%20subject%20matter%20experts.

    # assign class weights 0:non_NEOF, 1:NEOF
    weights = {0:1.0, 1:1.0}

    # model = SVC(kernel='linear', class_weight=weights)
    # model = SVC(kernel='poly', degree=3, class_weight=weights)
    # model = SVC(kernel='rbf', class_weight=weights)
    # model = AdaBoostClassifier(n_estimators=100, learning_rate=1)
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)

    logging.info(f'Built model {model}')
    return model


def train_model(model, X_train, y_train):
    # trains a model
    logging.info(f'Training model on X_train shape: {X_train.shape};\n\tLabels:')
    for label, count in y_train.value_counts().iteritems():
        logging.info(f'\t\t{label}: {count}')
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    # tests a model on the test set
    logging.info(f'Evauluating model on X_test shape: {X_test.shape};\n\tLabels:')
    for label, count in y_test.value_counts().iteritems():
        logging.info(f'\t\t{label}: {count}')
    logging.info('')

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test,y_pred)
    display_labels = model.classes_
    logging.info(pd.DataFrame(cm,
                index=[['True Class']*len(display_labels),[f'{x}' for x in display_labels]],
                columns=[['Predicted Class']*len(display_labels),[f'{x}' for x in display_labels]]))
    logging.info('')
    logging.info(classification_report(y_test,y_pred))


def save_model(model, path:str, now:str):
    filename = model.__class__.__name__ + '_' + now + '.pkl'
    path = os.path.join(path, filename)
    logging.info(f'Saving model {model} to {path}')
    pickle.dump(model, open(path, 'wb'))


def get_feature_importances(model, labels, top_n=10):
    '''
    Combine model feature importances
    INPUTS:
        model (sklearn model): model under test
        labels (list): list of labels returned from Pipeline
        top_n (int): number of highest impact features to return
    RETURNS:
        importances (pd.Series): list of top_n features, sorted by absolute value
    '''
    if isinstance(model, SVC):
        if model.kernel == 'linear':
            coefs = model.coef_[0]
            importances = pd.Series(coefs, index=labels)
            abs_sort = importances.abs().sort_values(ascending = False)
            importances = importances[abs_sort.index].head(top_n)
        else:
            logging.warn('WARNING: only an SVM with a linear kernel can retrieve feature importances')
            importances = None
    elif isinstance(model, AdaBoostClassifier):
        coefs = model.feature_importances_
        importances = pd.Series(coefs, index=labels)
        abs_sort = importances.abs().sort_values(ascending = False)
        importances = importances[abs_sort.index].head(top_n)
    elif isinstance(model, GradientBoostingClassifier):
        coefs = model.feature_importances_
        importances = pd.Series(coefs, index=labels)
        abs_sort = importances.abs().sort_values(ascending = False)
        importances = importances[abs_sort.index].head(top_n)
    else:
        logging.warn(f'WARNING: No get_feature_importances() case implemented for type {type(model)}')
        importances = None

    return importances


def start_logging(model, log_path, timestamp_str, console_print_level=logging.DEBUG):
    # create log filename
    logname = model.__class__.__name__ + '_' + timestamp_str + '.log'
    logpath = os.path.join(log_path, logname)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG) # log everything
    # create file handler which logs even debug messages
    fh = logging.FileHandler(logpath)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(console_print_level)
    # create formatter and add it to the handlers
    #formatter = logging.Formatter(' %(levelname)s: %(message)s')
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    root_logger.handlers = [] # must remove a default handler
    root_logger.addHandler(ch)
    root_logger.addHandler(fh)




def main():
    #### manual inputs
    TRAIN_DATA_PATH = '<PLACEHOLDER PARQUET PATH STRING>'
    PREDICTION_COLUMN = '<PLACEHOLDER TARGET COLUMN STRING'
    NONLOG_PARAM_PATH = '<PLACEHOLDER JSON PATH STRING>'
    LOG_PARAM_PATH = '<PLACEHOLDER JSON PATH STRING>'
    ENGINEERED_FEATURES_NUM_PATH = '<PLACEHOLDER JSON PATH STRING>'
    TRAIN_PIPELINE = True
    SAVE_PIPELINE = True
    PIPELINE_PATH = '<PLACEHOLDER PKL PATH STRING>'
    TEST_TRAIN_SPLIT = 0.20
    GRID_SEARCH = True
    SAVE_MODEL = True
    MODEL_PATH = '<PLACEHOLDER DIRECTORY PATH STRING>'

    # get timestamp
    now = datetime.datetime.now().isoformat(timespec="minutes").replace(":","_")

    ### build model - we do this first so that we will know the name of the model for logging
    model = build_model()

    #### load configuration file
    config = ConfigParser()
    config.read('<PLACEHOLDER CONFIG FILE STRING>')


    ### prepare logging
    start_logging(model, log_path=MODEL_PATH, timestamp_str=now)

    #### load data
    X_train, X_test, y_train, y_test, labels = preprocess_data(
        PREDICTION_COLUMN=PREDICTION_COLUMN,
        PARQUET_PATH=TRAIN_DATA_PATH,
        NO_OP_PATH=None,
        NUM_ATTRS_PATH=[NONLOG_PARAM_PATH, ENGINEERED_FEATURES_NUM_PATH],
        LOG_ATTRS_PATH=LOG_PARAM_PATH,
        CAT_ATTRS_PATH=None,
        ENCODED_ATTRS_PATH=None,
        TRAIN_PIPELINE=TRAIN_PIPELINE,
        SAVE_PIPELINE=SAVE_PIPELINE,
        PIPELINE_PATH=PIPELINE_PATH,
        TEST_TRAIN_SPLIT=TEST_TRAIN_SPLIT
        )
    
    #### train and evaluate
    model = train_model(model, X_train, y_train)
    evaluate_model(model, X_test, y_test)
    importances = get_feature_importances(model, labels, top_n=20)
    logging.info(f'FEATURE IMPORTANCES:\n{importances}')
    if SAVE_MODEL:
        save_model(model, MODEL_PATH, now)


if __name__ == '__main__':
    main()
