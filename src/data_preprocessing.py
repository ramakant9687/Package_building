import logging
import os
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit

LOG_FILENAME = "logfile.log"
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH)

    **Download the housing data**

    This function downloads the housing data

    |
    """
    logging.info("Fetch Housing Data.....")
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    """load_housing_data(housing_path=HOUSING_PATH)

    **Read the data file**

    This function reads the csv file containing housing data

    |
    """
    logging.info("Load Housing Data.....")
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def income_cat(housing):
    """Function that create a categrical column on the basis of median housing data

    Args:
        df (DataFrame): Housing Dataframe

    Returns:
        Pandas DataFrame: Housing Dataframe including income category as feature
    """
    logging.info("Creating Income Category.....")
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    return housing


def housing_labels_(strat_train_set):
    """Function used to extract dependent variable for the training set

    Args:
        df (DataFrame): Stratified training dataset
    Returns:
        Pandas DataFrame: Housing label
    """
    logging.info("Extracting Housing Label.....")
    housing_labels = strat_train_set["median_house_value"].copy()
    return housing_labels


def correlation_(housing):
    """corr_matrix(housing)

    **Correlation Matrix**

    The function displayes correlation between Median House Value and Attributes

    The function also displays correlation between Median House Value and Attributes by combining Attributes

    Args:
        df (DataFrame): Houing dataset
    Returns:
        Printing Correlation Matrix

    """
    logging.info("Finding Correlation.....")
    corr_matrix = housing.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))


def stratified_split(housing):
    """Function which is used to split the housing dataset using Stratified Shuffle Split

    Args:
        df (DataFrame): Housing Dataset

    Returns:
        Pandas DataFrame: Stratified Training Set and Test Set
    """
    logging.info("StratifiedShuffleSplit.....")
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    return strat_train_set, strat_test_set


def imputer(housing):
    """Function used to impute the missing values in the dataframe

    Args:
        df (DataFrame):Housing Dataset

    Returns:
        Pandas DataFrame: New Housing Trained Dataset
    """
    logging.info("Imputing Missing Values.....")
    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(
        X, columns=housing_num.columns, index=housing.index
    )
    return housing_tr


def feature_eng2(housing_tr, housing):
    """Function used to create various features

    Args:
        df (DataFrame):Housing Dataset, Housing Trained Dataset


    Returns:
        Pandas DataFrame: Housing Prepared Dataset
    """
    logging.info("Coming up with new features.....")
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )
    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(
        pd.get_dummies(housing_cat, drop_first=True)
    )
    return housing_prepared


def random_forest_test_Data(strat_test_set):
    """Function which is used to create test dataframe for Random Forest Model

    Args:
        df (DataFrame): Stratifed Test Dataset

    Returns:
        Pandas DataFrame: X_test_prepared, y_test
    """
    logging.info("Creation of Test Dataset for Random Forest Model.....")
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    X_test_num = X_test.drop("ocean_proximity", axis=1)
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_test_num)
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )
    X_test_prepared = feature_eng2(X_test_prepared, X_test)
    return X_test_prepared, y_test


def rfdata():
    """Function which is used to create test dataframe for Random Forest Model

    Args:


    Returns:
        Pandas DataFrame: X_test_prepared, y_test
    """
    fetch_housing_data()
    housing = load_housing_data()
    housing = income_cat(housing)
    strat_train_set, strat_test_set = stratified_split(housing)
    return random_forest_test_Data(strat_test_set)


def data_preprocess():
    """Main Function for Data Preprocessing.

    Args:


    Returns:
        Pandas DataFrame: Housing_prepared Dataset, Housing_labels Dataset
    """
    logging.info("Running Main function for Data_Preprocess module.....")
    fetch_housing_data()
    housing = load_housing_data()
    housing = income_cat(housing)
    strat_train_set, strat_test_set = stratified_split(housing)
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()
    housing_tr = imputer(housing)
    housing_prepared = feature_eng2(housing_tr, housing)
    return housing_prepared, housing_labels
