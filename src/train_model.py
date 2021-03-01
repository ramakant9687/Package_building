import logging
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

import data_preprocessing

LOG_FILENAME = "logfile.log"
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)


def linear_model_(housing_prepared, housing_labels):
    """Function used to train linear model

    Args:
        df (DataFrame):Housing_prepared Dataset, Housing_labels Dataset

    Returns:
        Pandas DataFrame: Linear Model
    """
    logging.info("Training the Linear Model....")
    lin_reg = LinearRegression()
    model1 = lin_reg.fit(housing_prepared, housing_labels)
    return model1


def dtreg(housing_prepared, housing_labels):
    """Function used to train Decision Tree model

    Args:
        df (DataFrame):Housing_prepared Dataset, Housing_labels Dataset

    Returns:
        Pandas DataFrame: Decision Tree Model
    """
    logging.info("Training the Decision Tree Model....")
    tree_reg = DecisionTreeRegressor(random_state=42)
    model2 = tree_reg.fit(housing_prepared, housing_labels)
    return model2


def rnd_forest(housing_prepared, housing_labels):
    """Function used to train Random Forest model using Grid Search as Hyperparameter Tuning

    Args:
        df (DataFrame):Housing_prepared Dataset, Housing_labels Dataset

    Returns:
        Pandas DataFrame: Random Forest Model
    """

    logging.info("Training the Random Forest Model....")
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {
            "bootstrap": [False],
            "n_estimators": [3, 10],
            "max_features": [2, 3, 4],
        },
    ]
    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(housing_prepared, housing_labels)
    final_model = grid_search.best_estimator_

    return final_model


def save_picklefile():
    """Function used to Save the pickle files of all the models.

    """
    logging.info("Saving the pickle File....")
    linear, dt, rnd = model_train()

    Pkl_Filename = "Pickle_Linear_Model.pkl"
    with open(Pkl_Filename, "wb") as file:
        pickle.dump(linear, file)

    Pkl_Filename = "Pickle_DT_Model.pkl"
    with open(Pkl_Filename, "wb") as file:
        pickle.dump(dt, file)

    Pkl_Filename = "Pickle_RF_Model.pkl"
    with open(Pkl_Filename, "wb") as file:
        pickle.dump(rnd, file)


def model_train():
    """Main Function to train all the three models.

    Returns:
        Pandas DataFrame: Linear Model, Decision Tree Model, Random Forest Model.
    """
    logging.info("Running the Main function for training model....")
    housing_prepared, housing_labels = data_preprocessing.data_preprocess()
    linear = linear_model_(housing_prepared, housing_labels)
    dt = dtreg(housing_prepared, housing_labels)
    rnd = rnd_forest(housing_prepared, housing_labels)
    return linear, dt, rnd
