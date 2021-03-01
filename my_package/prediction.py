import data_preprocessing
import train_model


def predict():
    housing_prepared, housing_labels = data_preprocessing.data_preprocess()
    linear, dt, rnd = train_model.model_train()
    Linear_Model_prediction = linear.predict(housing_prepared)
    DT_Model_prediction = dt.predict(housing_prepared)
    X_test_prepared, y_test = data_preprocessing.rfdata()
    RF_Model_prediction = rnd.predict(X_test_prepared)
    return Linear_Model_prediction, DT_Model_prediction, RF_Model_prediction
