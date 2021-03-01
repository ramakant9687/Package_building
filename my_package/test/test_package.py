# from src.prediction import predict
from my_package import data_preprocessing, prediction, score_eval, train_model

# import prediction


def test_function():
    (
        Linear_Model_prediction,
        DT_Model_prediction,
        RF_Model_prediction,
    ) = prediction.predict()

    assert Linear_Model_prediction == 68628.19819848922

    assert DT_Model_prediction == 0.0

    assert RF_Model_prediction == 48244.85220207772


test_function()
