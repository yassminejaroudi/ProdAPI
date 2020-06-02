import math
import numpy as np
from regression_model.predict import make_prediction
from regression_model.processing.data_management import load_dataset
from sklearn.metrics import mean_squared_error, r2_score

def test_make_single_prediction():
    # Given
    test_data = load_dataset(file_name='test.csv')
    single_test_json = test_data[0:1].to_json(orient='records')

    # When
    subject = make_prediction(input_data=single_test_json)

    # Then
    assert subject is not None
    #print('test mse: {}'.format(int(
        #mean_squared_error(test_data['Valeur_fonciere'], subject.get('predictions')[0]))))
    #print('test rmse: {}'.format(int(
        #np.sqrt(mean_squared_error(test_data['Valeur_fonciere'], subject.get('predictions')[0])))))
    #print('test r2: {}'.format(
        #r2_score(test_data['Valeur_fonciere'], subject.get('predictions')[0])))
    assert isinstance(subject.get('predictions')[0], float)
    #assert math.ceil(subject.get('predictions')[0]) == 112476


def test_make_multiple_predictions():
    # Given
    test_data = load_dataset(file_name='test.csv')
    original_data_length = len(test_data)
    multiple_test_json = test_data.to_json(orient='records')

    # When
    subject = make_prediction(input_data=multiple_test_json)
    # Then
    assert subject is not None
    # Then
    assert subject is not None
    
    
    
    #assert len(subject.get('predictions')) == 119700

    # We expect some rows to be filtered out
    assert len(subject.get('predictions')) == original_data_length
