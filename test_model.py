# test_model.py
import unittest
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.datasets import load_iris

class TestIrisModel(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Load model and data once for all tests"""
        cls.model_path = 'models/iris_model.pkl'
        cls.data_path = 'data/test_data.csv'
        
        # Load model
        with open(cls.model_path, 'rb') as f:
            cls.model = pickle.load(f)
        
        # Load test data
        cls.test_data = pd.read_csv(cls.data_path)
        cls.X_test = cls.test_data.drop('target', axis=1)
        cls.y_test = cls.test_data['target']
    
    def test_model_file_exists(self):
        """Test that model file exists"""
        self.assertTrue(os.path.exists(self.model_path), 
                       "Model file does not exist")
    
    def test_data_file_exists(self):
        """Test that test data file exists"""
        self.assertTrue(os.path.exists(self.data_path), 
                       "Test data file does not exist")
    
    def test_model_predictions(self):
        """Test that model can make predictions"""
        predictions = self.model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test),
                        "Number of predictions doesn't match test data")
    
    def test_prediction_values(self):
        """Test that predictions are valid iris classes (0, 1, 2)"""
        predictions = self.model.predict(self.X_test)
        unique_predictions = set(predictions)
        valid_classes = {0, 1, 2}
        self.assertTrue(unique_predictions.issubset(valid_classes),
                       "Invalid prediction values")
    
    def test_model_accuracy_threshold(self):
        """Test that model accuracy is above threshold"""
        predictions = self.model.predict(self.X_test)
        accuracy = np.mean(predictions == self.y_test)
        self.assertGreater(accuracy, 0.85, 
                          f"Model accuracy {accuracy:.2f} is below threshold 0.85")
    
    def test_data_shape(self):
        """Test that data has correct shape (4 features for iris)"""
        self.assertEqual(self.X_test.shape[1], 4,
                        "Test data should have 4 features")
    
    def test_data_validity(self):
        """Test that data contains no NaN values"""
        self.assertFalse(self.X_test.isnull().any().any(),
                        "Test data contains NaN values")
    
    def test_prediction_probabilities(self):
        """Test that model can output prediction probabilities"""
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(self.X_test)
            # Check that probabilities sum to 1
            prob_sums = probabilities.sum(axis=1)
            np.testing.assert_array_almost_equal(prob_sums, 
                                                 np.ones(len(probabilities)),
                                                 decimal=5)

if __name__ == '__main__':
    unittest.main()
