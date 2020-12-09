import unittest
from KNN.modelling import KNN_data_collection as knn
from KNN.assessment import model_metrics as mm
from KNN.modelling import generate_predictions as gp
import numpy as np

# TEST CASE 1
class TestClassificationMetrics(unittest.TestCase):
    accuracy, to_predict, assert_accuracy,assert_misclassify,assert_num_correct,num_incorrect,assert_num_incorrect =0,0, 0,0, 0, 0, 0
    test_knn_classifier = 0 #Class or Static Variable, will be accessed using Class name
    def setUpClass():
        TestClassificationMetrics.test_knn_classifier = knn.KNN("classifier",4)
        TestClassificationMetrics.test_knn_classifier.load_csv('datasets/Iris.csv','Species')
        TestClassificationMetrics.test_knn_classifier.train_test_split(0.2)
        print('from setUpClass')
    def setUp(self):
        
        
        self.to_predict=np.array([[6.3,2.3,4.4,1.3],[6.8,3.2,5.9,2.3],[4.3,3,1.1,0.1]])
        gp.generate_predictions(TestClassificationMetrics.test_knn_classifier,self.to_predict,'all')
        
    def test_accuracy(self):
        self.accuracy = 0.6666666666666666
        TestClassificationMetrics.assert_accuracy = mm.model_accuracy(['Iris-virginica', 'Iris-virginica', 'Iris-setosa'],gp.generate_predictions(TestClassificationMetrics.test_knn_classifier,self.to_predict,'all'))

        self.assertEqual(self.accuracy, self.assert_accuracy)
    def test_model_misclassification(self):
        self.misclassify = 0.3333333333333333
        TestClassificationMetrics.assert_misclassify = mm.model_misclassification(['Iris-virginica', 'Iris-virginica', 'Iris-setosa'],gp.generate_predictions(TestClassificationMetrics.test_knn_classifier,self.to_predict,'all'))

        self.assertEqual(self.assert_misclassify, self.misclassify)
        
    def test_model_num_correct(self):
        self.num_correct = 2
        TestClassificationMetrics.assert_num_correct = mm.model_num_correct(['Iris-virginica', 'Iris-virginica', 'Iris-setosa'],gp.generate_predictions(TestClassificationMetrics.test_knn_classifier,self.to_predict,'all'))
    
        self.assertEqual(self.assert_num_correct, self.num_correct)
        
    def test_model_num_incorrect(self):
        self.num_incorrect = 2
        TestClassificationMetrics.assert_num_incorrect = mm.model_num_incorrect(['Iris-virginica', 'Iris-virginica', 'Iris-setosa'],gp.generate_predictions(TestClassificationMetrics.test_knn_classifier,self.to_predict,'all'))
        self.assertNotEqual(self.assert_num_incorrect, self.num_incorrect)
    
    def tearDown(self):
        print('Test Success!!!')
    def tearDownClass():
        print("Results Summary", TestClassificationMetrics.assert_accuracy,TestClassificationMetrics.assert_misclassify,TestClassificationMetrics.assert_num_correct, TestClassificationMetrics.assert_num_incorrect)

# TEST CASE 2
class TestRegressionMetrics(unittest.TestCase):
    rsme, mpg_to_predict, assert_rmse,assert_mse,assert_mae,assert_mape =0,0, 0,0, 0, 0
    test_knn_regressor, mae, mape,actual_mpg,predicted_mpg, mse = 0, 0, 0, 0, 0, 0 #Class or Static Variable, will be accessed using Class name
    def setUpClass():
        TestRegressionMetrics.test_knn_regressor = knn.KNN("regressor",7)
        TestRegressionMetrics.test_knn_regressor.load_csv('datasets/auto_mpg.csv','mpg')
        TestRegressionMetrics.test_knn_regressor.train_test_split(0.25)
        print('from setUpClass')
    
    def setUp(self):
        
        self.actual_mpg=[19,18,23,28]
        self.predicted_mpg= gp.generate_predictions(TestRegressionMetrics.test_knn_regressor,self.mpg_to_predict,'train')
    
    def test_model_rmse(self):
        self.rmse = 3.195772079431429
        TestRegressionMetrics.assert_rmse = mm.model_rmse(self.actual_mpg,self.predicted_mpg)
        self.assertEqual(TestRegressionMetrics.assert_rmse,self.rmse)
        
    def test_model_mae(self):
        self.mae = 2.500000000000001
        TestRegressionMetrics.assert_mae = mm.model_mae(self.actual_mpg,self.predicted_mpg)
        
        self.assertEqual(self.mae, TestRegressionMetrics.assert_mae)
    def test_model_mape(self):
        self.mape = 10.17706338828438
        TestRegressionMetrics.assert_mape = mm.model_mape(self.actual_mpg,self.predicted_mpg)
        
        self.assertEqual(self.mape, TestRegressionMetrics.assert_mape)
        
    def test_model_mse(self):
        self.mse = 10.212959183673478
        TestRegressionMetrics.assert_mse = mm.model_mse(self.actual_mpg,self.predicted_mpg)
        
        self.assertEqual(self.mse, TestRegressionMetrics.assert_mse)
    def tearDown(self):
        print('Test Success!!!')
    def tearDownClass():
        print("Results Summary - Regressor:", TestRegressionMetrics.assert_rmse,TestRegressionMetrics.assert_mae, 
              TestRegressionMetrics.assert_mape,TestRegressionMetrics.assert_mse)
unittest.main (argv =[''], verbosity=2, exit= False) 