## REgressor
import unittest
from KNN.modelling import KNN_data_collection as knn
test_knn_regressor = knn.KNN("regressor",2)
test_knn_classifier = knn.KNN("classifier", 3)
import numpy as np
class TestDataCollection_regressor(unittest.TestCase):
    x_train = 0
    x_test = 0
    y_train = 0
    y_test = 0
    assert_xtrain = 0
    assert_xtest = 0
    assert_ytrain = 0
    assert_ytest = 0
    x =0
    y =0
    def setUpClass():
        test_knn_regressor = knn.KNN("regressor",2)
        
        
    def setUp(self):
        self.test_size = 0.25
        self.x = np.array([[6,250,100,3282,15],[6,250,88,3139,14.5], [4,122,86,2220,14],[4,116,90,2123,14]])
        self.y = np.array([12, 45, 11, 90])
        self.test_train_split_regressor_result = 0
        
    def test_train_test_split_regressor(self):
        
        test_knn_regressor.x = self.x
        test_knn_regressor.y = self.y
        
        self.x_train = int((1-self.test_size) * len(self.x))
        self.x_test = int(self.test_size * len(self.x))
        self.y_train = int((1-self.test_size) * len(self.y))
        self.y_test = int(self.test_size * len(self.y))
        
        
        test_knn_regressor.train_test_split(0.25)
        self.assert_xtrain = len(test_knn_regressor.x_train)
        self.assert_xtest = len(test_knn_regressor.x_test)
        self.assert_ytrain = len(test_knn_regressor.y_train)
        self.assert_ytest = len(test_knn_regressor.y_test)
    
    
        if ((self.assertEqual(self.x_train,self.assert_xtrain)) and (self.assertEqual(self.x_test,self.assert_xtest)) and (self.assertEqual(self.y_train,self.assert_ytrain)) and (self.assertEqual(self.y_test,self.assert_ytest))):
            test_train_split_regressor_result = 1
    
    def tearDown(self):
        print(self.x_train, self.x_test, self.y_train, self.y_test)
        print(self.assert_xtrain, self.assert_xtest, self.assert_ytrain, self.assert_ytest)
    def tearDownClass():
        print('Result successful')


## CLassifier
class TestDataCollection_classifier(unittest.TestCase):
    x_train = 0
    x_test = 0
    y_train = 0
    y_test = 0
    assert_xtrain = 0
    assert_xtest = 0
    assert_ytrain = 0
    assert_ytest = 0
    x =0
    y =0
    def setUpClass():
        test_knn_classifier = knn.KNN("classifier", 3)
    def setUp(self):
        self.test_size = 0.4
        self.x = np.array([[6.3,2.3,4.4,1.3],[6.8,3.2,5.9,2.3],[4.3,3,1.1,0.1],[3.5,3.1,1.0,0.5],[6.9,3.3,5.8,2.0]])
        self.y = np.array(['Iris-versicolor', 'Iris-virginica', 'Iris-setosa', 'Iris-setosa','Iris-virginica'])
        
    def test_train_test_split_classifier(self):
        
        test_knn_classifier.x = self.x
        test_knn_classifier.y = self.y
        
        self.x_train = int((1-self.test_size) * len(self.x))
        self.x_test = int(self.test_size * len(self.x))
        self.y_train = int((1-self.test_size) * len(self.y))
        self.y_test = int(self.test_size * len(self.y))
        
        
        
        test_knn_classifier.train_test_split(0.4)
        self.assert_xtrain = len(test_knn_classifier.x_train)
        self.assert_xtest = len(test_knn_classifier.x_test)
        self.assert_ytrain = len(test_knn_classifier.y_train)
        self.assert_ytest = len(test_knn_classifier.y_test)
        
        
        
        self.assertEqual(self.x_train, self.assert_xtrain)
        self.assertEqual(self.x_test,self.assert_xtest)
        self.assertEqual(self.y_train, self.assert_ytrain)
        self.assertEqual(self.y_test, self.assert_ytest)
    
    def tearDown(self):
        print(self.x_train, self.x_test, self.y_train, self.y_test)
        print(self.assert_xtrain,self.assert_xtest,self.assert_ytrain,self.assert_ytest)
    def tearDownClass():
        print('Result successfull')

unittest.main (argv =[''], verbosity=2, exit= False)      