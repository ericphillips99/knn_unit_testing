import unittest
from KNN.modelling import KNN_data_collection as knn
test_knn_regressor = knn.KNN("regressor",2)
test_knn_classifier = knn.KNN("classifier",4)
import numpy as np
        
class Testload_csv(unittest.TestCase):
    data=[]
    y_header, x_header = 0, 0
    assert_y_header = 0
    assert_x_header = 0
    assert_k = 0
    x,y =0,0
    
    def setUpClass():
        test_knn_regressor = knn.KNN("regressor",2)
    def setUp(self):
        self.data = np.arange(0,745).reshape(149,5)
        self.k = 2
    def test_load_csv(self):
        self.y_header = self.data[0,4]
        self.y = self.data[:,self.y_header] # counting the length of column header
        self.x = np.delete(self.data,self.y,axis=1).astype(float)
        
        test_knn_regressor.load_csv('data_banknote_authentication.txt', '-0.44699' )
        self.assert_y = test_knn_regressor.y
        self.assert_x = test_knn_regressor.x
        self.assert_k = test_knn_regressor.k
        
        #self.assertEqual(self.response,self.assert_response)
        self.assertEqual(np.size(self.y), np.size(self.assert_y))
        self.assertEqual(np.size(self.x), np.size(self.assert_x))
        self.assertEqual(self.assert_k,self.k)
    
    def tearDown(self):
        print(np.size(self.y_header),np.size(self.y),np.size(self.x))
    def tearDownClass():
        print('Result successfull')
    
    
class TestVariables(unittest.TestCase):
    x, y, assert_y, assert_x = 0, 0, 0, 0
    data = []
    def setUpClass():
        test_knn_classifier = knn.KNN("classifier",4)
    def setUp(self):
        self.data = np.arange(0,750).reshape(150,5)
        self.k = 4
        
    def test_varible_size(self):
        self.y_header = self.data[0,4]
        self.y = self.data[:,self.y_header]
        self.x = (np.delete(self.data, self.y, axis = 1))
        
        test_knn_classifier.load_csv('iris.csv', 'Species')
        
        self.assert_y = test_knn_classifier.y
        self.assert_x = test_knn_classifier.x
        
        self.assertIn(self.y_header, self.y)
        self.assertNotIn(self.x, self.y)
        self.assertNotIn(self.y, self.x)
    

unittest.main (argv =[''], verbosity=2, exit= False) 
