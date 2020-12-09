import unittest
from KNN.modelling import KNN_data_collection as knn
import numpy as np

# I divided the test cases into classes to show classification and regression. Repeating them for both will be iterative. 
#So I believe this meets the criteria of TestClass with 2 test cases (Testload_csv,TestVariables ) with 4 asserts. I just differentiated it with 2 TestClasses

class Testload_csv(unittest.TestCase):
    data=[]
    y_header, x_header = 0, 0
    assert_y_header = 0
    assert_x_header = 0
    assert_k = 0
    x,y,test_knn_regressor =0,0, 0
    
    def setUpClass():
        Testload_csv.test_knn_regressor = knn.KNN("regressor",2)
    def setUp(self):
        self.data = np.arange(0,745).reshape(149,5)
        self.k = 2
    def test_load_csv(self):
        self.y_header = self.data[0,4]
        self.y = self.data[:,self.y_header] # counting the length of column header
        self.x = np.delete(self.data,self.y,axis=1).astype(float)
        
        Testload_csv.test_knn_regressor.load_csv('datasets/data_banknote_authentication.txt', '-0.44699' )
        self.assert_y = Testload_csv.test_knn_regressor.y
        self.assert_x = Testload_csv.test_knn_regressor.x
        self.assert_k = Testload_csv.test_knn_regressor.k
        
        self.assertEqual(np.size(self.y), np.size(self.assert_y))
        self.assertEqual(np.size(self.x), np.size(self.assert_x))
        self.assertEqual(self.assert_k,self.k)
        self.assertIsNotNone(self.y)
        
    
    def tearDown(self):
        print(np.size(self.y_header),np.size(self.y),np.size(self.x))
    def tearDownClass():
        print('Test Successful!')
    
    
class TestVariables(unittest.TestCase):
    x, y, assert_y, assert_x,test_knn_classifier = 0, 0, 0, 0, 0
    data = []
    def setUpClass():
        TestVariables.test_knn_classifier = knn.KNN("classifier",4)
    def setUp(self):
        self.data = np.arange(0,750).reshape(150,5)
        self.k = 4
        
    def test_varible_size(self):
        self.y_header = self.data[0,4]
        self.y = self.data[:,self.y_header]
        self.x = (np.delete(self.data, self.y, axis = 1))
        
        TestVariables.test_knn_classifier.load_csv('datasets/iris.csv', 'Species')
        
        self.assert_y = TestVariables.test_knn_classifier.y
        self.assert_x = TestVariables.test_knn_classifier.x
        
        self.assertIn(self.y_header, self.y)
        self.assertNotIn(self.x, self.y)
        self.assertNotIn(self.y, self.x)
        self.assertIsNotNone(self.y)
    
    def tearDown(self):
        print(np.size(self.y_header),np.size(self.y),np.size(self.x))
    def tearDownClass():
        print('Test Successful!')

unittest.main (argv =[''], verbosity=2, exit= False) 