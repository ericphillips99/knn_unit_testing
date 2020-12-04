from KNN_data_collection import *

class GetPredictions(KNN):
    def euclidean_distance(point1,point2):
        distance=0
        for i in range(len(point1)):
            distance+=(old_point[i]-new_point[i])**2
        return np.sqrt(distance)

    
