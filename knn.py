import math
import csv

class KNearestNeighbor:

    def __init__(self, data_set_name):
        self.data_set_name = data_set_name

    def manhattan_distance(self, image1, image2):
        distance = 0
        for i in range(len(image1)):
            if(i != 0):
                distance = distance + abs(int(image1[i]) - int(image2[i]))
        return distance

    def euclidian_distance(self, image1, image2):
        distance = 0
        for i in range(len(image1)):
            if(i != 0):
                distance = distance + ((int(image1[i]) - int(image2[i]))** 2)
        distance = math.sqrt(distance)
        return distance

    def knn(self, distance_metric, k_parameters = [1]):
        with open(self.data_set_name) as data_set:
            reader = csv.reader(data_set, delimiter='\t')
            outer_id = 1
            misclassified_images = 0
            for current_image in reader:
                inner_id = 1
                label, distance = [], []

                with open(self.data_set_name) as inner_data_set:
                    inner_reader = csv.reader(inner_data_set, delimiter = '\t')
                    for image in inner_reader:
                        if inner_id != outer_id:
                            euclidian_dist = distance_metric(current_image, image)
                            #change it to work with whatever k param
                            if euclidian_dist < distance:
                                label = image[0]
                                distance = euclidian_dist
                        inner_id =  inner_id + 1
                outer_id = outer_id + 1
                # Change it calculate the error rate with whatever k param
                if label != current_image[0]:
                    misclassified_images = misclassified_images + 1
            print(misclassified_images / 300)

knn = KNearestNeighbor('test.csv')
knn.knn(knn.manhattan_distance)