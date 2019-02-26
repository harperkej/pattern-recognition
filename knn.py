import math
import csv
import time

class KNearestNeighbor:

    def __init__(self, data_set_name):
        self.data_set_name = data_set_name
    
    def manhattan_distance(self, image1, image2):
        """Calculate the manhattan distance of two vectors."""
        distance = 0
        for i in range(len(image1)):
            if(i != 0):
                distance = distance + abs(int(image1[i]) - int(image2[i]))
        return distance

    def euclidian_distance(self, image1, image2):
        """Calculate the euclidian distance of two vectors."""
        distance = 0
        for i in range(len(image1)):
            if(i != 0):
                distance = distance + ((int(image1[i]) - int(image2[i]))** 2)
        distance = math.sqrt(distance)
        return distance

    def find_most_frequent_label(self, labels):
        """Finds the most frequent element in the list."""
        label_set = {}
        max_count = 0
        max_item = None
        for label in labels:
            if label not in label_set:
                label_set[label] = 1
            else:
                label_set[label]+=1
            if(label_set[label] > max_count):
                max_count = label_set[label]
                max_item = label
        return max_item

    def find_k_nearest_labels(self, labels, distances, k_param):
        """Finds k smallest distances and returns their respective labels"""
        k = 0
        k_nearest_labels = []
        while k < k_param:
            minimum = min(distances)
            position = distances.index(minimum)
            del distances[position]
            k_nearest_labels.append(labels[position])
            del labels[position]
            k = k + 1
        return k_nearest_labels

    def knn(self, distance_metric, k_parameter = 1):
        total_images = 0
        with open(self.data_set_name) as data_set:
            reader = csv.reader(data_set, delimiter='\t')
            outer_id = 1
            misclassified_images = 0
            k_nearest_labels_of_current_img = []
            for current_image in reader:
                total_images += 1
                inner_id = 1
                labels, distances = [], []
                with open(self.data_set_name) as inner_data_set:
                    inner_reader = csv.reader(inner_data_set, delimiter = '\t')
                    for image in inner_reader:
                        if inner_id != outer_id:
                            euclidian_dist = distance_metric(current_image, image)
                            labels.append(image[0])
                            distances.append(euclidian_dist)
                        inner_id =  inner_id + 1
                k_nearest_labels_of_current_img = self.find_k_nearest_labels(labels, distances, k_parameter)
                most_frequent_label = self.find_most_frequent_label(k_nearest_labels_of_current_img)
                outer_id = outer_id + 1
                if most_frequent_label != current_image[0]:
                    misclassified_images = misclassified_images + 1
            success_rate = 1 - misclassified_images / total_images
            print(success_rate)
            print(total_images)

knn = KNearestNeighbor('train.csv')
print('Starting .... ')
start_time = time.time()
knn.knn(knn.manhattan_distance, 1)
end_time = time.time()
print('Finished')
print('Took in total: ')
print((end_time - start_time) + ' seconds')
