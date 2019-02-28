import math
import csv
import time

class KNearestNeighbor:

    def __init__(self, test_dataset, train_dataset):
        self.test_dataset = test_dataset
        self.train_dataset = train_dataset
    
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
        with open(self.test_dataset) as test_dataset:
            reader = csv.reader(test_dataset, delimiter='\t')
            misclassified_images = 0
            k_nearest_labels_of_current_img = []
            for current_image in reader:
                total_images += 1
                labels, distances = [], []
                with open(self.train_dataset) as train_dataset:
                    inner_reader = csv.reader(train_dataset, delimiter = '\t')
                    for image in inner_reader:
                        euclidian_dist = distance_metric(current_image, image)
                        labels.append(image[0])
                        distances.append(euclidian_dist)
                k_nearest_labels_of_current_img = self.find_k_nearest_labels(labels, distances, k_parameter)
                most_frequent_label = self.find_most_frequent_label(k_nearest_labels_of_current_img)
                if most_frequent_label != current_image[0]:
                    misclassified_images = misclassified_images + 1
            success_rate = 1 - misclassified_images / total_images
            print(success_rate)
            print(total_images)
            print(misclassified_images)

knn = KNearestNeighbor('dev_test.csv','dev_train.csv')
print('Starting .... (with big dataset)')
start_time = time.time()
knn.knn(knn.manhattan_distance, 1) 
end_time = time.time()
print('Finished')
print('Took in total: ')
print(end_time - start_time)
