import csv
import random
import math
import time

class KMeans:

    def __init__(self, training_dataset):
        self.training_dataset = training_dataset

    def euclidian_distance(self, image1, image2):
        """Calculate the euclidian distance of two vectors."""
        distance = 0
        length = min(len(image1), len(image2))
        for i in range(length):
            if(i != 0):
                distance = distance + ((int(image1[i]) - int(image2[i]))** 2)
        distance = math.sqrt(distance)
        return distance

    def init_random_centroids(self, k_clusters, data):
        centroids = {}
        """Generate random k_clusters cluster centers (centroids)"""
        for i in range(k_clusters):
            cluster_center = random.randint(0, len(data) - 1)
            if cluster_center not in centroids.keys():
                centroids[cluster_center] = data[cluster_center]
        return list(centroids.values())

    def read_training_dataset(self):
        with open(self.training_dataset) as training_dataset:
            reader = csv.reader(training_dataset, delimiter = '\t')
            data = list(reader)
            # Remove the label.
            for item in data:
                del item[0]
            return data

    def recompute_centroids(self, clusters):
        """
        Computes the centroids of the given clusters.
        The centroid of a cluster is simply an array of the same size as 
        the datapoints of the cluster but it's element at position i is the
        average of the elements of all datapoints at position i.
        In case the cluster happens to be empty; that is no datapoint is assigne to the 
        cluster, then the centroid of this cluster is simply an array of zeros. 
        """
        centroids = []
        for i in range(len(clusters)):
            centroid = []
            cluster = clusters[i]
            if (len(cluster) > 0):
                for j in range(len(cluster[0])):
                    sum = 0
                    for k in range(len(cluster)):
                        sum += float(cluster[k][j])
                    mean = sum / len(cluster)
                    centroid.append(mean)
                centroids.append(centroid)
            else:
                centroid = []
                for i in range(784):
                    centroid.append(0)
                centroids.append(centroid)
        return centroids

    def assign_datapoints_to_clusters(self, data, centroids, k_clusters):
        """
        Assigns datapoints to the cluster whose centroid is closest according to 
        Euclidian distance.
        """
        clusters = []
        for i in range(k_clusters):
            clusters.append([])
        for i in range(len(data)):
            distances = []
            image = data[i]
            for centroid in centroids:
                distance = self.euclidian_distance(image, centroid)
                distances.append(distance)
            centroid = 0
            distance = 0
            for j in range(len(distances)):
                if distances[j] >= distance:
                    distance = distances[j]
                    centroid = j
            clusters[centroid].append(image)
        return clusters

    def have_centroids_converged(self, centroids, recalculated_centroids, k_clusters):
        """
        Checks wether two centroids are the same or not.
        This is to check wether the centroids are changing or not between
        iterations.
        """
        converged = True
        for i in range(k_clusters):
            centroid = centroids[i]
            recalculated_centroid = recalculated_centroids[i]
            for j in range(len(centroid)):
                if centroid[j] != recalculated_centroid[j]:
                    converged = False
            if (converged == False):
                break
        return converged

    def single_linkage_cluster_distance(self, cluster1, cluster2):
        """
        Calculates the distance between two clusters according to the 
        single linkage method.
        That is, it checks the distance between every distinct pair of
        two clusters and selects the smallest.
        """
        min_distance = 100**100
        i = 0
        while (i < len(cluster1) - 1):
            j = i+1
            while (j < len(cluster2)):
                distance = self.euclidian_distance(cluster1[i],cluster2[j])
                if(distance < min_distance):
                    min_distance = distance
                j += 1
            i += 1
        return min_distance

    def calculate_dunn_index(self, clusters):
        """
        Calculates the dunn index of the clustering.
        """
        # Calculate the diameter of each cluster
        diameters = []
        for cluster in clusters:
            max_distance = 0
            i = 0
            while (i < len(cluster) - 1):
                j = i+1
                while (j < len(cluster)):
                    distance = self.euclidian_distance(cluster[i], cluster[j])
                    if (distance > max_distance):
                        max_distance = distance
                    j += 1
                i += 1
            diameters.append(max_distance)
        max_diameter = max(diameters)
        # Calculate the single linkage distance of earch pair of clusters and select the minimum.
        i = 0
        min_distance = 100*100
        while (i < (len(clusters) - 1)):
            j = i+1
            while (j < len(clusters)):
                single_linkage_cluster_distance = self.single_linkage_cluster_distance(clusters[i], clusters[j])
                if (single_linkage_cluster_distance < min_distance):
                    min_distance = single_linkage_cluster_distance
                j += 1
            i += 1
        return min_distance / max_diameter

    def c_index(self, clusters, data):
        """
        Calculate the C-index
        """
        #Calculate gama and alpha.
        gama = 0
        alpha = 0
        for cluster in clusters:
            size_of_cluster = len(cluster)
            if (size_of_cluster != 0):
                alpha += size_of_cluster * (size_of_cluster - 1) / 2
            i = 0
            while (i < (len(cluster) - 1)):
                j = i+1
                while (j < len(cluster)):
                    gama += self.euclidian_distance(cluster[i], cluster[j])
                    j += 1
                i += 1
        # Now, alpha min and max distances among all distinct paris 
        # of datapoints  of dataset are found
        min_distances = []
        max_distances = []
        i = 0
        while (i < len(data) - 1):
            j = i+1
            while (j < len(data)):
                distance = self.euclidian_distance(data[i], data[j])
                # If the array of minimum distances is not yet filled with
                # alpha distances, just insert the distance
                if (len(min_distances) < alpha):
                    min_distances.append(distance)
                # If the array of minimum distances has already 
                else:
                    min_distances.sort(reverse =True)
                    if (distance < min_distances[0]):
                        min_distances[0] = distance
                # The same approach is applied for maximum distances.
                if (len(max_distances) < alpha):
                    max_distances.append(distance)
                else:
                    max_distances.sort()
                    if (distance > max_distances[0]):
                        max_distances[0] = distance
                j += 1
            i += 1
        alpha_min = sum(min_distances)
        alpha_max = sum(max_distances)
        return (gama - alpha_min) / (alpha_max - alpha_min)

    def kmeans(self, k_clusters, iterations):
        data = self.read_training_dataset()
        # Generate random centroids
        centroids = self.init_random_centroids(k_clusters, data)
        # Initially asssign datapoints to the closest random generated centroids
        clusters = self.assign_datapoints_to_clusters(data, centroids, k_clusters)

        converged = False

        for i in range(iterations):
            # Recompute centroids
            recomputed_centroids = self.recompute_centroids(clusters)
            # Reassign data to the newly calculated centroids
            clusters = self.assign_datapoints_to_clusters(data, recomputed_centroids, k_clusters)
            converged = self.have_centroids_converged(centroids, recomputed_centroids, k_clusters)
            if(i > iterations or converged):
                break
            centroids = recomputed_centroids

        if(converged):
            print('Recomputing centroids (clustering the datapoints) finished because centroids converged.')
        else:
            print('Recomputing centroids (clustering the datapoints) finished because preconfigured number of iterations is over.')
        dunn_index = self.calculate_dunn_index(clusters)
        print('Dun index = ')
        print(dunn_index)
        c_index = self.c_index(clusters, data)
        print('C index = ')
        print(c_index)

kmeans = KMeans('dev_test.csv')
start_time = time.time()
print('For k = 5 ---------------------')
kmeans.kmeans(5, 30)
end_time = time.time()
print('Finished')
print('Took in total: ')
print(end_time - start_time)
print('For k = 5 ---------------------')

kmeans = KMeans('dev_test.csv')
start_time = time.time()
print('For k = 7 ---------------------')
kmeans.kmeans(7, 30)
end_time = time.time()
print('Finished')
print('Took in total: ')
print(end_time - start_time)
print('For k = 7 ---------------------')

kmeans = KMeans('dev_test.csv')
start_time = time.time()
print('For k = 10 ---------------------')
kmeans.kmeans(10, 30)
end_time = time.time()
print('Finished')
print('Took in total: ')
print(end_time - start_time)
print('For k = 10 ---------------------')


kmeans = KMeans('dev_test.csv')
start_time = time.time()
print('For k = 12 ---------------------')
kmeans.kmeans(12, 30)
end_time = time.time()
print('Finished')
print('Took in total: ')
print(end_time - start_time)
print('For k = 12 ---------------------')

kmeans = KMeans('dev_test.csv')
start_time = time.time()
print('For k = 15 ---------------------')
kmeans.kmeans(15, 30)
end_time = time.time()
print('Finished')
print('Took in total: ')
print(end_time - start_time)
print('For k = 15 ---------------------')