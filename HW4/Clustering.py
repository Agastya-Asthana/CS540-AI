import csv
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy


class Cluster:
    def __int__(self, clusters=[], points=[]):
        self.clusters = clusters
        self.points = points

    taken = 0

    def __str__(self):
        return f"C: {self.clusters} P: {self.points} T: {self.taken}"


def load_data(filepath):
    with open(filepath) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        file = []
        for row in reader:
            file.append(row)
    return file


def calc_features(row):
    feature = np.empty((6,), dtype=np.int64)
    feature[0] = int(row[6])  # attack
    feature[1] = int(row[8])  # sp. attack
    feature[2] = int(row[10])  # speed
    feature[3] = int(row[7])  # defense
    feature[4] = int(row[9])  # sp. defense
    feature[5] = int(row[5])  # HP
    return feature


def Get_Distance_Matrix(features):
    distance_matrix = np.empty((len(features), len(features)))
    for i in range(len(features)):
        for j in range(len(features)):
            if distance_matrix[j][i] != 0:
                distance_matrix[i][j] = distance_matrix[j][i]
            else:
                distance_matrix[i][j] = abs(np.linalg.norm(features[i]) - np.linalg.norm(features[j]))
    return distance_matrix


def Calc_Dist_Clusters(cluster1, cluster2, distance_matrix):
    max = int(np.inf * -1)
    for point in cluster1.points:
        for point2 in cluster2.points:
            if max < distance_matrix[point][point2]:
                max = distance_matrix[point][point2]
    return max


# Run to find the closests clusters in the given Z array
def Find_Closests_Clusters(Clusters, distance_matrix):
    min = int(np.inf)
    index1 = 0  # index of cluster 1
    index2 = 0  # indec of cluster 2
    for i in range(len(Clusters)):
        if Clusters[i].taken == 1:  # if taken
            continue

        for j in range(i + 1, len(Clusters)):
            if Clusters[j].taken == 1:  # if taken
                continue

    return 0;


def hac(features):
    Z = []  # Big Z array
    Clusters = []  # Big array containing all clusters
    distance_matrix = Get_Distance_Matrix(features)  # Distance from every point to every other (n x n)
    # Fill Clusters with each cluster being the individual points themselves
    for i in range(len(features)):
        Clusters.append(Cluster([i], [i]))
    """for i in range(len(features)):
        row = np.empty((5,), dtype=np.int64)  # each row in Z - 5 for the last column being a taken variable
        # print(i)
        row[0] = i
        row[1] = i
        row[2] = 0
        row[3] = 1
        row[4] = 0  # taken or not
        Z.append(row)"""
    # x = np.linalg.norm(features[1])
    # print(x)


def imshow_hac(z):
    fig = plt.figure()
    deno = hierarchy.dendrogram(z, orientation="top")
    plt.show()


pokemon = load_data("Pokemon.csv")
features = []
for i in range(1, len(pokemon)):
    features.append(calc_features(pokemon[i]))
Z = hierarchy.linkage(features, 'complete')
imshow_hac(Z)
