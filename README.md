# Obesity Classification Using CURE Algorithm

The "CURE Obesity Classifier" project aims to analyze a dataset containing BMI (Body Mass Index) and Age information to classify individuals into different weight categories (e.g., underweight, normal weight, obese) using the CURE (Clustering Using Representatives) Algorithm. Additionally, this project demonstrates how CURE outperforms K-means clustering for this problem statement.

# Applicability of CURE Algorithm

The CURE algorithm is particularly applicable in this project due to its ability to handle clusters of arbitrary shape and size. Since BMI and Age data may not necessarily form spherical clusters, CURE's flexibility in representing clusters using a small set of representative points makes it suitable for this analysis. Additionally, CURE is robust to outliers, which may be present in real-world BMI and Age datasets.

# CODE EXPLAINATION

1. getDistanceFromRepresentatives

Calculates the minimum distance from a given point to any of the representative points of a cluster.

```
def getDistanceFromRepresentatives(point, representativePoints_shifted):
    minimumDistance = float("inf")
    for repr_point in representativePoints_shifted:
        distance = getDistance(point, repr_point)
        if distance < minimumDistance:
            minimumDistance = distance
    return minimumDistance
```

Purpose: This function helps in assigning a point to the nearest cluster based on the shifted representative points.

2. computeCentroid

Computes the centroid (geometric center) of a cluster of points.

```
def computeCentroid(initialCluster):
    x = 0
    y = 0
    for point in initialCluster:
        x += point[0]
        y += point[1]
    numberOfPoints = len(initialCluster)
    return (x / numberOfPoints, y / numberOfPoints)

```

Purpose: To find the center of a cluster, which is used to shift representative points towards the centroid.

3. findRepresentativePoints

Selects n well-scattered representative points from a cluster.

```
def findRepresentativePoints(initialCluster):
    representativePoints = []
    representativePoints.append(list(initialCluster[0]))
    for i in range(n - 1):
        maximumDistance = float("-inf")
        for point in initialCluster:
            if point in representativePoints:
                continue
            minimumDistance = float("inf")
            for representativePoint in representativePoints:
                distance = getDistance(point, representativePoint)
                if distance < minimumDistance:
                    minimumDistance = distance
            if minimumDistance > maximumDistance:
                candidateRepresentativePoint = point
                maximumDistance = minimumDistance
        representativePoints.append(list(candidateRepresentativePoint))
    return representativePoints
```

Purpose: To select a set of points that are representative of the cluster’s shape and spread.

4. getDistance

Calculates the Euclidean distance between two points.

```
def getDistance(point1, point2):
    distance = (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
    return math.sqrt(distance)
```

Purpose: To measure the distance between two points, which is used in various parts of the clustering process.

5. clusterDistance

Calculates the minimum distance between any two points from two different clusters.

```
def clusterDistance(cluster1, cluster2):
    minimumDistance = float("inf")
    for point1 in cluster1:
        for point2 in cluster2:
            if minimumDistance > getDistance(point1, point2):
                minimumDistance = getDistance(point1, point2)
    return minimumDistance
```

Purpose: To determine the closeness of two clusters during hierarchical clustering.

6. formClusters_heirarchical

Performs hierarchical clustering on the sample data until the desired number of clusters k is achieved.

```
def formClusters_heirarchical(sampleData):
    clusters = [[i] for i in sampleData]
    iters = len(sampleData) - k
    for iter in range(iters):
        min = float("inf")
        for i in range(0, len(clusters) - 1):
            for j in range(i + 1, len(clusters)):
                if min > clusterDistance(clusters[i], clusters[j]):
                    min = clusterDistance(clusters[i], clusters[j])
                    c1 = i
                    c2 = j
        clusters[c1].extend(clusters[c2])
        del clusters[c2]
    return clusters
```

Purpose: To reduce the initial dataset into k clusters using a hierarchical approach.

# Main Execution Flow

## Reading Input Data:

Reads sample and complete data files along with parameters for clustering.

```
sampleDataFile = open(sys.argv[1]).readlines()
completeDataFile = open(sys.argv[2]).readlines()
k = int(sys.argv[3])
n = int(sys.argv[4])
p = float(sys.argv[5])
outputFileName = sys.argv[6]
```

## Parsing Data:

Converts the read data into lists of tuples representing points.

```
completeData = []
for line in completeDataFile:
    line = line.split(",")
    completeData.append((float(line[0]), float(line[1])))

sampleData = []
for line in sampleDataFile:
    line = line.split(",")
    sampleData.append((float(line[0]), float(line[1])))
sampleData = sorted(sampleData, key=lambda x: (x[0], x[1]))

```

## Hierarchical Clustering:

Forms initial clusters using hierarchical clustering on the sample data.

```
initialClusters = formClusters_heirarchical(sampleData)

```

## Finding and Shifting Representative Points:

For each cluster, finds representative points and shifts them towards the centroid.

```
# For plotting initial clusters
initialClusterAssignments = []
for clusterId, initialCluster in enumerate(initialClusters):
    for point in initialCluster:
        initialClusterAssignments.append((point, clusterId))

# Find representative points and shift them
for initialCluster in initialClusters:
    representivePoints = findRepresentativePoints(initialCluster)
    representativePointsList.append(representivePoints)
    shiftedRepresentativePoints = []
    centroid = computeCentroid(initialCluster)
    for representativePoint in representivePoints:
        shiftX = (centroid[0] - representativePoint[0]) * p
        shiftY = (centroid[1] - representativePoint[1]) * p
        shiftedRepresentativePoints.append((representativePoint[0] + shiftX, representativePoint[1] + shiftY))
    representativePoints_shifted.append(shiftedRepresentativePoints)

```

## Assigning Clusters:

Assigns each point in the complete dataset to the nearest cluster based on the representative points.

```
outputPointList = []
for point in completeData:
    minimumDistance = float("inf")
    for clusterNum in range(k):
        distance = getDistanceFromRepresentatives(point, representativePoints_shifted[clusterNum])
        if distance < minimumDistance:
            minimumDistance = distance
            clusterId = clusterNum
    outputPointList.append((point, clusterId))
```

## Writing Output:

Saves the clustering results to a specified output file.

```
w = open(outputFileName, 'w')
for point in outputPointList:
    w.write(str(point[0][0]) + "," + str(point[0][1]) + "," + str(point[1]) + "\n")
w.close()
```

## Plotting the Results:

Uses matplotlib to create a scatter plot of the points colored by their assigned cluster. One for initial clusters and one for final clusters

```
# Plot initial clusters
colors = plt.cm.get_cmap("tab10", k)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for point, clusterId in initialClusterAssignments:
    plt.scatter(point[0], point[1], color=colors(clusterId))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Initial Clusters')

# Plot final clusters
plt.subplot(1, 2, 2)
for point, clusterId in outputPointList:
    plt.scatter(point[0], point[1], color=colors(clusterId))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Final CURE Clusters')

plt.show()
```

# Overview of the CURE Algorithm Implementation

1. **Sampling:** Initially, a sample of the data is used to create clusters.
2. **Hierarchical Clustering:** The sample data undergoes hierarchical clustering to form k clusters.
3. **Representative Points:** For each cluster, n representative points are chosen that best capture the geometry of the cluster.
4. **Point Shifting:** These representative points are then shifted towards the cluster's centroid by a fraction p to reduce the effect of outliers.
5. **Assigning Clusters:** Each point in the complete dataset is assigned to the nearest cluster based on the minimum distance to the shifted representative points.
6. **Visualization:** Finally, the clustered points are plotted using different colors to visualize the clustering results.

# HOW TO RUN:

1. Clone the repo

```
   git clone https://github.com/Keshav-shar/CUREObesityClassifier.git
   cd CUREObesityClassifier
```

2. Install pandas, matplotlib and (scikit-learn for k-means clustring in K-Means_and_CURE_comparison.py)

```
   pip install pandas
   pip install matplotlib
   pip install scikit-learn
```

3. Ensure you have complete_age_bmi_data.csv file containing your complete data. Run the script to generate sample_age_bmi_data.csv:

```
   python sample_data_generation.py
```

4. Open the cure_algorithm_with_plots.py script and update the parameters as needed. Run the script below, replace sample_age_bmi_data.csv, complete_age_bmi_data.csv, and output.csv with your provided file names,in case of modification.
   Adjust the parameters (3, 10, 0.5), (representing number of clusters,Representatives in each cluster and shifting offset) according to your requirements.

```
   python CURE_cluster_generation.py sample_age_bmi_data.csv complete_age_bmi_data.csv 3 10 0.5 output.csv
```

5. After running the script, two plots will be displayed: one showing the initial clusters and the other showing the final clusters.

6. To run K-Means_and_CURE_comparison.py to compare the result of K-Means vs CURE run the following script. Make sure to replace sample_age_bmi_data.csv, complete_age_bmi_data.csv, and output.csv with your actual file names. Adjust the parameters (3, 10, 0.5) according to your requirements.

```
python K-Means_and_CURE_comparison.py sample_age_bmi_data.csv complete_age_bmi_data.csv 3 10 0.5 output_cure.csv

```
7. After running this script, two plots will be displayed: one showing the K-means Clusters and the other showing the Final CURE clusters for complete_age_bmi_data.csv.

