# PCA

# Step 1: Understanding Principal Component Analysis for feature extraction
# Importing the dataset

dataset = read.csv('google_review_ratings.csv')
dataset = dataset[, -ncol(dataset)]

# Splitting the dataset into the training set and test set
# install.packages('caTools')
library(caTools)
set.seed(629) # setting personal seed number
split = sample.split(dataset$User, SplitRatio=0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[, -c(1, 26)] = scale(training_set[, -c(1, 26)])
test_set[, -c(1, 26)] = scale(test_set[, -c(1, 26)])

# Step 2: Using preProcess function in R for PCA to extract principal components

# Applying PCA
# install.packages('caret')
library(caret)
# install.packages('e1071')
library(e1071)

pca = preProcess(x=training_set[, -c(1, 26)], method='pca', pcaComp=2) 
# pcaComp=2: end up with 2 new independent features

# PCA training set
training_set = predict(pca, training_set) 
# Apply pca transformation object on the original training set

# Move first column (dependent variable) to last column
training_set = training_set[c(2, 3, 1)] # old order 1 2 3 becomes 2 3 1

# Do the same for test set (in one step) to obtain PCA test set
test_set = predict(pca, test_set)[c(2, 3, 1)]

# Applying K means clustering
# Step 1: Set the number of clusters
set.seed(629)
k = 8 # Number of clusters
kmeans_model = kmeans(training_set[, 1:2], centers = k) # Apply K-means on PC1 and PC2

# Add cluster assignments to the training set
training_set$Cluster = as.factor(kmeans_model$cluster)

# Step 2: Visualize the clusters for the training set
library(ggplot2)

# Scatter plot of PC1 vs PC2, color-coded by clusters
ggplot(training_set, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point() +
  ggtitle("K-means Clustering on Training Set") +
  theme_minimal()

# Step 3: Assign test set users to the nearest cluster centroid
# Predict the cluster for each test set point based on the nearest centroid
test_clusters = apply(test_set[, 1:2], 1, function(x) {
  which.min(colSums((t(kmeans_model$centers) - x)^2))
})

# Add predicted clusters to the test set
test_set$Cluster = as.factor(test_clusters)

# Step 4: Visualize test set users with cluster assignments
# Overlay test set points onto the training set clusters
ggplot() +
  geom_point(data = training_set, aes(x = PC1, y = PC2, color = Cluster), alpha = 0.5) +
  geom_point(data = test_set, aes(x = PC1, y = PC2), color = "brown", shape = 17, size = 1) +
  ggtitle("Cluster Visualization with Training and Test Sets") +
  theme_minimal()

# To make the plot interactive to see the user# and cluster#:
# Step 1: install.packages("plotly")
library(plotly)

# Step 2: Add user numbers as a column for the hover text
training_set$User = rownames(training_set) # Ensure rownames represent user numbers
test_set$User = rownames(test_set)

# Step 3: Create an interactive scatter plot for the training set
cluster_colors = c("1"="blue", 
                   "2"="green", 
                   "3"="red", 
                   "4"="purple",
                   "5"="orange", 
                   "6"="pink", 
                   "7"="yellow",
                   "8"="cyan",
                   "Test Set"="brown") # ensures cluster colors remain consistent
p = plot_ly(
  data = training_set,
  x = ~PC1, 
  y = ~PC2, 
  type = 'scatter',
  mode = 'markers',
  color = ~Cluster,
  colors = cluster_colors,
  text = ~paste("User:", User), # Hover text to display user numbers
  hoverinfo = 'text'
) %>%
  layout(
    title = "K-means Clustering on Training Set",
    xaxis = list(title = "PC1"),
    yaxis = list(title = "PC2")
  )

# Display the interactive K-means Clustering on Training Set plot
p

# Including both the training and test sets in the visualization,
# add cluster and user information for test set
test_set$Cluster = "Test Set" # Distinguish test set points

# Combine training and test sets into one for visualization
combined_data = rbind(
  training_set,
  transform(test_set, Cluster="Test Set") # Label test points differently
)

# Create an interactive scatter plot for combined data
p = plot_ly(
  data = combined_data,
  x = ~PC1, 
  y = ~PC2, 
  type = 'scatter',
  mode = 'markers',
  color = ~Cluster,
  colors = cluster_colors,
  text = ~paste("User:", User, "<br>Cluster:", Cluster), # Hover text
  hoverinfo = 'text'
) %>%
  layout(
    title = "Cluster Visualization with Training and Test Sets",
    xaxis = list(title = "PC1"),
    yaxis = list(title = "PC2")
  )

# Display the interactive Cluster Visualization with Training and Test Sets plot
p # Simply puts places the test set points onto the graph with no assignment yet

# Predict clusters for test set users
test_clusters = apply(test_set[, 1:2], 1, function(x) {
  which.min(colSums((t(kmeans_model$centers) - x)^2)) # Using training set centroids
})
# Add the cluster assignments to the test set
test_set$AssignedCluster = as.factor(test_clusters)

# Plot assignment of test set users to clusters w.r.t its color
p = plot_ly() %>%
  # Add training set points
  add_trace(
    data = training_set,
    x = ~PC1,
    y = ~PC2,
    type = 'scatter',
    mode = 'markers',
    marker = list(size = 10, opacity = 0.1), # Style for training set points
    color = ~Cluster,
    colors = cluster_colors,
    text = ~paste("User:", User, "<br>Cluster:", Cluster),
    hoverinfo = 'text'
  ) %>%
  # Add test set points (color-coded by AssignedCluster)
  add_trace(
    data = test_set,
    x = ~PC1,
    y = ~PC2,
    type = 'scatter',
    mode = 'markers',
    marker = list(size = 6, opacity = 2), # Style for test set points
    color = ~AssignedCluster, # Use the assigned cluster for color
    text = ~paste("User:", User, "<br>Assigned Cluster:", AssignedCluster),
    hoverinfo = 'text'
  ) %>%
  layout(
    title = "Interactive Clustering: Training and Test Sets",
    xaxis = list(title = "PC1"),
    yaxis = list(title = "PC2")
  )

# Display the plot
p