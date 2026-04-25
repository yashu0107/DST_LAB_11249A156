import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. Load dataset
# Ensure the file 'Mall_Customers.csv' is in the same directory
data = pd.read_csv("Mall_Customers.csv")

# 2. Select numerical features
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# 3. Feature scaling (Crucial for PCA and KMeans)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply PCA (Reduce 3D data to 2D for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# 5. Elbow Method to find optimal K
wcss = [] # List to store "Within-Cluster Sum of Squares"
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.show()

# 6. Apply KMeans with optimal K (Based on the elbow, usually 4 or 5)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
clusters = kmeans.fit_predict(X_pca)

# 7. Plot Clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', edgecolors='k')
plt.title("K-Means Clustering with PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label='Cluster ID')
plt.show()
