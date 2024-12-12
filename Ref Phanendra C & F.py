#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder


# In[16]:


# Load dataset
data = pd.read_csv("Wine.csv")


# In[17]:


data.info()


# In[18]:


# Normalize numerical data
scaler = StandardScaler()
data[['Alcohol', 'Malic_Acid', 'Ash']] = scaler.fit_transform(
    data[['Alcohol', 'Malic_Acid', 'Ash']]
)


# In[19]:


# Histogram/Bar Chart
plt.figure(figsize=(8, 5))
sns.histplot(data['Alcohol'], kde=True, color='skyblue')
plt.title('Distribution of Alcohol Content')
plt.xlabel('Alcohol (Standardized)')
plt.ylabel('Frequency')
plt.show()


# In[20]:


# K-means clustering with elbow and silhouette method
range_clusters = range(1, 11)
distortions = []

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data[['Alcohol', 'Malic_Acid', 'Ash']])
    distortions.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range_clusters, distortions, marker='o', linestyle='--', color='purple')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.grid(True)
plt.show()

optimal_clusters = 3  # Determined from the elbow plot
kmeans_model = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans_model.fit_predict(data[['Alcohol', 'Malic_Acid', 'Ash']])
data['Cluster'] = clusters

silhouette_avg = silhouette_score(data[['Alcohol', 'Malic_Acid', 'Ash']], clusters)
print(f"Silhouette Score: {silhouette_avg}")


# In[21]:


# Scatter plot for clusters
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=data['Alcohol'], y=data['Malic_Acid'], hue=data['Cluster'], palette='viridis', 
    size=data['Ash'], sizes=(20, 200), alpha=0.8
)
plt.title('K-Means Clustering Results')
plt.xlabel('Alcohol (Standardized)')
plt.ylabel('Malic Acid (Standardized)')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[22]:


# Line fitting (Linear Regression)
model = LinearRegression()
X = data[['Ash']].values
y = data['Alcohol'].values
model.fit(X, y)

# Predictions and visualization
predictions = model.predict(X)

plt.figure(figsize=(8, 5))
sns.scatterplot(x=data['Ash'], y=data['Alcohol'], label='Actual Data', color='blue', alpha=0.6)
sns.lineplot(x=data['Ash'], y=predictions, color='red', label='Fitted Line', linewidth=2)
plt.title('Alcohol vs Ash (Linear Regression)')
plt.xlabel('Ash (Standardized)')
plt.ylabel('Alcohol (Standardized)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


# In[23]:


# Correlation matrix and heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(
    data.corr(), annot=True, cmap='magma', fmt='.2f', cbar_kws={'label': 'Correlation Coefficient'}
)
plt.title('Correlation Heatmap')
plt.show()


# In[ ]:




