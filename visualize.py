import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Before clustering
# df = pd.read_csv("song_data.csv", header=None)
# df.columns = ["Energy", "Speechiness", "Instrumentalness"]
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(df["Energy"], df["Speechiness"], df["Instrumentalness"])
# ax.set_xlabel("Energy")
# ax.set_ylabel("Speechiness")
# ax.set_zlabel("Instrumentalness")
# plt.title("Scatterplot of features")

# After clustering
plt.figure()
df = pd.read_csv("./persistedData/distributed-50e-6c.csv")
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(df.x, df.y, df.z, c=df.c, cmap="viridis")
ax.set_xlabel("Energy")
ax.set_ylabel("Speechiness")
ax.set_zlabel("Instrumentalness")
plt.title("Clustered: Features with cluster labels")

# Create a color bar for the cluster labels
colorbar = plt.colorbar(scatter)
colorbar.set_label("Cluster")
plt.show()
