from ucimlrepo import fetch_ucirepo

# fetch dataset
air_quality = fetch_ucirepo(id=360)

# data (as pandas dataframes)
X = air_quality.data.features
y = air_quality.data.targets

df=X

import pandas as pd
import numpy as np
import torch
from torch import Tensor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import k_means
import anfis
from membership import membershipfunction

# Replace -200 with NaN for better handling of missing data
df.replace(-200, pd.NA, inplace=True)

# Drop rows with missing data (originally indicated by -200)
df.dropna(inplace=True)

# Reset the index after dropping rows
df.reset_index(drop=True, inplace=True)

# Display the cleaned DataFrame
df.head()
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix
correlation_matrix = df[df.columns[2:]].corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cbar=True, square=True, vmin=-1 , vmax=1)
plt.title('Correlation Matrix Heatmap of All Features', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.show()


# Select features and target
features = ['CO(GT)','C6H6(GT)', 'PT08.S2(NMHC)']  # Example feature columns
target = 'PT08.S1(CO)'  # Target column

X = df[features].to_numpy()
Y = df[target].to_numpy()

# Normalize features and target
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

X = scaler_X.fit_transform(X)
Y = scaler_Y.fit_transform(Y.reshape(-1, 1)).flatten()

# Step 3: Split the data into train, validation, and test sets
x_train, x_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.4, random_state=73)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=73)


# Step 4: Define the RBF Model
class RBF():
    def __init__(self, inFeatures: int, outFeatures: int, nClusters: int) -> None:
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.nCluster = nClusters
        self.weights = None

        # Perform K-Means clustering to find centroids
        self.centroids, _, _ = k_means(x_train, n_clusters=self.nCluster)
        dists = []
        for i in range(self.centroids.shape[0]):
            for j in range(self.centroids.shape[0]):
                if i != j:
                    d = np.linalg.norm(self.centroids[i] - self.centroids[j])
                    dists.append(d)
        dMax = max(dists)
        spread = (dMax ** 2) / (self.nCluster)
        self.scale = -2 * (spread)

    def euclidean(self, X: Tensor, C: Tensor):
        cMat = torch.tensor(np.tile(C, (X.shape[0], 1)))
        diff = torch.norm(X - cMat, dim=1)
        return diff

    def train(self, X: Tensor, y: Tensor):
        phiList = []
        for c in self.centroids:
            r = self.euclidean(X, c)
            phi = torch.exp(r / self.scale)
            phiList.append(phi.unsqueeze(1).permute(1, 0))
        phiT = torch.cat(phiList)
        phis = phiT.permute(1, 0)
        weights = (torch.linalg.inv(phiT @ phis)) @ phiT @ y
        self.weights = weights

    def test(self, X: Tensor):
        phiList = []
        for c in self.centroids:
            r = self.euclidean(X, c)
            phi = torch.exp(r / self.scale)
            phiList.append(phi.unsqueeze(1).permute(1, 0))
        phiT = torch.cat(phiList)
        phis = phiT.permute(1, 0)
        yHat = phis @ self.weights
        return yHat

# Step 5: Train and test the RBF model
rbf = RBF(inFeatures=3, outFeatures=1, nClusters=5)

# Convert data to PyTorch tensors
x_train_torch = torch.tensor(x_train)
y_train_torch = torch.tensor(y_train).unsqueeze(1)
x_test_torch = torch.tensor(x_test)

# Train the RBF model
rbf.train(x_train_torch, y_train_torch)

# Test the RBF model
y_pred_rbf = rbf.test(x_test_torch).detach().numpy()

# Evaluate the RBF model
y_pred_rbf_denorm = scaler_Y.inverse_transform(y_pred_rbf)
y_test_denorm = scaler_Y.inverse_transform(y_test.reshape(-1, 1))

mse_rbf = mean_squared_error(y_test_denorm, y_pred_rbf_denorm)
r2_rbf = r2_score(y_test_denorm, y_pred_rbf_denorm)

print(f"RBF Model - Mean Squared Error: {mse_rbf}")
print(f"RBF Model - R-Squared: {r2_rbf}")

# Step 6: Define and train the ANFIS model
mf = [
    [['gaussmf', {'mean': np.mean(x_train[:, 0]), 'sigma': np.std(x_train[:, 0])}],
     ['gaussmf', {'mean': np.mean(x_train[:, 0]) - 1, 'sigma': np.std(x_train[:, 0]) * 1.5}]],
    [['gaussmf', {'mean': np.mean(x_train[:, 1]), 'sigma': np.std(x_train[:, 1])}],
     ['gaussmf', {'mean': np.mean(x_train[:, 1]) + 1, 'sigma': np.std(x_train[:, 1]) * 1.5}]],
    [['gaussmf', {'mean': np.mean(x_train[:, 2]), 'sigma': np.std(x_train[:, 2])}],
     ['gaussmf', {'mean': np.mean(x_train[:, 2]) - 1, 'sigma': np.std(x_train[:, 2]) * 1.5}]],
]

# Create the ANFIS model
mfc = membershipfunction.MemFuncs(mf)
anf = anfis.ANFIS(x_train, y_train, mfc)

# Train the ANFIS model
anf.trainHybridJangOffLine(epochs=20)

# Test the ANFIS model
y_pred_anfis = anfis.predict(anf , x_test)

# Evaluate the ANFIS model
y_pred_anfis_denorm = scaler_Y.inverse_transform(y_pred_anfis)
mse_anfis = mean_squared_error(y_test_denorm, y_pred_anfis_denorm)
r2_anfis = r2_score(y_test_denorm, y_pred_anfis_denorm)

print(f"ANFIS Model - Mean Squared Error: {mse_anfis}")
print(f"ANFIS Model - R-Squared: {r2_anfis}")

# Step 7: Compare the models
if mse_rbf < mse_anfis:
    print("The RBF model performed better with a lower MSE.")
else:
    print("The ANFIS model performed better with a lower MSE.")

if r2_rbf > r2_anfis:
    print("The RBF model performed better with a higher R-Squared.")
else:
    print("The ANFIS model performed better with a higher R-Squared.")
    

import matplotlib.pyplot as plt

# Plot RBF, ANFIS, and actual test results
plt.figure(figsize=(10, 5))

# Plot actual data
plt.plot(y_test_denorm, label='Actual Data', linestyle='-', marker='^', markersize=4, color='black', linewidth=1.5)

# Plot RBF predictions
plt.plot(y_pred_rbf_denorm, label='RBF Predictions', linestyle='dotted', marker='o', markersize=5, color='purple', linewidth=1.2)

# Plot ANFIS predictions
plt.plot(y_pred_anfis_denorm, label='ANFIS Predictions', linestyle='dashdot', marker='s', markersize=4, color='orange', linewidth=1.2)

# Add labels and legend
plt.title('Performance Comparison of Models', fontsize=13, fontweight='bold')
plt.xlabel('Sample Index', fontsize=11)
plt.ylabel('Target Value', fontsize=11)
plt.legend(fontsize=10, loc='best', frameon=True)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()

plt.show()
