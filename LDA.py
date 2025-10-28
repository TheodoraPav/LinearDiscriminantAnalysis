import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neural_network import MLPClassifier

# --- Styling for plots ---
sns.set_context("talk")
sns.set_style("whitegrid")

# --- Load dataset ---
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
class_labels = data.target_names
y_labels = pd.Categorical.from_codes(y, class_labels)

# --- Standardize features ---
X_scaled = StandardScaler().fit_transform(X)

# --- Step 1: Feature Discriminability Ratio ---
def discriminability_ratio(X, y):
    df = pd.DataFrame(X)
    df['class'] = y
    ratios = []
    for col in df.columns[:-1]:
        overall_mean = df[col].mean()
        classes = df['class'].unique()
        sb, sw = 0, 0
        for c in classes:
            group = df[df['class'] == c][col]
            n_c = len(group)
            mean_c = group.mean()
            sb += n_c * (mean_c - overall_mean)**2
            sw += ((group - mean_c)**2).sum()
        ratios.append(sb / sw)
    return ratios

ratios = discriminability_ratio(X_scaled, y)
plt.figure(figsize=(8,5))
sns.barplot(x=data.feature_names, y=ratios, palette="crest")
plt.title("Feature Discriminability Ratio (Between / Within Class Variance)")
plt.ylabel("Discriminability Ratio")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# --- Step 3: Apply LDA ---
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)
explained_var = lda.explained_variance_ratio_ * 100

# --- Step 4: LDA Scree Plot ---
plt.figure(figsize=(6,4))
sns.barplot(x=[f"LD{i+1}" for i in range(len(explained_var))],
            y=explained_var, palette="mako")
plt.plot(range(len(explained_var)), np.cumsum(explained_var), color='black', marker='o', linestyle='--', label="Cumulative %")
plt.title("Scree Plot of Linear Discriminants (LDA)")
plt.ylabel("Explained Discriminative Power (%)")
plt.legend()
plt.tight_layout()
plt.show()

# --- Step 5: 2D LDA Projection ---
plt.figure(figsize=(7,6))
for i, label in enumerate(class_labels):
    plt.scatter(X_lda[y==i,0], X_lda[y==i,1], label=label, alpha=0.8)
plt.title("Projection onto LDA Space (LD1–LD2)")
plt.xlabel(f"LD1 ({explained_var[0]:.1f}% info)")
plt.ylabel(f"LD2 ({explained_var[1]:.1f}% info)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# --- Step 6: Comparison Before & After LDA (first 2 features for raw space) ---
fig, axes = plt.subplots(1,2, figsize=(12,5))
sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1], hue=y_labels, palette="deep", ax=axes[0])
axes[0].set_title("Original Feature Space (first 2 features)")
axes[0].set_xlabel(data.feature_names[0])
axes[0].set_ylabel(data.feature_names[1])
sns.scatterplot(x=X_lda[:,0], y=X_lda[:,1], hue=y_labels, palette="deep", ax=axes[1])
axes[1].set_title("LDA Transformed Space (LD1–LD2)")
axes[1].set_xlabel("LD1")
axes[1].set_ylabel("LD2")
plt.tight_layout()
plt.show()
