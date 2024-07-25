import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create DataFrame for easier plotting
train_df = pd.DataFrame(X_train, columns=iris.feature_names)
train_df["set"] = "Train"
train_df["species"] = [iris.target_names[i] for i in y_train]

test_df = pd.DataFrame(X_test, columns=iris.feature_names)
test_df["set"] = "Test"
test_df["species"] = [iris.target_names[i] for i in y_test]

# Combine both datasets into one DataFrame for plotting
full_df = pd.concat([train_df, test_df])


# Function to plot class distribution comparison as proportions
def plot_class_proportion(data):
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )  # Use subplots to directly create a figure and axes
    # Get the total counts for each category in each dataset to use for normalization
    class_counts = data.groupby(["set", "species"]).size().unstack(fill_value=0)
    class_proportions = class_counts.div(class_counts.sum(axis=1), axis=0)
    # Plot the proportions using the axes created above
    class_proportions.plot(
        kind="bar",
        stacked=True,
        color=["#1f77b4", "#ff7f0e", "#2ca02c"],
        alpha=0.75,
        ax=ax,
    )
    ax.set_title("Proportional Class Distribution in Training and Test Sets")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Proportion")
    ax.legend(title="Species", loc="upper right")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.show()


# Plot class distribution as proportions
plot_class_proportion(full_df)
