import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# plt.rcParams['text.usetex'] = True
# Create a dictionary with the dataset dimensions and corresponding precisions
data = {
    "Algorithm": [
        "ANN NB", "NB", "FTNB", "SVM", "DT", "Logistic Regression", "XG Boost",
        "ANN NB", "NB", "FTNB", "SVM", "DT", "Logistic Regression", "XG Boost",
        "ANN NB", "NB", "FTNB", "SVM", "DT", "Logistic Regression", "XG Boost"
    ],
    "Precision": [
        0.8936, 0.8588, 0.8588, 0.8977, 0.8379, 0.8959, 0.9021,  # Dataset 1
        0.5217, 0.1373, 0.1563, 0.3636, 0.1864, 0.2823, 0.3253,  # Dataset 2
        0.8936, 0.8033, 0.7867, 0.8851, 0.7673, 0.8695, 0.9146   # Dataset 3
    ],
    "Dataset Dimension": [
        25079, 25079, 25079, 25079, 25079, 25079, 25079,  # Dataset 1
        20495, 20495, 20495, 20495, 20495, 20495, 20495,  # Dataset 2
        23375, 23375, 23375, 23375, 23375, 23375, 23375   # Dataset 3
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Set style
sns.set(style="whitegrid")

# Create the plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Dataset Dimension', y='Precision', hue='Algorithm', marker='o')

# Adding titles and labels
plt.title('Precision vs. Dataset Dimension for Various Algorithms')
plt.xlabel('Dataset Dimension')
plt.ylabel('Precision')
plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.savefig('Precision_vs_Dimension.pdf', format='pdf', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
