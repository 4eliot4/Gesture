# Import required libraries
import pickle  # For saving/loading data and models
import matplotlib.pyplot as plt  # For plotting confusion matrix
import seaborn as sns  # For heatmap visualization
import numpy as np  # For numerical operations

# Import project-specific utilities and ML tools
from utils import *  # Custom utility functions (e.g., valid_labels)
from sklearn.ensemble import RandomForestClassifier  # ML model
from sklearn.model_selection import train_test_split  # For train/test split (optional)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix  # Metrics
from collections import Counter  # For counting label occurrences



# Load preprocessed data from pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Filter out data points with length > 64 to ensure only one hand is present
# Only keep data with valid labels
data_dict_filtered = {
    'data': [item for item, label in zip(data_dict['data'], data_dict['labels']) if len(item) <= 64 and label in valid_labels],
    'labels': [label for item, label in zip(data_dict['data'], data_dict['labels']) if len(item) <= 64 and label in valid_labels],
    'source': [source for item, label, source in zip(data_dict['data'], data_dict['labels'], data_dict['source']) if len(item) <= 64 and label in valid_labels]
}

# Find the maximum length among all remaining data points for padding
max_length = max(len(item) for item in data_dict_filtered["data"])

padded_data = []

# Pad or truncate each data point to ensure uniform length
# (Padding not strictly necessary for one hand, but useful for future two-hand support)
for item in data_dict_filtered["data"]:
    if len(item) < max_length:
        # Pad shorter sequences with zeros
        padded_item = np.pad(item, (0, max_length - len(item)), 'constant')
    else:
        # Truncate longer sequences to max_length
        padded_item = item[:max_length]
    padded_data.append(padded_item)

# Convert data and labels to numpy arrays for ML processing
data = np.array(padded_data)
labels = np.asarray(data_dict_filtered['labels'])
source = np.asarray(data_dict_filtered['source'])

# Print data distribution for each source and label
unique_sources = np.unique(source)
print("Data distribution per source and label:")
for src in unique_sources:
    src_labels = labels[source == src]
    label_counts = Counter(src_labels)
    print(f"Source {src}:")
    for label, count in label_counts.items():
        print(f"  Label {label}: {count} data points")
print()

# Select which source to use for validation (test set)
validation_source = 3

# Split data based on source
x_train = data[source != validation_source]
y_train = labels[source != validation_source]


# Prepare test set from selected validation source
x_test = data[source == validation_source]
y_test = labels[source == validation_source]

# Alternative: use sklearn's train_test_split for random split (commented out)
# x_train, x_test, y_train, y_test = train_test_split(
#     data, 
#     labels, 
#     test_size=0.2,  # 20% for testing
#     random_state=42,  # Ensures reproducibility
#     stratify=labels  # Maintain balanced class distribution
# )

# Print shape and info about training and test sets
print(f"Length of a data point in x_train: {len(x_train[0])}")
print(f"Length of a data point in x_test: {len(x_test[0])}")
print("Data is pre-processed and formatted for training.")

# Initialize and train the Random Forest Classifier
print("Training the RandomForestClassifier...")
model = RandomForestClassifier()
model.fit(x_train, y_train)
print("Training done.")

# Make predictions on test set
y_predict = model.predict(x_test)

# Calculate various performance metrics
accuracy = accuracy_score(y_test, y_predict)
precision = precision_score(y_test, y_predict, average='weighted')
recall = recall_score(y_test, y_predict, average='weighted')
f1 = f1_score(y_test, y_predict, average='weighted')

# Print performance metrics
print('Accuracy: {:.2f}%'.format(accuracy * 100))
print('Precision: {:.2f}%'.format(precision * 100))
print('Recall: {:.2f}%'.format(recall * 100))
print('F1 Score: {:.2f}%'.format(f1 * 100))

# Generate and plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_predict)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

# Save confusion matrix visualization
cm_filename = 'confusion_matrix.png'
print(f'Saving the confusion matrix as {cm_filename}.')
plt.savefig(cm_filename)

# Save the trained model to a pickle file for later inference
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)