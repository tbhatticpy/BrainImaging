import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import f_oneway, spearmanr
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier



plt.style.use('default')



#%% Task 1: Load and inspect node strength data (10 points)

"""
Here, we are gonna look at a clinical dataset (modified for this task).
In this dataset, MEG data has been recorded from subjects labeled as either:
   1. healthy
   2. suffering from major depressive disorder (MDD)
   3. suffering from post-traumatic stress disorder (PTSD)
   
From these subjects, MEG data was recorded, source-reconstructed into 400 parcels,
    and phase synchrony between parcels was estimated in 32 frequencies using the iPLV.   
To reduce data size, you are given the node strength of iPLV rather than 
    the edge strengths.
    
Load the node strength data and the labels that you downloaded from the shared folder. 
  

Run a statistical test that is suited for comparing 3 (or more) groups at once.
Correct the p_values for multiple comparisons with Benjamini-Hochberg method.

Plot the mean strength for each of the three groups as a function of frequency
   (with logarithmic x-axis). Add a legend.
Indicate on the plot for which frequencies the test was significant (p<0.05) after correction.
         
"""

DATA_DIR = 'data'
NODE_STRENGTH_DIR = os.path.join(DATA_DIR, 'node_strength.npy')
LABELS_DIR = os.path.join(DATA_DIR, 'labels.csv')
SYMPTOMS_DIR = os.path.join(DATA_DIR, 'symptoms.csv')

# Load the data
node_strength = np.load(NODE_STRENGTH_DIR)
labels = pd.read_csv(LABELS_DIR, header=None).squeeze().values
symptoms = pd.read_csv(SYMPTOMS_DIR, sep=';')

freqs = np.array([
        2.1,  2.5,  2.9,  3.3,  3.7,  4.15, 4.8,  5.4,  5.9,
        6.6,  7.4,  8.1,  9.0,  9.8, 10.9, 11.9, 13.1, 14.8,
       16.3, 17.8, 19.7, 21.6, 23.7, 26.6, 28.7, 31.8, 34.5 ,
       37.9, 42.5, 46.9, 52.1, 59.3 ])

group_names = ['HC', 'MDD', 'PTSD']
node_strength_avg = node_strength.mean(axis=2)

# ANOVA for each frequency
p_values = []
for i in range(node_strength_avg.shape[1]):
    group_data = [node_strength_avg[labels == j, i] for j in range(3)]
    stat, p = f_oneway(*group_data)
    p_values.append(p)

p_values = np.array(p_values)

# Correct for multiple comparisons using Benjamini-Hochberg
_, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')

plt.figure(figsize=(12, 6))
for i, group in enumerate(group_names):
    mean_strength = node_strength_avg[labels == i].mean(axis=0)
    plt.plot(freqs, mean_strength, label=group)

significant_freqs = freqs[p_corrected < 0.05]
print("Significant frequencies after correction (p < 0.05):", significant_freqs)
significant_y = node_strength_avg.mean(axis=0)[p_corrected < 0.05]
plt.scatter(significant_freqs, significant_y, color='red', marker='*', label='p < 0.05')

plt.xscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Mean Node Strength')
plt.title('Mean Node Strength vs Frequency')
plt.legend()
plt.grid(True)
plt.show()


#%% Task 2: Inspect symptoms (6 points)


"""
Load the symptom data.
For each of the two symptoms, make violinplots for the 3 groups.

"""

symptoms['Group'] = labels
symptoms['Group'] = symptoms['Group'].map({0: 'HC', 1: 'MDD', 2: 'PTSD'})

plt.figure(figsize=(14, 6))
# Color palette
palette = sns.color_palette("Set2", n_colors=3)

# BDI
plt.subplot(1, 2, 1)
sns.violinplot(x='Group', y='BDI', data=symptoms, hue='Group', palette=palette, legend=False)
plt.title('BDI Scores by Group')
plt.xlabel('Group')
plt.ylabel('BDI Score')

# PCL
plt.subplot(1, 2, 2)
sns.violinplot(x='Group', y='PCL', data=symptoms, hue='Group', palette=palette, legend=False)
plt.title('PCL Scores by Group')
plt.xlabel('Group')
plt.ylabel('PCL Score')

plt.tight_layout()
plt.show()




#%% Task 3: Correlate symptoms with node strength. (10 points)

"""
For each frequency, compute the correlation of subjects' mean node strength 
   with each of the two symptoms. Which test is more appropriate here?
      
Plot the correlations with the symptoms as a function of frequency (log x-axis).
For each symptom, mark the frequencies where the test was significant (p < 0.05).

Use the Benjaminini-Hochberg method to control for multiple comparisons
  (for each symptom separately) and repeat the plot with the new significants.

Describe your findings.

"""


correlations_bdi = []
p_values_bdi = []
correlations_pcl = []
p_values_pcl = []

# Spearman correlation for each frequency
for i in range(node_strength_avg.shape[1]):
    mean_strength = node_strength_avg[:, i]

    # BDI
    corr_bdi, p_bdi = spearmanr(mean_strength, symptoms['BDI'])
    correlations_bdi.append(corr_bdi)
    p_values_bdi.append(p_bdi)

    # PCL
    corr_pcl, p_pcl = spearmanr(mean_strength, symptoms['PCL'])
    correlations_pcl.append(corr_pcl)
    p_values_pcl.append(p_pcl)

correlations_bdi = np.array(correlations_bdi)
p_values_bdi = np.array(p_values_bdi)
correlations_pcl = np.array(correlations_pcl)
p_values_pcl = np.array(p_values_pcl)

# Benjamini-Hochberg correction
_, p_corrected_bdi, _, _ = multipletests(p_values_bdi, method='fdr_bh')
_, p_corrected_pcl, _, _ = multipletests(p_values_pcl, method='fdr_bh')

plt.figure(figsize=(14, 6))

# BDI
plt.subplot(1, 2, 1)
plt.plot(freqs, correlations_bdi, label='BDI Correlation')
plt.scatter(freqs[p_values_bdi < 0.05], correlations_bdi[p_values_bdi < 0.05],
            color='red', marker='*', label='Significant (p < 0.05)')
plt.scatter(freqs[p_corrected_bdi < 0.05], correlations_bdi[p_corrected_bdi < 0.05],
            color='blue', marker='o', label='Corrected Significant (p < 0.05)')
plt.xscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Spearman Correlation')
plt.title('Correlation with BDI vs Frequency')
plt.legend()
plt.grid(True)

# PCL
plt.subplot(1, 2, 2)
plt.plot(freqs, correlations_pcl, label='PCL Correlation')
plt.scatter(freqs[p_values_pcl < 0.05], correlations_pcl[p_values_pcl < 0.05],
            color='red', marker='*', label='Significant (p < 0.05)')
plt.scatter(freqs[p_corrected_pcl < 0.05], correlations_pcl[p_corrected_pcl < 0.05],
            color='blue', marker='o', label='Corrected Significant (p < 0.05)')
plt.xscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Spearman Correlation')
plt.title('Correlation with PCL vs Frequency')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#%% Task 4: Supervised learning: Classification (14 points)


"""
Now we will run a classifier on our data.
Use the node strength values (or a subset and/or means of it) as features.

In this task, it is sufficient to split the data into training and
    test sets, no validation set will be used.   
    
Choose a classifier from sklearn library and train it on the training set,
    then test it on the test set. Report the classification accuracy on the 
    training and test sets.
Plot and annotate the 3-way confusion matrix.

Try to select features so that you get > 80% accuracy on test data.

If this were real data, how useful would the classifier be for diagnosing patients?

"""

X = node_strength_avg
y = labels

# 80-20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Classification report
report = classification_report(y_test, y_test_pred, target_names=group_names)

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=group_names)
plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('3-Way Confusion Matrix')
plt.show()

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")
print("Classification Report:\n")
print(report)



#%% Task 5: Systematic evaluation of classifiers (10 points)

"""
In addition to the classifier you used in task 4, choose two others.
Run all 3 classifiers at least 20 times, using the same features,
    then plot the mean and stdev of accuracy (on test set) for each classifier. 
Which one performs best?

"""


# Define classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

# Store results
train_results = {name: [] for name in classifiers.keys()}
test_results = {name: [] for name in classifiers.keys()}
classification_reports = {name: [] for name in classifiers.keys()}

# Run each classifier 20 times
for _ in range(20):
    # Split the data with random shuffling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        train_results[name].append(train_accuracy)
        test_results[name].append(test_accuracy)
        report = classification_report(y_test, y_test_pred, target_names=group_names, output_dict=True, zero_division=0)
        classification_reports[name].append(report)

# Calculate mean and std of accuracies
mean_train_accuracies = {name: np.mean(accs) for name, accs in train_results.items()}
mean_test_accuracies = {name: np.mean(accs) for name, accs in test_results.items()}

mean_classification_reports = {}
for name, reports in classification_reports.items():
    mean_report = {}
    # Average metrics for each class and overall
    for key in reports[0].keys():
        if isinstance(reports[0][key], dict):
            mean_report[key] = {
                metric: np.mean([rep[key][metric] for rep in reports]) for metric in reports[0][key].keys()
            }
        else:
            mean_report[key] = np.mean([rep[key] for rep in reports])
    mean_classification_reports[name] = mean_report

# Plotting the results
plt.figure(figsize=(10, 6))
plt.bar(mean_test_accuracies.keys(), mean_test_accuracies.values(),
        yerr=[np.std(accs) for accs in test_results.values()], capsize=10, color=['red', 'green', 'blue'])
plt.ylabel('Test Accuracy')
plt.title('Mean and Standard Deviation of Test Accuracy (20 Runs)')
plt.grid(axis='y')
plt.show()

# Train/test accuracy and mean classification report
for name in classifiers.keys():
    print(f"\n{name}")
    print(f"Mean Training Accuracy: {mean_train_accuracies[name] * 100:.2f}%")
    print(f"Mean Test Accuracy: {mean_test_accuracies[name] * 100:.2f}%\n")

    print("Mean Classification Report:")
    report = mean_classification_reports[name]
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            print(f"  {label}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.2f}")
        else:
            print(f"  {label}: {metrics:.2f}")



#%% Baseline Approaches

"""
This is an additional task. I want to implement some non-ML baseline classifiers for comparison.
Scikit-learn offers the following 2 strategies for baseline DummyClassifier
1. 'stratified': Predicts labels by respecting the training set’s class distribution (randomly).
2. 'most_frequent': Always predicts the most frequent class from the training set.
3. 'uniform': Predicts each class randomly with equal probability.
4. 'constant': Always predicts a constant class label (specified via the 'constant' parameter).

resource: https://medium.com/@preethi_prakash/understanding-baseline-models-in-machine-learning-3ed94f03d645

"""

dummy_strategies = {
    'stratified': DummyClassifier(strategy='stratified', random_state=42),
    'most_frequent': DummyClassifier(strategy='most_frequent'),
    'uniform': DummyClassifier(strategy='uniform', random_state=42),
    'constant(HC)': DummyClassifier(strategy='constant', constant=0)    # For constant, I will make the algo always predict HC.
}

train_accuracies = []
test_accuracies = []
strategy_names = []

for name, dummy_clf in dummy_strategies.items():
    dummy_clf.fit(X_train, y_train)

    # Predictions on both train and test
    y_train_pred = dummy_clf.predict(X_train)
    y_test_pred = dummy_clf.predict(X_test)

    # Accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # Store results
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    strategy_names.append(name)

    # Print results
    print(f"\nDummy strategy: {name}")
    print(f"  Training Accuracy: {train_acc * 100:.2f}%")
    print(f"  Test Accuracy:     {test_acc * 100:.2f}%")

    print("  Classification Report (Test):")
    print(classification_report(y_test, y_test_pred, target_names=group_names, zero_division=0))

plt.figure(figsize=(12, 5))

# Training Accuracy
plt.subplot(1, 2, 1)
plt.bar(strategy_names, train_accuracies, color=['red', 'green', 'blue', 'yellow'])
plt.title("Dummy Classifiers: Training Accuracy")
plt.ylabel("Accuracy")
plt.ylim([0, 1.0])
for i, acc in enumerate(train_accuracies):
    plt.text(i, acc+0.01, f"{acc*100:.1f}%", ha='center')

# Test Accuracy
plt.subplot(1, 2, 2)
plt.bar(strategy_names, test_accuracies, color=['red', 'green', 'blue', 'yellow'])
plt.title("Dummy Classifiers: Test Accuracy")
plt.ylabel("Accuracy")
plt.ylim([0, 1.0])
for i, acc in enumerate(test_accuracies):
    plt.text(i, acc+0.01, f"{acc*100:.1f}%", ha='center')

plt.tight_layout()
plt.show()
