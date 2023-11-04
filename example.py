import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
# Assume vr_smote.py is in the same directory
from vr_smote import vr_smote

def create_imbalanced_data(n_samples=1000, n_features=20, n_classes=2, weights=(0.9, 0.1)):
    """
    Creates an imbalanced dataset with the given specifications.
    """
    X, y = make_classification(n_samples=n_samples,
                               n_features=n_features,
                               n_classes=n_classes,
                               weights=list(weights),
                               flip_y=0,
                               random_state=42)
    return X, y

def main():
    # Create an imbalanced dataset
    X, y = create_imbalanced_data()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Apply VR-SMOTE to the training data
    X_train_balanced, y_train_balanced = vr_smote(np.hstack((X_train, y_train.reshape(-1, 1))))

    # Initialize the classifier
    classifier = RandomForestClassifier(random_state=42)

    # Train the classifier on the original imbalanced dataset
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    print("Classification report for the original imbalanced dataset:")
    print(classification_report(y_test, predictions))

    # Train the classifier on the balanced dataset
    classifier.fit(X_train_balanced[:, :-1], X_train_balanced[:, -1])
    predictions_balanced = classifier.predict(X_test)
    print("\nClassification report for the dataset balanced with VR-SMOTE:")
    print(classification_report(y_test, predictions_balanced))

if __name__ == "__main__":
    main()
