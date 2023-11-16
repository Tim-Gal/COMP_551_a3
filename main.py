import numpy as np
from datasets import load_dataset
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer


class CustomNaiveBayes:

    def __init__(self):
        self.class_priors = []
        self.feature_likelihoods = []
        self.class_prior_probs = {}  # Prior probabilities for each class
        self.feature_probs = defaultdict(dict)  # Probabilities of features per class
        self.unique_classes = set()  # Unique classes in the dataset

    # Calculate prior probabilities for each class
    def calculate_class_priors(self, labels):
        total_count = len(labels)
        classes, priors = np.unique(labels, return_counts=True)
        self.class_prior_probs = dict(zip(classes, priors / total_count))

    # Calculate likelihoods for each feature given a class
    def calculate_feature_likelihoods(self, feature_matrix, labels, smoothing_factor=0.01):
        num_samples, num_features = feature_matrix.shape
        class_feature_counts = np.zeros((6, num_features))
        class_total_counts = np.zeros(6)

        for i in np.unique(labels):
            class_feature_subset = feature_matrix[labels == i]
            class_feature_counts[i] = np.sum(class_feature_subset, axis=0)
            class_total_counts[i] = np.sum(class_feature_subset)

        smoothed_counts = class_feature_counts + smoothing_factor
        smoothed_totals = class_total_counts + num_features * smoothing_factor
        feature_likelihoods = smoothed_counts / smoothed_totals[:, None]

        return feature_likelihoods

    def fit(self, features, labels):
        # Calculate unique classes and priors
        self.unique_classes = np.unique(labels)
        self.calculate_class_priors(labels)

        class_priors_list = []
        for cls in self.unique_classes:
            class_prior = self.class_prior_probs[cls]
            class_priors_list.append(class_prior)
        self.class_priors = np.array(class_priors_list)

        self.feature_likelihoods = self.calculate_feature_likelihoods(features, labels)


    def predict_helper(self, feature):
        # Convert sparse matrix row to a dense array
        dense_feature = feature.toarray().ravel()
        posterior_probs = np.zeros(len(self.unique_classes))

        # Compute posterior probability for each class
        for i, _ in enumerate(self.unique_classes):
            log_prior = np.log(self.class_priors[i])
            log_likelihoods = np.log(self.feature_likelihoods[i, :])
            posterior_probs[i] = np.sum(log_likelihoods * dense_feature) + log_prior

        # Return the class with the highest posterior probability
        return self.unique_classes[np.argmax(posterior_probs)]

    def predict(self, features):
        # Generate predictions for each feature
        predictions = []
        for i in range(features.shape[0]):
            prediction = self.predict_helper(features[i])
            predictions.append(prediction)
        return predictions

    def evaluate_acc(self, true_labels, predicted_labels):
        correct_count = sum(int(actual == predicted) for actual, predicted in zip(true_labels, predicted_labels))
        accuracy = correct_count / len(true_labels)
        return accuracy


# Load dataset
dataset = load_dataset("dair-ai/emotion")

# Preprocess data
train_data = dataset['train']
test_data = dataset['test']

text_vectorizer = CountVectorizer()

train_features = text_vectorizer.fit_transform(train_data['text'])
train_labels = [entry['label'] for entry in train_data]

test_features = text_vectorizer.transform(test_data['text'])
test_labels = [entry['label'] for entry in test_data]

# Train and evaluate the model
bayes_model = CustomNaiveBayes()
bayes_model.fit(train_features, train_labels)
predictions = bayes_model.predict(test_features)

model_accuracy = bayes_model.evaluate_acc(test_labels, predictions)
print(f"Model Accuracy: {model_accuracy:.2f}")
