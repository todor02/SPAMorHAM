import csv
import random
import math
import re
from collections import defaultdict

def load_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            label = row[0].strip()
            message = row[1].strip()
            data.append((message, label))
    random.shuffle(data)
    return data

def split_data(data, split_ratio=0.8):
    split_index = int(len(data) * split_ratio)
    return data[:split_index], data[split_index:]

def tokenize(text):
    text = text.lower()
    return re.findall(r"\b\w+['\w]*\b", text)

def build_vocabulary(training_data):
    vocabulary = set()
    for message, _ in training_data:
        tokens = tokenize(message)
        vocabulary.update(tokens)
    return sorted(vocabulary)

class MultinomialNaiveBayes:
    def __init__(self):
        self.vocab = None
        self.class_priors = None
        self.class_word_counts = None
        self.class_total_words = None
        self.V = 0

    def train(self, training_data):
        self.vocab = build_vocabulary(training_data)
        self.V = len(self.vocab)
        
        self.class_priors = defaultdict(int)
        self.class_word_counts = {'spam': defaultdict(int), 'ham': defaultdict(int)}
        self.class_total_words = {'spam': 0, 'ham': 0}
        
        for message, label in training_data:
            self.class_priors[label] += 1
            tokens = tokenize(message)
            for token in tokens:
                if token in self.vocab:
                    self.class_word_counts[label][token] += 1
                    self.class_total_words[label] += 1
        
        total_samples = len(training_data)
        self.prior_spam = self.class_priors['spam'] / total_samples
        self.prior_ham = self.class_priors['ham'] / total_samples
        
    def predict(self, message):
        tokens = tokenize(message)
        log_prob_spam = math.log(self.prior_spam)
        log_prob_ham = math.log(self.prior_ham)
        
        for token in tokens:
            if token in self.vocab:
                count_spam = self.class_word_counts['spam'].get(token, 0)
                log_prob_spam += math.log((count_spam + 1) / (self.class_total_words['spam'] + self.V))
                
                count_ham = self.class_word_counts['ham'].get(token, 0)
                log_prob_ham += math.log((count_ham + 1) / (self.class_total_words['ham'] + self.V))
        
        return 'spam' if log_prob_spam > log_prob_ham else 'ham'
    
# Load and prepare data
data = load_data('spam.csv')  # Replace with your dataset
train_data, test_data = split_data(data)

# Train classifier
classifier = MultinomialNaiveBayes()
classifier.train(train_data)

# Evaluate accuracy
correct = 0
total = len(test_data)
for message, true_label in test_data:
    predicted = classifier.predict(message)
    if predicted == true_label:
        correct += 1

print(f"Accuracy: {correct / total:.2f}")


def interactive_demo(classifier):
    print("\n--- Spam Classifier Demo ---")
    print("Type 'exit' to quit\n")
    while True:
        message = input("Enter a message to classify: ")
        if message.lower() == 'exit':
            break
        prediction = classifier.predict(message)
        print(f"Prediction: {prediction.upper()}\n")

def show_examples(classifier):
    test_cases = [
        ("WIN A FREE iPhone! Click now!", "spam"),
        ("Hey, are we meeting tomorrow at 5 PM?", "ham"),
        ("URGENT: Your account has been compromised", "spam"),
        ("Don't forget to buy milk", "ham")
    ]
    
    print("\n--- Example Predictions ---")
    for msg, true_label in test_cases:
        pred = classifier.predict(msg)
        print(f"Message: {msg}")
        print(f"Predicted: {pred} | Actual: {true_label}")
        print("✓ Correct" if pred == true_label else "✗ Wrong", "\n")

def detailed_metrics(classifier, test_data):
    tp, fp, tn, fn = 0, 0, 0, 0
    
    for message, true_label in test_data:
        pred = classifier.predict(message)
        if true_label == 'spam':
            tp += 1 if pred == 'spam' else 0
            fn += 1 if pred == 'ham' else 0
        else:
            tn += 1 if pred == 'ham' else 0
            fp += 1 if pred == 'spam' else 0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n--- Detailed Metrics ---")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1-Score:  {f1:.2f}")

if __name__ == "__main__":
    # Load data and train
    data = load_data('spam.csv')
    train_data, test_data = split_data(data)
    classifier = MultinomialNaiveBayes()
    classifier.train(train_data)

    # Evaluate
    correct = sum(1 for msg, lbl in test_data if classifier.predict(msg) == lbl)
    print(f"\nAccuracy: {correct / len(test_data):.2f}")

    # Showcase
    detailed_metrics(classifier, test_data)
    show_examples(classifier)
    interactive_demo(classifier)
