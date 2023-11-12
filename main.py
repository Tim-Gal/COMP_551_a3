from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer


dataset = load_dataset("dair-ai/emotion")

train_dataset = dataset['train']
validation_dataset = dataset['validation']
test_dataset = dataset['test']

"""
with open("text.txt", 'w') as file:
    for line in train_dataset['text']:
        file.write(line + '\n')
"""

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(train_dataset['text'])

print(X_train_vectorized.toarray())
print("Vocabulary:", vectorizer.get_feature_names_out())


