from datasets import load_dataset

dataset = load_dataset("dair-ai/emotion")

train_dataset = dataset['train']
validation_dataset = dataset['validation']
test_dataset = dataset['test']

print(train_dataset[0])
