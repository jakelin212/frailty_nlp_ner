#PyTorch 3.8
import pandas as pd
proj = "incontinence_binary"
#proj = "falling"
#proj = "lonely" 
#proj = "mobility2"

#our pipeline loads and stores labeled data in this naming convention following BIO tagging 
dataset = pd.read_parquet(proj + "_bio_labeled_data.parquet")


from sklearn.model_selection import train_test_split

# Get the unique original_ids
unique_ids = dataset['text_id'].unique()

# Split the unique_ids into training, test, and validation using a fixed seed for reproducibility
train_ids, temp_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)
test_ids, val_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

# Using these IDs, filter the original dataset to get the respective data splits
train_data = dataset[dataset['text_id'].isin(train_ids)]
test_data = dataset[dataset['text_id'].isin(test_ids)]
val_data = dataset[dataset['text_id'].isin(val_ids)]


from collections import Counter

# Inspect class distribution
train_labels = [label for sample in train_data["bi_tags"] for label in sample]
label_counts = Counter(train_labels)
print("Class distribution:", label_counts)

import matplotlib.pyplot as plt

data = label_counts#{'milk': 60, 'water': 10}
names = list(label_counts.keys())
values = list(label_counts.values())

plt.bar(range(len(data)), values, tick_label=names)
#plt.show()
plt.xticks(rotation=90)
plt.title(proj)
plt.savefig("./_label_balance_" + proj + ".png", bbox_inches='tight')