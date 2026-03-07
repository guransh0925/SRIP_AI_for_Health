import pandas as pd
df = pd.read_pickle('Dataset/breathing_dataset.pkl')
print(df.shape)
print(df.head())
print(df['label'].value_counts())