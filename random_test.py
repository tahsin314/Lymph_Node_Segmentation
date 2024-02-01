import os
import numpy as np
import pandas as pd
from config.config import config_params

for key, value in config_params.items():
    if isinstance(value, str):
        exec(f"{key} = '{value}'")
    else:
        exec(f"{key} = {value}")
        
df = pd.read_csv(f"{data_dir}/train_labels.csv")
df = df.drop_duplicates()
train_df = df[(df['fold_patient'] != fold)] 
valid_df = df[df['fold_patient'] == fold]
test_df = pd.read_csv(f"{data_dir}/test_labels.csv")
all_df = pd.read_csv(f"{data_dir}/labels.csv")
all_df = all_df.drop_duplicates()
# print(len(np.unique(df['patient_id'].tolist())))
num_train, num_val, num_test, num_all = [], [], [], []

for i in np.unique(all_df['patient_id'].tolist()):
    num_all.append(len(os.listdir(f"{data_dir}/{i}/images")))
    # print(os.listdir(f"{data_dir}/{i}/images/"))
    
for i in np.unique(train_df['patient_id'].tolist()):
    num_train.append(len(os.listdir(f"{data_dir}/{i}/images")))
    
for i in np.unique(valid_df['patient_id'].tolist()):
    num_val.append(len(os.listdir(f"{data_dir}/{i}/images")))
    
for i in np.unique(test_df['patient_id'].tolist()):
    num_test.append(len(os.listdir(f"{data_dir}/{i}/images")))

# print(num_train, num_val, num_test, num_all, num_train + num_val + num_test)
print(f"All mean: {np.mean(num_all)} SD: {np.std(num_all)}")
print(f"Train mean: {np.mean(num_train)} SD: {np.std(num_train)}")
print(f"Val mean: {np.mean(num_val)} SD: {np.std(num_val)}")
print(f"Test mean: {np.mean(num_test)} SD: {np.std(num_test)}")

print(len(all_df))
print(len(train_df) + len(valid_df) + len(test_df))
print(len(np.unique(test_df['patient_id'].tolist())))