import pandas as pd
from sklearn.model_selection import train_test_split
# data = pd.read_csv('data/all_capable2.csv')
# print(data)
# data_ = pd.DataFrame(columns=['Input','Output'])
# inputs = []
# outputs = []
# for i in range(len(data)):
#     input = data['head'][i]
#     if data['relation'][i] == 'CapableOf':
#         input += ' CapableOf'
#     else:
#         input += ' NotCapableOf'
#     inputs.append(input)
#     outputs.append(data['tail'][i])
# data_['Input'] = inputs
# data_['Output'] = outputs
# data_.to_csv('data/data.csv',sep='\t')
# print(data_)

data = pd.read_csv('data/data.csv',sep='\t',index_col=0)
print(data)
train, test = train_test_split(data, test_size=0.1, random_state=42)
train, val = train_test_split(train, test_size=0.1, random_state=42)
train.to_csv('data/train.csv')
test.to_csv('data/test.csv')
val.to_csv('data/val.csv')
print(train)