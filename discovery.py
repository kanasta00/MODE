import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker

class DataManual:
    def __init__(self, lookback, sampling_interval, num_layers, dim_hidden, 
                 rela_error, training_time, param_all, sequence):
        self.lookback = lookback
        self.sampling_interval = sampling_interval
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.sequence = sequence
        self.rela_error = rela_error
        self.training_time = training_time
        self.param_all = param_all

opts = np.load('pareto_optimal.npy', allow_pickle=True)

data = []
for entry in opts:
    data.append({
        'lookback': entry.lookback,  
        'sampling_interval': entry.sampling_interval,  
        'num_layers': entry.num_layers, 
        'dim_hidden': entry.dim_hidden, 
        'sequence': entry.sequence,
        'rela_error': entry.rela_error,  
        'training_time': entry.training_time,  
        'param_all': entry.param_all  
    })

data_df = pd.DataFrame(data)
data_rescal = data_df.copy()

for col in ['rela_error', 'training_time', 'param_all']:  
    min_val = data_rescal[col].min()
    max_val = data_rescal[col].max()
    data_rescal[col] = (data_rescal[col] - min_val) / (max_val - min_val)

data_rescal['p1'] = data_rescal[['rela_error', 'training_time', 'param_all']].mean(axis=1)
data_rescal['p2'] = 1 * data_rescal['rela_error'] + 0 * data_rescal['training_time'] + 0 * data_rescal['param_all']
data_rescal['p3'] = 0 * data_rescal['rela_error'] + 1 * data_rescal['training_time'] + 0 * data_rescal['param_all']
data_rescal['p4'] = 0.6 * data_rescal['rela_error'] + 0.2 * data_rescal['training_time'] + 0.2 * data_rescal['param_all']

p51 = np.zeros(len(data_rescal))
for i in range(len(data_rescal)):
    if data_df['rela_error'][i] <= 11.5:
        p51[i] = 0
    else:
        p51[i] = 1000
p62 = np.zeros(len(data_rescal))
for i in range(len(data_rescal)):
    if data_df['training_time'][i] <= 200:
        p62[i] = 0
    else:
        p62[i] = 1000
p63 = np.zeros(len(data_rescal))
for i in range(len(data_rescal)):
    if data_df['param_all'][i] <= 20000:
        p63[i] = 0
    else:
        p63[i] = 1000

p72 = np.zeros(len(data_rescal))
for i in range(len(data_rescal)):
    p72[i] = np.log2(data_df['training_time'][i])

p73 = np.zeros(len(data_rescal))
for i in range(len(data_rescal)):
    if data_df['param_all'][i] <= 10000:
        p73[i] = 0
    else:
        p73[i] = data_df['param_all'][i]-10000
data_rescal['p5'] = p51 + 0.2 * data_rescal['training_time'] + 0.8 * data_rescal['param_all']
data_rescal['p6'] = 1 * data_rescal['rela_error'] + p62 + p63
data_rescal['p7'] = 0.7 * data_rescal['rela_error'] + 0.01 * p72 + 0.01 * p73

min_p1 = data_rescal['p1'].min()
min_p2 = data_rescal['p2'].min()
min_p3 = data_rescal['p3'].min()
min_p4 = data_rescal['p4'].min()
min_p5 = data_rescal['p5'].min()
min_p6 = data_rescal['p6'].min()
min_p7 = data_rescal['p7'].min()

min_p1_row = data_rescal[data_rescal['p1'] == min_p1]
min_p2_row = data_rescal[data_rescal['p2'] == min_p2]
min_p3_row = data_rescal[data_rescal['p3'] == min_p3]
min_p4_row = data_rescal[data_rescal['p4'] == min_p4]
min_p5_row = data_rescal[data_rescal['p5'] == min_p5]
min_p6_row = data_rescal[data_rescal['p6'] == min_p6]
min_p7_row = data_rescal[data_rescal['p7'] == min_p7]

print(f"Min p1: {min_p1}, corresponding values: \n{min_p1_row[['sequence', 'num_layers', 'dim_hidden']]}")
print(f"Min p2: {min_p2}, corresponding values: \n{min_p2_row[['sequence', 'num_layers', 'dim_hidden']]}")
print(f"Min p3: {min_p3}, corresponding values: \n{min_p3_row[['sequence', 'num_layers', 'dim_hidden']]}")
print(f"Min p4: {min_p4}, corresponding values: \n{min_p4_row[['sequence', 'num_layers', 'dim_hidden']]}")

output_filename = 'discovery.csv'
data_rescal.to_csv(output_filename, index=False)

print(f"Data has been written to {output_filename}")

# Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_rescal['rela_error'], data_rescal['training_time'], data_rescal['param_all'], label='Pareto-optimal',color='r', alpha=0.3, zorder=1)
ax.scatter(min_p1_row['rela_error'], min_p1_row['training_time'], min_p1_row['param_all'], label='Min p1', c='k', marker='*', s=80, zorder=2)  
ax.scatter(min_p2_row['rela_error'], min_p2_row['training_time'], min_p2_row['param_all'], label='Min p2', c='m', marker='*', s=80, zorder=2) 
ax.scatter(min_p3_row['rela_error'], min_p3_row['training_time'], min_p3_row['param_all'], label='Min p3', c='b', marker='*', s=80, zorder=2) 
ax.scatter(min_p4_row['rela_error'], min_p4_row['training_time'], min_p4_row['param_all'], label='Min p4', c='c', marker='*', s=80, zorder=2)

ax.set_xlabel('Relative L2 Error', fontsize=12) 
ax.set_ylabel('Training Time', fontsize=12) 
ax.set_zlabel('Number of Parameters', fontsize=12) 
ax.tick_params(axis='x', labelsize=12) 
ax.tick_params(axis='y', labelsize=12) 
ax.tick_params(axis='z', labelsize=12) 
ax.legend()
plt.show()

