import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter

class DataManual:
    def __init__(self, lookback, sampling_interval, num_layers, dim_hidden, 
                 rela_error, training_time, param_all, sequence):
        self.lookback = lookback
        self.sampling_interval = sampling_interval
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.rela_error = rela_error
        self.training_time = training_time
        self.param_all = param_all
        self.sequence = sequence

def find_pareto_optimal(data):
    pareto_optimal = []
    for candidate in data:
        is_dominated = False
        for other in data:
            if (other.rela_error <= candidate.rela_error and
                other.training_time <= candidate.training_time and
                other.param_all <= candidate.param_all and
                (other.rela_error < candidate.rela_error or
                 other.training_time < candidate.training_time or
                 other.param_all < candidate.param_all)):
                is_dominated = True
                break
        if not is_dominated:
            pareto_optimal.append(candidate)
    return pareto_optimal

def main():
    folder_prefixes = ['Integrated1', 'Integrated2', 'Integrated3', 'Integrated4', 'Integrated5', 'Integrated6']
    case_prefixes = ['Case_simple_1', 'Case_simple_2', 'Case_simple_3']
    num_remove_map = {
        'Integrated1':[],
        'Integrated2': [0, 1, 2, 5, 8, 9, 10, 11, 14, 17, 18, 19, 20, 23, 26, 27, 28, 29, 32, 35, 36, 37, 38,
                        41, 44, 45, 46, 47, 50, 53, 54, 55, 56, 59, 62, 63, 64, 65, 68, 71, 72, 73, 74, 77],
        'Integrated3': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                    25, 26, 27, 28, 35, 36, 37, 44, 45, 46, 53, 54, 55, 62, 63, 64, 71, 72, 73],
        'Integrated4': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                    25, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77],
    }

    num_keep_map = {
        'Integrated5': [30, 31, 33, 34, 39, 40, 42, 43, 48, 49, 51, 52, 57, 58, 60, 61, 66, 67, 69, 70, 75, 76, 78, 79],
        'Integrated6': [30, 31, 33, 34, 39, 40, 42, 43, 48, 49, 51, 52, 57, 58, 60, 61, 66, 67, 69, 70, 75, 76, 78, 79],
    }

    case_all_filtered = []
    rela_error_all_filtered = []
    training_time_all_filtered = []
    param_all_filtered = []
    sequence_all_filtered = []

    for prefix0 in folder_prefixes:
        if prefix0 in num_remove_map:
            num_remove1 = num_remove_map[prefix0]
        elif prefix0 in num_keep_map:
            num_keep1 = num_keep_map[prefix0]
        else:
            num_remove1 = []  
            num_keep1 = None 

        for prefix in case_prefixes:
            cases = np.load(f'../{prefix0}/{prefix}/case_all.npy', allow_pickle=True)
            errors = np.load(f'../{prefix0}/{prefix}/rela_l2_all.npy', allow_pickle=True)
            times = np.load(f'../{prefix0}/{prefix}/Time_all.npy', allow_pickle=True)
            params = np.load(f'../{prefix0}/{prefix}/param_all.npy', allow_pickle=True)
            
            for i, case in enumerate(cases):
                if prefix0 in num_remove_map and case[2] not in num_remove1:
                    case_all_filtered.append(case)
                    rela_error_all_filtered.append(errors[i])
                    training_time_all_filtered.append(times[i])
                    param_all_filtered.append(params[i])
                    sequence_all_filtered.append(prefix0)
                    
                elif prefix0 in num_keep_map and case[2] in num_keep1:
                    case_all_filtered.append(case)
                    rela_error_all_filtered.append(errors[i])
                    training_time_all_filtered.append(times[i])
                    param_all_filtered.append(params[i])
                    sequence_all_filtered.append(prefix0)

    case_all_filtered = np.array(case_all_filtered)
    rela_error_all_filtered = np.array(rela_error_all_filtered)
    training_time_all_filtered = np.array(training_time_all_filtered)
    param_all_filtered = np.array(param_all_filtered)
    sequence_all_filtered = np.array(sequence_all_filtered)

    data = []
    for i in range(len(case_all_filtered)):
        lookback, sampling_interval, num_layers, dim_hidden = case_all_filtered[i]
        data.append(DataManual(
            lookback=lookback,
            sampling_interval=sampling_interval,
            num_layers=num_layers,
            dim_hidden=dim_hidden,
            rela_error=rela_error_all_filtered[i],
            training_time=training_time_all_filtered[i],
            param_all=param_all_filtered[i],
            sequence=sequence_all_filtered[i]
        ))

    for entry in data:
        print(entry.__dict__)


    pareto_optimal = find_pareto_optimal(data)
    np.save('pareto_optimal.npy', pareto_optimal)
    print(f"Total data: {len(data)}")
    print(f"Pareto optimal data: {len(pareto_optimal)}")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([entry.rela_error for entry in data], 
            [entry.training_time for entry in data], 
            [entry.param_all for entry in data],  
            label='All data', marker='o', color='b', alpha=0.2, zorder=1)
    ax.scatter([entry.rela_error for entry in pareto_optimal], 
           [entry.training_time for entry in pareto_optimal], 
           [entry.param_all for entry in pareto_optimal],  
           color='r', label='Pareto-optimal', marker='o', s=60, zorder=2)  

    ax.set_xlabel('Relative L2 Error', fontsize=12)
    ax.set_ylabel('Training Time', fontsize=12)
    ax.set_zlabel('Number of Parameters', fontsize=12)
    ax.zaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='z', style='sci', scilimits=(0,0))
    ax.tick_params(axis='x', labelsize=12) 
    ax.tick_params(axis='y', labelsize=12) 
    ax.tick_params(axis='z', labelsize=12) 
    ax.legend()
    plt.savefig('pareto_optimal.pdf')


    fig = plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(3, 1, 1)
    ax1.scatter([entry.rela_error for entry in data],
                [entry.training_time for entry in data],
                label='All data', marker='o', color='b', alpha=0.25, s=15, zorder=1)
    ax1.scatter([entry.rela_error for entry in pareto_optimal],
                [entry.training_time for entry in pareto_optimal],
                label='Pareto optimal', marker='o', color='r', s=30, zorder=2)
    ax1.set_xlabel('Error', fontsize=14)
    ax1.set_ylabel('Time (s)', fontsize=14)
    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax1.yaxis.get_offset_text().set_fontsize(14)
    ax1.set_xticks([])
    ax1.tick_params(axis='both', labelsize=14)
    ax2 = plt.subplot(3, 1, 2)
    ax2.scatter([entry.rela_error for entry in data],
                [entry.param_all for entry in data],
                marker='o', color='b', alpha=0.25, s=15, zorder=1)
    ax2.scatter([entry.rela_error for entry in pareto_optimal],
                [entry.param_all for entry in pareto_optimal],
                marker='o', color='r', s=30, zorder=2)
    ax2.set_xlabel('Error', fontsize=14)
    ax2.set_ylabel('Parameters', fontsize=14)
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax2.yaxis.get_offset_text().set_fontsize(14)
    ax2.set_xticks([])
    ax2.tick_params(axis='both', labelsize=14)
    ax3 = plt.subplot(3, 1, 3)
    ax3.scatter([entry.training_time for entry in data],
                [entry.param_all for entry in data],
                marker='o', color='b', alpha=0.25, s=15, zorder=1)
    ax3.scatter([entry.training_time for entry in pareto_optimal],
                [entry.param_all for entry in pareto_optimal],
                marker='o', color='r', s=30, zorder=2)

    ax3.set_xlabel('Time (s)', fontsize=14)
    ax3.set_ylabel('Parameters', fontsize=14)
    ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax3.yaxis.get_offset_text().set_fontsize(14)
    ax3.tick_params(axis='both', labelsize=14)
    ax1.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15)  
    plt.savefig('pareto_optimal_projections.pdf', bbox_inches='tight')
    plt.show()

    

if __name__ == "__main__":
    main()
