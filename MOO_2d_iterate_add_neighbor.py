import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import random
import re
from itertools import product

random.seed(89) 

# class DataManual:
#     def __init__(self, n_gru, n_lstm, n_attention, n_ssm, dim_hidden, sequence,
#                  rela_error, training_time, param_all):
#         self.n_gru = n_gru
#         self.n_lstm = n_lstm
#         self.n_attention = n_attention
#         self.n_ssm = n_ssm
#         self.dim_hidden = dim_hidden
#         self.sequence = sequence
#         self.rela_error = rela_error
#         self.training_time = training_time
#         self.param_all = param_all

class DataManual:
    def __init__(
        self,
        window,
        learning_rate,
        batch_size,
        hidden_dim,
        epochs,
        layer_dim,
        val_rmse,
        training_time,
        model_instance=None,
    ):
        self.window = int(window)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.hidden_dim = int(hidden_dim)
        self.epochs = int(epochs)
        self.layer_dim = int(layer_dim)

        self.val_rmse = float(val_rmse)
        self.training_time = float(training_time)

        self.model_instance = model_instance


# def find_pareto_optimal_2d(data):
#     pareto_optimal = []
#     for candidate in data:
#         is_dominated = False
#         for other in data:
#             if (other.rela_error <= candidate.rela_error and
#                 other.training_time <= candidate.training_time and
#                 (other.rela_error < candidate.rela_error or
#                  other.training_time < candidate.training_time)):
#                 is_dominated = True
#                 break
#         if not is_dominated:
#             pareto_optimal.append(candidate)
#     return pareto_optimal

def find_pareto_optimal_2d(data):

    pareto_optimal = []

    for candidate in data:

        is_dominated = False

        for other in data:

            if (
                other.val_rmse <= candidate.val_rmse
                and other.training_time <= candidate.training_time
                and (
                    other.val_rmse < candidate.val_rmse
                    or other.training_time < candidate.training_time
                )
            ):
                is_dominated = True
                break

        if not is_dominated:
            pareto_optimal.append(candidate)

    return pareto_optimal





def seq_to_e(seq):
    m = re.search(r'(\d+)', str(seq))
    return int(m.group(1)) if m else 0


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

    ## The order of blocks: GRU, LSTM, Attention, SSM
    all_combinations = list(product(range(3), repeat=4))
    filtered_combinations = [combo for combo in all_combinations if combo != (0, 0, 0, 0)]
    num_blocks_all = [list(combo) for combo in filtered_combinations]

    case_all_new_list = []
    for row, seq in zip(case_all_filtered, sequence_all_filtered):
        a, b, c, d = row
        c1, c2, c3, c4 = num_blocks_all[int(c)]
        e = seq_to_e(seq)
        case_all_new_list.append([int(c1), int(c2), int(c3), int(c4), int(d), int(e)])

    case_all_new = np.array(case_all_new_list, dtype=int)

    data = []
    for i in range(len(case_all_new)):
        n_gru, n_lstm, n_attention, n_ssm, dim_hidden, e = case_all_new[i]
        data.append(DataManual(
            n_gru=n_gru,
            n_lstm=n_lstm,
            n_attention=n_attention,
            n_ssm=n_ssm,
            dim_hidden=dim_hidden,
            sequence=e,
            rela_error=rela_error_all_filtered[i],
            training_time=training_time_all_filtered[i],
            param_all=param_all_filtered[i]
        ))

    for entry in data:
        print(entry.__dict__)

    # Find Pareto-optimal data for all data
    pareto_optimal = find_pareto_optimal_2d(data)
    print(f"Pareto optimal data in Total data: {len(pareto_optimal)} in {len(data)}")
    # Find Pareto-optimal data for stochastic partial data, which is ratio% of the total data
    ratio = 0.05
    stochastic_sample = random.sample(data, int(len(data) * ratio)) if len(data) > 0 else []
    pareto_optimal_5pct = find_pareto_optimal_2d(stochastic_sample)
    print(f"Pareto optimal data in {ratio*100}% of total data: {len(pareto_optimal_5pct)} in {len(data[:int(len(data)*ratio)])}")

########################################################
### Sample new architectures which are near the architecture of Pareto front
#######################################################
    ALLOWED_BLOCKS = (0, 1, 2)
    ALLOWED_HIDDEN = (16, 32, 64)
    ALLOWED_SEQ    = (1, 2, 3, 4, 5, 6)
    N_ROUNDS = 3
    PER_CENTER = 7

    def _get_arch_tuple_from_datamanual(x):
        return (int(x.n_gru), int(x.n_lstm), int(x.n_attention), int(x.n_ssm), int(x.dim_hidden), int(x.sequence))

    def _neighbors_single_step_blocks(arch_tuple):
        c1, c2, c3, c4, d, e = arch_tuple
        base = [c1, c2, c3, c4]
        neigh = set()
        for i in range(4):
            for delta in (-1, +1):
                v = base[i] + delta
                if v in ALLOWED_BLOCKS:
                    nb = base.copy()
                    nb[i] = v
                    neigh.add((nb[0], nb[1], nb[2], nb[3], d, e))
        return list(neigh)

    def _try_fill_with_hidden_seq(center_tuple, current_set, need_more):
        c1, c2, c3, c4, d, e = center_tuple
        added = []
        h_opts = [h for h in ALLOWED_HIDDEN if h != d]
        s_opts = [s for s in ALLOWED_SEQ if s != e]
        for h in h_opts:
            if len(added) >= need_more:
                break
            cand = (c1, c2, c3, c4, h, e)
            if cand not in current_set:
                current_set.add(cand)
                added.append(cand)
        if len(added) < need_more:
            for s in s_opts:
                if len(added) >= need_more:
                    break
                cand = (c1, c2, c3, c4, d, s)
                if cand not in current_set:
                    current_set.add(cand)
                    added.append(cand)
        if len(added) < need_more:
            for h in h_opts:
                for s in s_opts:
                    if len(added) >= need_more:
                        break
                    cand = (c1, c2, c3, c4, h, s)
                    if cand not in current_set:
                        current_set.add(cand)
                        added.append(cand)
                if len(added) >= need_more:
                    break
        return added

    def _arch_tuple_from_dm(x):
        return (int(x.n_gru), int(x.n_lstm), int(x.n_attention), int(x.n_ssm), int(x.dim_hidden), int(x.sequence))

    arch_to_idx = {tuple(case_all_new[i]): i for i in range(len(case_all_new))}
    current_sample = list(stochastic_sample)
    sampled_set = set(_arch_tuple_from_dm(x) for x in current_sample)
    rounds_new_points = []
    fronts_per_round = []

    for r in range(N_ROUNDS):
        centers = find_pareto_optimal_2d(current_sample)
        neighbors_this_round = []
        for idx_c, p in enumerate(centers):
            center = _get_arch_tuple_from_datamanual(p)
            neigh = _neighbors_single_step_blocks(center)
            filtered = []
            for t in neigh:
                if t == center:
                    continue
                if t in sampled_set:
                    continue
                c1, c2, c3, c4, d, e = t
                if (c1 in ALLOWED_BLOCKS and c2 in ALLOWED_BLOCKS and
                    c3 in ALLOWED_BLOCKS and c4 in ALLOWED_BLOCKS and
                    d in ALLOWED_HIDDEN and e in ALLOWED_SEQ):
                    filtered.append(t)
            random.shuffle(filtered)
            take = filtered[:PER_CENTER]

            if len(take) < PER_CENTER:
                cur_set = set(take)
                _ = _try_fill_with_hidden_seq(center, cur_set, PER_CENTER - len(take))
                take = [t for t in cur_set if t not in sampled_set][:PER_CENTER]
            for t in take:
                idx = arch_to_idx.get(tuple(int(v) for v in t), None)
                if idx is None:
                    continue
                c1, c2, c3, c4, d, e = t
                dm = DataManual(
                    n_gru=c1, n_lstm=c2, n_attention=c3, n_ssm=c4,
                    dim_hidden=d, sequence=e,
                    rela_error=float(rela_error_all_filtered[idx]),
                    training_time=float(training_time_all_filtered[idx]),
                    param_all=float(param_all_filtered[idx])
                )
                neighbors_this_round.append(dm)
                sampled_set.add(t)

        current_sample.extend(neighbors_this_round)
        rounds_new_points.append(neighbors_this_round)
        fronts_per_round.append(find_pareto_optimal_2d(current_sample))

    pareto_optimal_combined = fronts_per_round[-1] if fronts_per_round else find_pareto_optimal_2d(current_sample)
    fontsize = 14
    plt.figure(figsize=(9, 4))
    plt.scatter([entry.rela_error for entry in data],
                [entry.training_time for entry in data],
                label='All data (708)',
                marker='o', color='#9bbcff', alpha=0.25, zorder=1)
    pareto_sorted = sorted(pareto_optimal, key=lambda x: x.rela_error)
    plt.plot([entry.rela_error for entry in pareto_sorted],
            [entry.training_time for entry in pareto_sorted],
            color='#0048ba', linestyle='-', linewidth=2.5, alpha=1,
            label='Pareto front of all data', zorder=3)
    plt.scatter([entry.rela_error for entry in stochastic_sample],
                [entry.training_time for entry in stochastic_sample],
                label=f'Stochastic data (5%: {len(stochastic_sample)})',
                color='#9acd32', marker='o', alpha=0.55,
                edgecolor='black', linewidths=0.4, zorder=2)
    pareto_5pct_sorted = sorted(pareto_optimal_5pct, key=lambda x: x.rela_error)
    plt.plot([entry.rela_error for entry in pareto_5pct_sorted],
            [entry.training_time for entry in pareto_5pct_sorted],
            color='#2e8b57', linestyle='-', linewidth=1.5, alpha=0.9,
            label=f'Pareto front of 5% data', zorder=4)
    round_colors = [ '#9467bd','#000000', '#d62728']       
    round_line_colors = ['#9467bd','#000000', '#d62728']
    markers = ['s', '^', '*', 'X'] 
    for r, (pts_new, front_r) in enumerate(zip(rounds_new_points, fronts_per_round)):
        if pts_new:
            plt.scatter(
                [e.rela_error for e in pts_new],
                [e.training_time for e in pts_new],
                label=f'R{r+1} new ({len(pts_new)})',
                color=round_colors[r % len(round_colors)],
                marker=markers[r % len(markers)], 
                alpha=0.5,
                linewidths=0.5,
                zorder=3 + r
            )
        if front_r:
            fr_sorted = sorted(front_r, key=lambda x: x.rela_error)
            plt.plot(
                [e.rela_error for e in fr_sorted],
                [e.training_time for e in fr_sorted],
                color=round_line_colors[r % len(round_line_colors)],
                linestyle='-',
                linewidth=1.5,
                alpha=0.7,
                label=f'Pareto front after R{r+1}',
                zorder=6 + r
            )

    plt.xlabel('Relative L2 Error', fontsize=fontsize)
    plt.ylabel('Training Time (s)', fontsize=fontsize)
    plt.tick_params(axis='x', labelsize=fontsize)
    plt.tick_params(axis='y', labelsize=fontsize)
    #plt.xticks([])
    #plt.yticks([])
    plt.legend(loc='upper right', fontsize=fontsize-2)
    plt.tight_layout()
    plt.savefig(f'pareto_iterative_3rounds_seed0_neighbor{PER_CENTER}.pdf')
    plt.show()

    

    

if __name__ == "__main__":
    main()
