import numpy as np
import matplotlib.pyplot as plt

pathologist1 = [22, 48, 11, 37, 14, 10, 21]
pathologist2 = [27, 44, 12, 30, 16, 11, 18]
pipeline = [18.3, 40.1, 7.4, 24.4, 9.4, 10.0, 16.5]

def bland_altman_plot(data1, data2, label1, label2, ax):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    means = np.mean([data1, data2], axis=0)
    diffs = data1 - data2

    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1) 
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff

    ax.scatter(means, diffs, alpha=0.7, edgecolors='k')

    ax.axhline(mean_diff, color='red', linestyle='-', label='Mean Difference (Bias)')
    ax.axhline(upper_loa, color='gray', linestyle='--', label='Upper LoA (+1.96 SD)')
    ax.axhline(lower_loa, color='gray', linestyle='--', label='Lower LoA (-1.96 SD)')
    
    ax.text(ax.get_xlim()[1] * 0.98, mean_diff, f'Mean: {mean_diff:.2f}', ha='right', va='bottom', color='red')
    ax.text(ax.get_xlim()[1] * 0.98, upper_loa, f'Upper Limit: {upper_loa:.2f}', ha='right', va='bottom', color='gray')
    ax.text(ax.get_xlim()[1] * 0.98, lower_loa, f'Lower Limit: {lower_loa:.2f}', ha='right', va='top', color='gray')

    ax.set_title(f'Bland-Altman Plot: {label1} vs. {label2}', fontsize=14)
    ax.set_xlabel(f'Mean of {label1} and {label2}', fontsize=12)
    ax.set_ylabel(f'Difference ({label1} - {label2})', fontsize=12)
    ax.grid(True, linestyle='--')

fig1, ax1 = plt.subplots(figsize=(8, 6))
bland_altman_plot(pathologist1, pathologist2, 'Pathologist 1', 'Pathologist 2', ax1)
fig1.suptitle("Analysis 1: Inter-Rater Reliability", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])

expert_average = np.mean([pathologist1, pathologist2], axis=0)

fig2, ax2 = plt.subplots(figsize=(8, 6))
bland_altman_plot(pipeline, expert_average, 'Pipeline', 'Pathologist Average', ax2)
fig2.suptitle("Analysis 2: System vs. Pathologist Consensus", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.show()