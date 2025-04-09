import matplotlib.pyplot as plt
import json
import numpy as np

if __name__ == '__main__':
    with open('assets/durations.json', 'r') as f:
        data = json.load(f)

    print(f'Length of data: {len(data)}')
    data_arr = np.array(data)
    print(f'Data mean: {data_arr.mean()}')

    # New variable for chosen duration
    chosen_duration = 20.0

    # Create histogram and get bin data
    plt.figure(figsize=(13, 7))
    n, bins, patches = plt.hist(
        data, bins=50, color='skyblue', edgecolor='black')

    # Calculate cumulative distribution
    cumsum = np.cumsum(n)
    total = len(data)
    half_total = total * 0.5

    # Find 50% threshold bin boundary
    threshold_idx = np.argmax(cumsum >= half_total)
    threshold_bin_edge = bins[threshold_idx + 1]

    # Find most frequent bin (mode)
    max_freq_idx = np.argmax(n)
    max_freq = n[max_freq_idx]
    bin_center = (bins[max_freq_idx] + bins[max_freq_idx + 1]) / 2

    # Calculate chosen duration statistics
    count_chosen = sum(1 for x in data if x <= chosen_duration)
    percent_chosen = (count_chosen / total) * 100

    # Add vertical lines
    plt.axvline(bin_center, color='red', linestyle='--', linewidth=1.5,
                label=f'Most Frequent: {bin_center:.2f}\n(Count: {max_freq})')

    plt.axvline(threshold_bin_edge, color='green', linestyle=':', linewidth=1.5,
                label=f'50% Threshold: {threshold_bin_edge:.2f}\n({cumsum[threshold_idx]} items)')

    plt.axvline(chosen_duration, color='purple', linestyle='-.', linewidth=2,
                label=f'Chosen Duration: {chosen_duration}\n'
                f'Samples ≤ {chosen_duration}: {count_chosen} ({percent_chosen:.1f}%)')

    # Add labels and title
    plt.xlabel('Duration', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Duration Distribution with Key Metrics', fontsize=14, pad=20)
    plt.legend(loc='upper right', fontsize=10)

    # Save and close
    plt.savefig('assets/durations_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
