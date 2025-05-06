import json
import matplotlib.pyplot as plt
import numpy as np

def draw_data_distribution(file_path, client_num, classes_num):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    counts = {str(i): [0] * client_num for i in range(classes_num)}
    counts_sum = [0] * client_num
    for key, value in data.items():
        for sub_key, count in value['distribution'].items():
            counts[sub_key][int(key)] += count
        counts_sum[int(key)] += value['total']
    labels = list(range(client_num))
    values = np.array([counts[str(i)] for i in range(classes_num)])
    plt.figure(figsize=(20, 10))
    colors = [
        'red',
        'orange',
        'yellow',
        'green',
        'blue',
        'purple',
        'pink',
        'cyan',
        'brown',
        'gray',
        'magenta',
        'lime',
        'navy',
        'teal',
        'gold',
        'coral',
        'salmon',
        'violet',
        'indigo',
        'orchid',
        'khaki'
    ]
    plt.bar(labels, values[0], color=colors[0], label='0')
    for i in range(1, classes_num):
        plt.bar(labels, values[i], bottom=values[:i].sum(axis=0), color=colors[i], label=str(i))
    plt.xticks(rotation=45)
    plt.legend(title='number', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('./data_distribution.jpg', dpi=300)
    mean = np.mean(counts_sum)
    std_dev = np.std(counts_sum)
    max_value = np.max(counts_sum)
    min_value = np.min(counts_sum)
    print(f"sum: {sum(counts_sum)}")
    print(f"mean: {mean:.2f}")
    print(f"std: {std_dev:.2f}")
    print(f"maximum: {max_value}")
    print(f"minimum: {min_value}")
