import json
import matplotlib.pyplot as plt
import numpy as np


def draw_data_distribution(file_path, client_num, classes_num):
    # 指定文件路径

    # 打开并读取 JSON 文件

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    counts = {str(i): [0] * client_num for i in range(classes_num)}  # 0-9的计数，100个编号

    counts_sum = [0] * client_num
    for key, value in data.items():
        for sub_key, count in value['distribution'].items():
            counts[sub_key][int(key)] += count
        counts_sum[int(key)] += value['total']

    labels = list(range(client_num))
    values = np.array([counts[str(i)] for i in range(classes_num)])  # 0-9的数量
    plt.figure(figsize=(20, 10))  # 宽12英寸，高6英寸
    colors = [
        'red',  # 红色
        'orange',  # 橙色
        'yellow',  # 黄色
        'green',  # 绿色
        'blue',  # 蓝色
        'purple',  # 紫色
        'pink',  # 粉色
        'cyan',  # 青色
        'brown',  # 棕色
        'gray',  # 灰色
        'magenta',  # 洋红色
        'lime',  # 酸橙色
        'navy',  # 海军蓝
        'teal',  # 水鸭色
        'gold',  # 金色
        'coral',  # 珊瑚色
        'salmon',  # 鲑鱼色
        'violet',  # 紫罗兰色
        'indigo',  # 靛蓝
        'orchid',  # 兰花色
        'khaki'  # 卡其色
    ]
    plt.bar(labels, values[0], color=colors[0], label='0')
    for i in range(1, classes_num):
        plt.bar(labels, values[i], bottom=values[:i].sum(axis=0), color=colors[i], label=str(i))

    # 添加标签和标题
    plt.xticks(rotation=45)
    plt.legend(title='数字', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y')

    # 显示图形
    plt.tight_layout()  # 自动调整布局
    plt.savefig('./data_distribution.jpg', dpi=300)
    mean = np.mean(counts_sum)  # 均值
    std_dev = np.std(counts_sum)  # 标准差
    max_value = np.max(counts_sum)  # 最大值
    min_value = np.min(counts_sum)  # 最小值

    print(f"sum: {sum(counts_sum)}")
    print(f"mean: {mean:.2f}")
    print(f"std: {std_dev:.2f}")
    print(f"maximum: {max_value}")
    print(f"minimum: {min_value}")
