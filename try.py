import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pathlib import Path

# 定义Times New Roman字体路径
font_path = Path('TIMESBD.ttf')

# 加载字体
font_prop = FontProperties(fname=font_path)

# beta     [6.25, 34.0, 52.5, 68, 82, 89, 94.67]  beta in {1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6}
# agent    [54.5, 60, 68, 72.2/73.375, 78/74.75]               NUM_AGENT in {3,4,5,6,7}
# capacity [24.79 , 49.57, 68, 77.6]              capacity in {1,2,3,4}
# 示例数据
lambda_list = [2,4,6,8,10]
# ONDOC_plus = [93.5, 83.5, 76.5, 65.5, 59.5]
Mine = [80, 70, 68, 55, 47.5]
OnDoc = [57, 42, 28, 26.5, 21]
COFE = [70.5, 53, 38.5, 34, 23]

# [74.5, 67.5, 53.5, 44, 39]

fig, ax = plt.subplots()
# 设置柱状图的宽度
bar_width = 0.2

# 设置x轴的位置
r1 = np.arange(len(lambda_list))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
# r4 = [x + bar_width for x in r3]

# 创建柱状图
# plt.bar(r1, ONDOC_plus, color='lime', hatch='///', edgecolor='black',  width=bar_width, label='no cache')
plt.bar(r1, Mine, color='lime', hatch='\\\\\\', edgecolor='black', width=bar_width, label='Mine')
plt.bar(r2, OnDoc, color='cyan', hatch='xxx', edgecolor='black', width=bar_width, label='OnDoc')
plt.bar(r3, COFE, color='magenta', hatch='///', edgecolor='black', width=bar_width, label='COFE')

# 添加标签和标题
plt.xlabel('Lambda', fontweight='bold', fontsize=12, font=Path('TNR_bold.ttf'))
plt.ylabel('QoS(%)', fontsize=12, font=Path('TNR_bold.ttf'))
plt.xticks([r + bar_width for r in range(len(lambda_list))], lambda_list)
plt.title('QoS for Different Lambda Values', font=Path('TNR_bold.ttf'))
plt.grid(True, linestyle='--', alpha=0.3, zorder=0)
plt.gca().set_axisbelow(True)
plt.ylim(0, 100)

# 添加图例
ax.legend(prop=Path('TNR_bold.ttf'))
# 显示图形
plt.savefig('1.pdf', format='pdf',dpi=1000)
