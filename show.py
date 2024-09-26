import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import rl_utils

# 提取用于排序的关键字函数
def extract_key(filename):
    match = re.search(r'ES_(\d+)', filename)
    # match = re.search(r'AC-([\deE.-]+)_CR', filename)
    if match:
        return float(match.group(1))
    return float('inf')  # 如果没有匹配，返回一个很大的值放在最后

def extract_ac_cr(filename):
    ac_match = re.search(r'AC-([\d.]+)', filename)
    cr_match = re.search(r'CR-([\d.]+)', filename)
    ac_value = ac_match.group(1) if ac_match else 'unknown'
    cr_value = cr_match.group(1) if cr_match else 'unknown'
    return ac_value, cr_value

# npy_files = sorted(glob.glob('return_list_default/*.npy'), key=extract_key)
npy_files = sorted(glob.glob('./*.npy'), key=extract_key)

fig, ax1 = plt.subplots()

for file in npy_files:
    data = np.load(file)
    my_return = rl_utils.moving_average(data, 9)
    SR = (200+my_return/20)/200
    ac_lr, cr_lr = extract_ac_cr(file)
    es = extract_key(file)
    # 绘制第一个y轴的数据
    ax1.plot(my_return, label=f'ES={es}')
    # ax1.plot(my_return, label=f'AC_LR={ac_lr},CR_LR={cr_lr}')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Reward', color='black')
    ax1.tick_params(axis='y')
    # ax1.set_ylim(-3500, -1000)
    ax1.set_ylim(-3500, -500)
    # 创建第二个y轴，共享同一个x轴
    ax2 = ax1.twinx()
    # ax2.plot(SR, label=file)
    ax2.set_ylabel('QoS(%)', color='black')
    ax2.tick_params(axis='y')
    # ax2.set_ylim(0.125, 0.75)
    ax2.set_ylim(0.125, 0.875)
    
# 添加图例
ax1.legend(loc='lower right')
ax1.grid(True, linestyle='--', alpha=0.5)
# 调整布局，防止标签和图形重叠
fig.tight_layout()
# 显示图形
plt.savefig('result.png', format='png', dpi=1000)