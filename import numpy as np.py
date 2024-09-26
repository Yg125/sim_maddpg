import numpy as np
import itertools
avail_action = [[[0, 0, 0, 0, 1],[0, 0, 0, 1, 0],[0, 0, 1, 0, 0],[0, 1, 0, 0, 0],[1, 0, 0, 0, 0]],
                        [[0, 0, 0, 1, 1],[0, 0, 1, 0, 1],[0, 0, 1, 1, 0],[0, 1, 0, 0, 1],[0, 1, 0, 1, 0],[0, 1, 1, 0, 0],[1, 0, 0, 0, 1],[1, 0, 0, 1, 0],[1, 0, 1, 0, 0],[1, 1, 0, 0, 0]],
                        [[0, 0, 1, 1, 1],[0, 1, 0, 1, 1],[0, 1, 1, 0, 1],[0, 1, 1, 1, 0],[1, 0, 0, 1, 1],[1, 0, 1, 0, 1],[1, 0, 1, 1, 0],[1, 1, 0, 0, 1],[1, 1, 0, 1, 0],[1, 1, 1, 0, 0]],
                        [[0, 1, 1, 1, 1],[1, 0, 1, 1, 1],[1, 1, 0, 1, 1],[1, 1, 1, 0, 1],[1, 1, 1, 1, 0]]]
actions = []
next_policy = np.zeros((5,5))

for i in range(5):
    valid_actions = [action for action in avail_action[2] if all(action[j] == 1 for j in range(5) if next_policy[i][j] == 1)]
    actions.append(valid_actions)
    
# print(actions)
all_combinations = list(itertools.product(*actions))
batch_size = 4000
batched_combinations = [all_combinations[i:i + batch_size] for i in range(0, len(all_combinations), batch_size)]
# 打印划分后的每一组的长度
print(batched_combinations[0])
# for idx, batch in enumerate(batched_combinations):
#     print(f"Batch {idx + 1} has {len(batch)} combinations.")

# 如果你想查看每一组的具体内容，可以取消下面的注释。
# for idx, batch in enumerate(batched_combinations):
#     print(f"Batch {idx + 1}:")
#     for combination in batch:
#         print(combination)
#     print()
# print(batched_combinations)
# for idx, combination in enumerate(all_combinations):
#     print(f"Combination {idx + 1}: {combination}")
