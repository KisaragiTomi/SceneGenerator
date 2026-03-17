import matplotlib.pyplot as plt
import re


log_path = './checkpoints/normaltest/loss_log.txt'
# losses = {'G_ID': [], 'mid': [], 'shallow': [], 'dice_G_ID': [], 'dice_shallow': []}
losses = {'G_ID': [], 'GAN_Feat': [], 'D_real': []}

with open(log_path, 'r') as f:
    for line in f:
        # 使用正则提取数值
        for key in losses.keys():
            match = re.search(f'{key}: ([\d.]+)', line)
            if match:
                losses[key].append(float(match.group(1)))

plt.figure(figsize=(10, 5))
for key, values in losses.items():
    plt.plot(values, label=key)
plt.title('Training Loss Curves')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()

