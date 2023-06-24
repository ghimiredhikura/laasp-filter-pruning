import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Define the two dictionaries
# Imagenet 
# Resnet18
dict1 = {'conv1': 64, 'layer1.0.conv1': 64, 'layer1.0.conv2': 64, 'layer1.1.conv1': 64, 'layer1.1.conv2': 64, 'layer2.0.conv1': 128, 'layer2.0.conv2': 128, 'layer2.0.downsample.0': 128, 'layer2.1.conv1': 128, 'layer2.1.conv2': 128, 'layer3.0.conv1': 256, 'layer3.0.conv2': 256, 'layer3.0.downsample.0': 256, 'layer3.1.conv1': 256, 'layer3.1.conv2': 256, 'layer4.0.conv1': 512, 'layer4.0.conv2': 512, 'layer4.0.downsample.0': 512, 'layer4.1.conv1': 512, 'layer4.1.conv2': 512}
dict2 = {'conv1': 58, 'layer1.0.conv1': 39, 'layer1.0.conv2': 58, 'layer1.1.conv1': 24, 'layer1.1.conv2': 58, 'layer2.0.conv1': 102, 'layer2.0.conv2': 128, 'layer2.0.downsample.0': 128, 'layer2.1.conv1': 48, 'layer2.1.conv2': 128, 'layer3.0.conv1': 229, 'layer3.0.conv2': 245, 'layer3.0.downsample.0': 245, 'layer3.1.conv1': 96, 'layer3.1.conv2': 245, 'layer4.0.conv1': 188, 'layer4.0.conv2': 434, 'layer4.0.downsample.0': 434, 'layer4.1.conv1': 392, 'layer4.1.conv2': 434}

# Get the unique keys in the dictionaries and sort them
#keys = list(set(dict1.keys()).union(set(dict2.keys())))
keys = list(dict1.keys())

# Create an array of heights for each bar, with 0 if a key is not present in dict2
heights = np.array([dict2.get(key, 0) for key in keys])

# Create a list of colors, with red for bars where dict2 has a value and blue otherwise
colors = ['#7a2b22' if dict2.get(key) is not None else 'g' for key in keys]

# Set font properties
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12

# Create the horizontal bar chart
fig, ax = plt.subplots(figsize=(10,8), dpi=180)
#ax.set_yticks(range(0, 60, 5))

#plt.xticks(rotation=60,ha='right')
labels = ax.get_xticklabels()
for label in labels:
    label.set_rotation(60)
    label.set_horizontalalignment('right')
    label.set_verticalalignment('top')

#ax.set_ylim(bottom=0) # Set the lower limit of y-axis to zero
ax.tick_params(axis='y', which='both', length=0, pad=10, labelcolor='black', zorder=1) # Hide the ticks and change their color to match the bars
# Add a grid
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Plot the bar chart
plt.bar(keys, list(dict1.values()), width=0.5,bottom=0.1, color='#393b3e', alpha=0.5, zorder=2)
plt.bar(keys, heights, color=colors, width=0.5,bottom=0.1, zorder=2)

ax.set_ylabel('Total Number of Filters')

plt.subplots_adjust(left=0.1, right=0.9, bottom=0.25, top=0.9)
# Add a title to the plot
#ax.set_title('Number of Filters in Different Layers')

# Remove the top and right spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Show the plot
plt.tight_layout()

# Add a legend
plt.legend(['Pruned Filters', 'Retained Filters'])

# Show the plot
plt.show()