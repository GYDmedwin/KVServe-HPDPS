import json
import numpy as np
with open('config1.json', 'r', encoding='utf-8') as f:
    config1 = json.load(f)

with open('config2.json', 'r', encoding='utf-8') as f:
    config2 = json.load(f)

with open('config3.json', 'r', encoding='utf-8') as f:
    config3 = json.load(f)


list1 = np.array(config1['compression_ratio_list'])
list2 = np.array(config2['compression_ratio_list'])
list3 = np.array(config3['compression_ratio_list'])

# Check if all elements in list3 - list2 are greater than 0
diff_positive = np.all((list2 - list1) > 0)
print(f"All elements in list3 - list2 are greater than 0: {diff_positive}")
