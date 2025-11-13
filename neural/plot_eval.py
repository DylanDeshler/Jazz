import json
import matplotlib.pyplot as plt

from collections import defaultdict

import numpy as np

with open('LAM_M_analysis.json', 'r') as f:
    res = json.load(f)

guidance_res = defaultdict(list)
alpha_res = defaultdict(list)
for k, v in res.items():
    guidance, alpha = k.split(',')
    guidance = int(guidance)
    alpha = float(alpha)
    
    guidance_res[guidance].extend(v)
    alpha_res[alpha].extend(v)
    

plt.plot(guidance_res.values(), [np.mean(v) for v in guidance_res.values()])
plt.show()