import os
OUTPUTS_DIR = os.getenv('VH_OUTPUTS_DIR', './outputs')
path = os.path.join(OUTPUTS_DIR, 'ok.png')

import matplotlib.pyplot as plt
plt.plot([1,2,3], [4,5,6])
plt.savefig(path)

