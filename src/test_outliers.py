import time
from tqdm import tqdm


"""
            COOL PROGRESS BAR
"""

for i in tqdm(range(100000), desc='SIM', mininterval=1):
    time.sleep(0.01)
