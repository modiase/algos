import math
import random as rn
from pathlib import Path

for i in range(1, 10):
    print(f'Generating inputs: {i}')
    a_val = 1
    b_val = 1
    a = []
    b = []

    threshold = 1 - min(0.5, 10*math.pow(10, -i))
    for _ in range(int(math.pow(10,i))):
        if rn.random() > threshold:
            a_val += 1
        if rn.random() > threshold:
            b_val += 1
        a.append(str(a_val))
        b.append(str(b_val))
    a_path = Path(f'.tmp/a_{i}.txt')
    b_path = Path(f'.tmp/b_{i}.txt')
    a_path.write_text(' '.join(a))
    b_path.write_text(' '.join(b))

