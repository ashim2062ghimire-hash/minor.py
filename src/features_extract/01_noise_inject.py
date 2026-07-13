d = (5/16) 
D = 1.537
n = 9
theta = 0.0
rpm=1797
import numpy as np
fr = rpm / 60.0
ratio = (d / D) * np.cos(theta)

bpfo = (n / 2) * fr * (1 - ratio)
bpfi = (n / 2) * fr * (1 + ratio)
bsf = (D / (2 * d)) * fr * (1 - ratio ** 2)

print(f"bpfo={bpfo}")
print(f"bpfi={bpfi}")
print(f"bsf={bsf}")