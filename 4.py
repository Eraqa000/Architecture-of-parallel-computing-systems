import math
import matplotlib.pyplot as plt

M = 1e6
B = 1e8
T_latency = 1e-5

base = T_latency = M/B
ps = [4, 8, 16, 32, 64]

def T_ring(p):
    return (p-1)*base

def T_mesh(p):
    retutn (math.sqrt(p)-1)*2*base

def T_hypercube(p):
    return math.log2(p)*base

def T_fully(p):
    return base

plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.plot(ps, [T_ring(p) for p in ps], marker = 'o', color='blue')
plt.title("Сақина (Ring)")
plt.xlabel("процессор саны (p)")
plt.ylabel("Tcomm (c)")
plt.grid(true)

plt.subplot(2,2,2)
plt.plot(ps, [T_mesh(p) for p in ps], marker = 's', color='green')
plt.title("Тор (Mesh)")
plt.xlabel("процессор саны (p)")
plt.ylabel("Tcomm (c)")
plt.grid(true)

plt.subplot(2,2,3)
plt.plot(ps, [T_hypercube(p) for p in ps], marker = '^', color='orange')
plt.title("Гиперкуб (hypercube)")
plt.xlabel("процессор саны (p)")
plt.ylabel("Tcomm (c)")
plt.grid(true)


plt.subplot(2,2,4)
plt.plot(ps, [T_fully(p) for p in ps], marker = 'x', color='red')
plt.title("Толық байланыс (Fully connected)")
plt.xlabel("процессор саны (p)")
plt.ylabel("Tcomm (c)")
plt.grid(true)

plt.tight_layout()
plt.show()
