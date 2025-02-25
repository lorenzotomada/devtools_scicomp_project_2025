import matplotlib.pyplot as plt
import numpy as np
import os

csv_file = "tmp_data/timings.csv"

if not os.path.exists(csv_file):
    print(f"Error: {csv_file} not found!\nMake sure you have executed the script 'scalability.sh' in the 'shell' folder.")
    exit(1)


data = np.genfromtxt(csv_file, delimiter=",", dtype=str, skip_header=1)


N = data[:, 0].astype(int)
backend = data[:, 1]
time = data[:, 2].astype(float)
backends = np.unique(backend)


plt.figure(figsize=(8, 6))
for b in backends:
    mask = backend == b
    plt.plot(N[mask], time[mask], linestyle="-", label=b)


plt.xlabel(r"$\log(d)$")
plt.ylabel("$t$")
plt.xscale('log')
#plt.yscale('log')
plt.title("Scalability: numba vs numpy")


plt.legend()
plt.grid(True)
plt.tight_layout()


output_dir = "logs"
output_path = os.path.join(output_dir, "scalability.png")
plt.savefig(output_path)
print(f"Plot saved to {output_path}")


plt.show()
plt.close()
plt.clf()