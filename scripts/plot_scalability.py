import matplotlib.pyplot as plt
import numpy as np
import os


csv_file = "tmp_data/timings.csv"

if not os.path.exists(csv_file):
    print(
        f"Error: {csv_file} not found!\nMake sure you have executed the script 'submit.sbatch' in the 'shell' folder."
    )
    exit(1)

data = np.genfromtxt(csv_file, delimiter=",", dtype=str, skip_header=1)


dims = data[:, 0].astype(int)
func_names = data[:, 1]
times = data[:, 2].astype(float)


unique_funcs = np.unique(func_names)

plt.figure(figsize=(8, 6))


for func in unique_funcs:
    mask = func_names == func
    plt.plot(dims[mask], times[mask], marker="o", linestyle="-", label=func)


plt.xlabel(r"$\log(d)$")
plt.ylabel("$t$")
plt.xscale("log")
plt.yscale("log")
plt.title("Scalability")

plt.legend()
plt.grid(True)
plt.tight_layout()


output_dir = "logs"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "scalability.png")
plt.savefig(output_path)
print(f"Plot saved to {output_path}")

plt.show()
plt.close()
plt.clf()
