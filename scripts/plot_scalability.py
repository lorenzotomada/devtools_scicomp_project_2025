import matplotlib.pyplot as plt
import pandas as pd
import os

csv_file = "logs/timings.csv"

if not os.path.exists(csv_file):
    print(
        f"Error: {csv_file} not found!\nMake sure you have executed the script 'submit.sbatch' in the 'shell' folder."
    )
    exit(1)

df = pd.read_csv(csv_file)

dims = df["dim"].astype(int)
func_names = df["function"]
times = df["time"].astype(float)

unique_funcs = df["function"].unique()

plt.figure(figsize=(8, 6))

for func in unique_funcs:
    subset = df[df["function"] == func]
    print(f"{func}: {subset['dim'].values}")
    plt.plot(subset["dim"], subset["time"], marker="o", linestyle="-", label=func)

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
