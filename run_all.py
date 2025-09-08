import subprocess
from utils import plot_comparisons, plot_cross_correlation
import time

start = time.time()
models = ['original', 'n2v', 'calibrated', 'n2n']

# Train
print("Training N2V model...")
subprocess.run(["python train.py -d Training -m n2v"], check=True, shell=True)
print("Training N2N model...")
subprocess.run(["python train.py -d Training -m n2n"], check=True, shell=True)

# Inference
print("Performing N2V inference...")
subprocess.run(["python inference.py -d Science -m n2v"], check=True, shell=True)
print("Performing N2N inference...")
subprocess.run(["python inference.py -d Science -m n2n"], check=True, shell=True)

# Evaluation
print("Evaluating N2V results...")
subprocess.run(["python evaluation.py -d DenoisedScience -m n2v"], check=True, shell=True)
print("Evaluating N2N results...")
subprocess.run(["python evaluation.py -d DenoisedScience -m n2n"], check=True, shell=True)

# Plotting
print("Generating plots...")
plot_comparisons(models)
plot_cross_correlation(models)

# Done
print("Done!")
end = time.time()
print(f"Elapsed time: {end - start:.2f} seconds")
