import subprocess
from utils import plot_comparisons

# # Train
# print("Starting training...")
# subprocess.run(["python", "train.py", "-d", "Training", "-m", "cnn"], check=True)

# # Inference
# print("Running inference...")
# subprocess.run(["python", "inference.py", "-d", "Training", "-m", "cnn"], check=True)

# # Evaluation
# print("Evaluating results...")
# subprocess.run(["python", "evaluation.py", "-d", "Training", "-m", "cnn"], check=True)

# # Plotting
# print("Generating plots...")
plot_comparisons(['n2v', 'standard'])

print("Done!")
