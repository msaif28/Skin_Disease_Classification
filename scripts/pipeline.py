import subprocess
import os
import multiprocessing

# Define the command for each model training
commands = [
    "python train.py --model_name AlexNet --gpu 0 > /var/tmp/logs/AlexNet.log 2>&1",
    "python train.py --model_name VGG --gpu 0 > /var/tmp/logs/VGG.log 2>&1",
    "python train.py --model_name ResNet --gpu 0 > /var/tmp/logs/ResNet.log 2>&1",
    "python train.py --model_name EfficientNet --gpu 0 > /var/tmp/logs/EfficientNet.log 2>&1",
    "python vit_training.py --model_name ViT --gpu 0 > /var/tmp/logs/ViT.log 2>&1"
]

def run_command(cmd):
    """Run a command using subprocess.run"""
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"Completed: {cmd}")
    except subprocess.CalledProcessError:
        print(f"Failed: {cmd}")


def main():
    os.chdir("/home/saifm/Skin_Disease_Classification-Jan2024")

    for i in range(1, 6):  # Looping for 5 experiments
        print(f"Starting Experiment {i}")

        # Deleting directories
        print("Removing train, test, and val directories if they exist...")
        subprocess.run("rm -rf data/train data/test data/val", shell=True)
        print("Directories removed.")

        # Distributing data
        print("Distributing data into train, test, and val directories...")
        subprocess.run("python data/data_distribution.py", shell=True)
        print("Data distribution completed.")

        # Training models in parallel
        print("Training models in parallel...")
        with multiprocessing.Pool(processes=len(commands)) as pool:
            pool.map(run_command, commands)

        print("All training jobs completed.")
        print("Model training completed.")

        # Evaluating models
        print("Evaluating models...")
        subprocess.run("python scripts/evaluate.py", shell=True)
        print("Model evaluation completed.")

        print(f"Experiment {i} completed.")

    print("All experiments completed.")


if __name__ == "__main__":
    main()
