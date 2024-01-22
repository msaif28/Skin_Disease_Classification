# Assuming this script is running from the project root and the script, data, and models are all direct subdirectories.

cd /home/saifm/Skin_Disease_Classification-Jan2024  # navigate to the project root

# Repeat the whole process for 5 experiments
for i in {1..5}
do
    echo "Starting Experiment $i"

    # Step 1: Delete train, test, val directories
    rm -rf data/train data/test data/val

    # Step 2: Run data distribution script (correct the path as needed)
    python data/data_distribution.py

    # Step 3: Train models in parallel (correct the path as needed)
    bash scripts/train_parallel.sh

    # Step 4: Evaluate models and wait until it writes all metrics (correct the path as needed)
    python scripts/evaluate.py

    echo "Experiment $i completed"
done

echo "All experiments completed"

