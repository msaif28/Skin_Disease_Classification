import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Functions to read the classification report and the ROC AUC report
def read_classification_report(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = {'Category': [], 'Precision': [], 'Recall': [], 'F1-Score': []}
    for line in lines:
        parts = line.split()
        if len(parts) == 5 and parts[0] != 'accuracy':
            data['Category'].append(parts[0])
            data['Precision'].append(float(parts[1]))
            data['Recall'].append(float(parts[2]))
            data['F1-Score'].append(float(parts[3]))
    return data

def read_roc_auc_report(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    roc_auc_data = {}
    for line in lines:
        parts = line.split(':')
        category = parts[0].split()[-1]
        roc_auc = float(parts[1].strip())
        roc_auc_data[category] = roc_auc
    return roc_auc_data

# Base directory where the report files are located
base_dir = '/home/saifm/Experiments/Experiment01/scripts/'

# Find all classification report files
classification_files = glob.glob(os.path.join(base_dir, 'classification_report_*.txt'))

for classification_file in classification_files:
    model_name = os.path.basename(classification_file).split('classification_report_')[-1].split('.txt')[0]

    # Construct the corresponding ROC AUC file path
    roc_auc_file = os.path.join(base_dir, f'roc_auc_report_{model_name}.txt')
    
    if os.path.exists(roc_auc_file):
        # Read the data from the files
        classification_data = read_classification_report(classification_file)
        roc_auc_data = read_roc_auc_report(roc_auc_file)

        # Combine the data into a DataFrame
        df = pd.DataFrame(classification_data)
        df['ROC AUC'] = df['Category'].map(roc_auc_data)

        # Save the DataFrame for plotting
        combined_data_filepath = os.path.join(base_dir, f'combined_data_{model_name}.csv')
        df.to_csv(combined_data_filepath, index=False)

        # Plotting
        fig, ax = plt.subplots(figsize=(15, 8))

        # Bar chart for Precision, Recall, F1-Score
        df[['Precision', 'Recall', 'F1-Score']].plot(kind='bar', ax=ax, width=0.8)

        # Line chart for ROC AUC
        roc_auc_line = ax.plot(df['ROC AUC'], color='darkorange', marker='o', linewidth=2, label='ROC AUC')

        # Customizing the x-axis to show class names instead of numbers
        ax.set_xticklabels(df['Category'], rotation=90)

        # Labels and title
    ax.set_xlabel('Category')
    ax.set_ylabel('Precision, Recall, F1-Score')
    ax2 = ax.twinx()
    ax2.set_ylabel('ROC AUC')

    # Set the color and style of the ROC AUC line directly
    roc_auc_line[0].set_color('darkorange')  # Set a distinctive color for ROC AUC
    roc_auc_line[0].set_linewidth(2.5)       # Make the ROC AUC line thicker
    roc_auc_line[0].set_linestyle('--')      # Set the line style to dashed

    # Adding legends
    ax.legend(loc='upper left')
    ax2.legend(handles=roc_auc_line, loc='upper right')

    # Adjusting layout to fit everything
    plt.tight_layout()

    # Save the plot
    plot_filename = os.path.join(base_dir, f'performance_metrics_plot_with_class_names_{model_name}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')

    print(f"Plot saved as {plot_filename}")

    # Show plot
    plt.show()
else:
    print(f"No matching ROC AUC report file found for model {model_name}")

