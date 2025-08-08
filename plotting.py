import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_metrics(csv_path, output_dir):
    # Read the CSV file
    metrics = pd.read_csv(csv_path)
    
    # Create plots directory if it doesn't exist
    plot_dir = output_dir / "plots_2"
    plot_dir.mkdir(exist_ok=True)
    
    # Create figure with two subplots
    plt.figure(figsize=(12, 5))
    
    # Plot mIoU
    plt.subplot(1, 2, 1)
    plt.plot(metrics['epoch'], metrics['train_mIoU'], label='Train mIoU')
    plt.plot(metrics['epoch'], metrics['val_mIoU'], label='Val mIoU')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.title('Training and Validation mIoU')
    plt.legend()
    plt.grid(True)
    
    # Plot FPS
    plt.subplot(1, 2, 2)
    plt.plot(metrics['epoch'], metrics['train_fps'], label='Train FPS')
    plt.plot(metrics['epoch'], metrics['val_fps'], label='Val FPS')
    plt.xlabel('Epoch')
    plt.ylabel('FPS')
    plt.title('Training and Validation FPS')
    plt.legend()
    plt.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(plot_dir / "training_plots.png")
    plt.close()

if __name__ == "__main__":
    output_dir = Path("D:\\Suhaib\\DalLake\\output(ENet-50epochs)")  # Adjust this path as needed
    csv_path = output_dir / "training_metrics.csv"
    plot_metrics(csv_path, output_dir)