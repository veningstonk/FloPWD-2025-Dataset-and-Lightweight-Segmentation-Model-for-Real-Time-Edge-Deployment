import pandas as pd
from pathlib import Path
import numpy as np

# Define the base directory and model names
base_dir = Path("D:\\Suhaib\\DalLake")
model_names = ['ENet', 'CustomENet', 'CustomENetNoSE', 'CustomENetNoDW', 'CustomENetNoAsym']
output_csv_path = base_dir / "ablation_comparison.csv"

# Initialize a list to store summary metrics
summary_data = []

# Process each model's metrics
for model_name in model_names:
    metrics_path = base_dir / f"output_{model_name}" / "training_metrics.csv"
    stats_path = base_dir / f"output_{model_name}" / "model_stats.txt"
    
    # Check if metrics file exists
    if not metrics_path.exists():
        print(f"Metrics file for {model_name} not found at {metrics_path}")
        continue
    
    # Read the metrics CSV
    df = pd.read_csv(metrics_path)
    
    # Find the epoch with the best val_mIoU
    best_idx = df['val_mIoU'].idxmax()
    best_metrics = df.iloc[best_idx]
    
    # Read FLOPS and params from model_stats.txt
    flops = 0.0
    params = 0.0
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("FLOPS"):
                    flops = float(line.split(":")[1].strip())
                if line.startswith("Parameters"):
                    params = float(line.split(":")[1].strip())
    else:
        print(f"Stats file for {model_name} not found at {stats_path}")
    
    # Extract relevant metrics
    summary = {
        'Model': model_name,
        'Best_Val_mIoU': best_metrics['val_mIoU'],
        'Best_Val_mPA': best_metrics['val_mPA'],
        'Best_Val_Loss': best_metrics['val_loss'],
        'Best_Val_FPS': best_metrics['val_fps'],
        'FLOPS (GFLOPS)': flops,
        'Parameters (M)': params
    }
    
    summary_data.append(summary)

# Create a DataFrame from the summary data
summary_df = pd.DataFrame(summary_data)

# Identify the best configuration based on val_mIoU, val_mPA, val_fps, and val_loss
if not summary_df.empty:
    # Normalize metrics (mIoU, mPA, and FPS should be maximized, loss should be minimized)
    norm_mIoU = (summary_df['Best_Val_mIoU'] - summary_df['Best_Val_mIoU'].min()) / (summary_df['Best_Val_mIoU'].max() - summary_df['Best_Val_mIoU'].min())
    norm_mPA = (summary_df['Best_Val_mPA'] - summary_df['Best_Val_mPA'].min()) / (summary_df['Best_Val_mPA'].max() - summary_df['Best_Val_mPA'].min())
    norm_FPS = (summary_df['Best_Val_FPS'] - summary_df['Best_Val_FPS'].min()) / (summary_df['Best_Val_FPS'].max() - summary_df['Best_Val_FPS'].min())
    norm_Loss = (summary_df['Best_Val_Loss'].max() - summary_df['Best_Val_Loss']) / (summary_df['Best_Val_Loss'].max() - summary_df['Best_Val_Loss'].min())
    
    # Calculate a composite score (equal weights for simplicity)
    composite_score = (norm_mIoU + norm_mPA + norm_FPS + norm_Loss) / 4
    
    # Print the calculation method
    print("\nCalculation for Best Model Selection:")
    print("1. Normalize Best_Val_mIoU: (x - min(mIoU)) / (max(mIoU) - min(mIoU))")
    print("2. Normalize Best_Val_mPA: (x - min(mPA)) / (max(mPA) - min(mPA))")
    print("3. Normalize Best_Val_FPS: (x - min(FPS)) / (max(FPS) - min(FPS))")
    print("4. Normalize Best_Val_Loss: (max(Loss) - x) / (max(Loss) - min(Loss))")
    print("5. Composite Score: (Norm_mIoU + Norm_mPA + Norm_FPS + Norm_Loss) / 4")
    print("6. Select model with the highest Composite Score\n")
    
    # Print computed values
    print("Computed Values:")
    print(f"{'Model':<15} {'mIoU':<10} {'Norm_mIoU':<12} {'mPA':<10} {'Norm_mPA':<12} {'FPS':<10} {'Norm_FPS':<12} {'Loss':<10} {'Norm_Loss':<12} {'Composite Score':<15}")
    print("-" * 100)
    for idx, row in summary_df.iterrows():
        print(f"{row['Model']:<15} {row['Best_Val_mIoU']:<10.4f} {norm_mIoU[idx]:<12.4f} {row['Best_Val_mPA']:<10.4f} {norm_mPA[idx]:<12.4f} {row['Best_Val_FPS']:<10.2f} {norm_FPS[idx]:<12.4f} {row['Best_Val_Loss']:<10.4f} {norm_Loss[idx]:<12.4f} {composite_score[idx]:<15.4f}")
    
    # Find the model with the highest composite score
    best_model_idx = composite_score.idxmax()
    best_model = summary_df.loc[best_model_idx]
    
    print(f"\nBest Configuration: {best_model['Model']}")
    print(f"Best Val mIoU: {best_model['Best_Val_mIoU']:.4f}")
    print(f"Best Val mPA: {best_model['Best_Val_mPA']:.4f}")
    print(f"Best Val Loss: {best_model['Best_Val_Loss']:.4f}")
    print(f"Best Val FPS: {best_model['Best_Val_FPS']:.2f}")
    print(f"FLOPS: {best_model['FLOPS (GFLOPS)']:.2f} GFLOPS")
    print(f"Parameters: {best_model['Parameters (M)']:.2f} M")
else:
    print("No metrics found for any model.")

# Save the summary to a CSV file (excluding normalized values)
summary_df.to_csv(output_csv_path, index=False)
print(f"\nSaved ablation comparison to {output_csv_path}")

# Print the summary table (excluding normalized values)
print("\nAblation Study Summary:")
print(summary_df[['Model', 'Best_Val_mIoU', 'Best_Val_mPA', 'Best_Val_Loss', 'Best_Val_FPS', 'FLOPS (GFLOPS)', 'Parameters (M)']].to_string(index=False))