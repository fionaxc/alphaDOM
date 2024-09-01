import csv
import os

def save_params_to_csv(params: dict, output_dir: str, filename: str = "params.csv"):
    """
    Save parameters to a CSV file.

    Args:
    params (dict): Dictionary containing parameter names and values.
    output_dir (str): Directory to save the CSV file.
    filename (str): Name of the CSV file (default is "params.csv").

    Returns:
    str: Full path to the saved CSV file.
    """
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Value"])
        for key, value in params.items():
            writer.writerow([key, value])
    
    return filepath

def stable_softmax(x):
    x = x - x.max(dim=-1, keepdim=True)[0]
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)