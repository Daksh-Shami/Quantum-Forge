import subprocess
import time
import sys
import re
import argparse
from pyfiglet import Figlet
from termcolor import colored, cprint
import psutil
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Path to the QF executable
WORKSPACE_DIR = "C:/Users/User/quantum_forge_workspace"
QF_EXE_PATH = os.path.join(WORKSPACE_DIR, "target/release/quantum_compiler.exe")

# Modern color palette
COLORS = {
    'Qiskit (IBM)': '#4361EE',      # Rich blue
    'QF': '#7209B7',                # Deep purple
    'Cirq (Google)': '#F72585',     # Vibrant pink
    'PennyLane (Xanadu)': '#4CC9F0', # Light blue
    'tket (Quantinuum)': '#560BAD'  # Royal purple
}

TERMINAL_COLORS = {
    'Qiskit (IBM)': 'blue',
    'QF': 'magenta',
    'Cirq (Google)': 'red',
    'PennyLane (Xanadu)': 'cyan',
    'tket (Quantinuum)': 'magenta'
}

def set_style():
    # Set the style for better-looking plots
    plt.style.use('seaborn-v0_8-dark-palette')
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.5)
    
    # Custom matplotlib params
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'axes.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'grid.color': '#EEEEEE',
        'grid.linestyle': '-',
        'grid.linewidth': 1,
        'grid.alpha': 0.5
    })

def print_fancy_text(text, font='slant', color='cyan', attrs=['bold']):
    f = Figlet(font=font)
    cprint(f.renderText(text), color, attrs=attrs)

def extract_total_time(output):
    match = re.search(r"Total time for entire program:\s+(\d+\.\d+)", output)
    return float(match.group(1)) if match else None

def extract_peak_memory(output):
    match = re.search(r"Peak memory usage:\s+(\d+\.\d+)\s+MB", output)
    return float(match.group(1)) if match else None

def format_time(seconds):
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m {seconds % 60:.1f}s"
    elif seconds >= 60:
        return f"{int(seconds // 60)}m {seconds % 60:.1f}s"
    else:
        return f"{seconds:.3f}s"

def run_implementation(name, command, is_rust=False):
    print_fancy_text(f"Running {name}...", color=TERMINAL_COLORS[name])
    start_time = time.time()

    if is_rust:
        qasm_file = os.path.abspath(command[1])
        process = subprocess.Popen(
            [command[0], qasm_file, command[2]], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            universal_newlines=True,
            cwd=WORKSPACE_DIR
        )
        output, _ = process.communicate()
        end_time = time.time()
        peak_memory = extract_peak_memory(output) or 0
    else:
        process = psutil.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        max_memory = 0
        while process.poll() is None:
            mem_info = process.memory_info()
            max_memory = max(max_memory, mem_info.rss)
            time.sleep(0.1)
        output, _ = process.communicate()
        end_time = time.time()
        peak_memory = max_memory / (1024 * 1024)

    print(output)
    return output, end_time - start_time, peak_memory

def create_performance_visualization(results_df, output_file):
    # Set the style
    set_style()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create the scatter plot
    scatter = ax.scatter(
        results_df['Memory (MB)'],
        results_df['Time (s)'],
        s=200,  # Marker size
        c=[COLORS[comp] for comp in results_df['Compiler']],
        alpha=0.7,
        edgecolor='white',
        linewidth=2
    )
    
    # Add labels for each point
    for idx, row in results_df.iterrows():
        ax.annotate(
            row['Compiler'],
            (row['Memory (MB)'], row['Time (s)']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=11,
            alpha=0.8
        )
    
    # Customize the plot
    ax.set_xlabel('Memory Usage (MB)', fontsize=12, color='#333333')
    ax.set_ylabel('Execution Time (s)', fontsize=12, color='#333333')
    ax.set_title('Quantum Compiler Performance', 
                fontsize=16, 
                color='#333333', 
                pad=20,
                fontweight='bold')
    
    # Add efficiency arrows and labels
    ax_min_x = ax.get_xlim()[0]
    ax_max_y = ax.get_ylim()[1]
    
    # Add "Faster" arrow
    plt.annotate('FASTER →', 
                xy=(ax_min_x, results_df['Time (s)'].min()),
                xytext=(ax_min_x, results_df['Time (s)'].min() * 0.8),
                color='#666666',
                fontsize=10,
                alpha=0.7)
    
    # Add "Efficient" arrow
    plt.annotate('EFFICIENT →', 
                xy=(results_df['Memory (MB)'].min(), ax_max_y),
                xytext=(results_df['Memory (MB)'].min() * 0.8, ax_max_y),
                rotation=-90,
                color='#666666',
                fontsize=10,
                alpha=0.7)
    
    # Add trend line
    max_memory = results_df['Memory (MB)'].max()
    max_time = results_df['Time (s)'].max()
    plt.plot([0, max_memory], [0, max_time], 
             '--', 
             color='#CCCCCC', 
             alpha=0.5, 
             linewidth=1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run quantum circuit simulations with visualizations")
    parser.add_argument("qasm_file", help="Input QASM file")
    parser.add_argument("iterations", help="Number of iterations")
    parser.add_argument("--no-pennylane", action="store_true", help="Skip running PennyLane simulator")
    parser.add_argument("--no-tket", action="store_true", help="Skip running tket simulator")
    parser.add_argument("--output", default="benchmark_results.png", help="Output PNG file for visualization")
    args = parser.parse_args()

    results = []
    
    # Run implementations
    implementations = [
        ("Qiskit (IBM)", ["python", "qiskit_simulator.py", args.qasm_file, args.iterations], False),
        ("Cirq (Google)", ["python", "cirq_simulator.py", args.qasm_file, args.iterations], False),
        ("QF", [QF_EXE_PATH, args.qasm_file, args.iterations], True)
    ]

    if not args.no_pennylane:
        implementations.append(
            ("PennyLane (Xanadu)", ["python", "pennylane_simulator.py", args.qasm_file, args.iterations], False)
        )
    
    if not args.no_tket:
        implementations.append(
            ("tket (Quantinuum)", ["python", "tket_simulator.py", args.qasm_file, args.iterations], False)
        )

    for name, command, is_rust in implementations:
        output, total_time, peak_memory = run_implementation(name, command, is_rust)
        results.append({
            'Compiler': name,
            'Time (s)': extract_total_time(output) or total_time,
            'Memory (MB)': peak_memory
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create visualization
    create_performance_visualization(results_df, args.output)
    print(f"\nVisualization saved to {args.output}")

    # Find and display the fastest implementation
    fastest = results_df.loc[results_df['Time (s)'].idxmin()]
    print_fancy_text("Results Summary", color='cyan')
    cprint(f"Fastest Implementation: {fastest['Compiler']}", TERMINAL_COLORS[fastest['Compiler']], attrs=['bold'])
    cprint(f"Time: {format_time(fastest['Time (s)'])}", attrs=['bold'])
    cprint(f"Memory: {fastest['Memory (MB)']:.0f} MB", attrs=['bold'])

if __name__ == "__main__":
    main()