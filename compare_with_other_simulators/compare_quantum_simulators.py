import subprocess
import time
import sys
import re
import argparse
from pyfiglet import Figlet
from termcolor import colored, cprint
from tabulate import tabulate
import os

# Path to the QF executable (Please change this to the root directory of your Quantum Forge workspace)
WORKSPACE_DIR = "/home/dakshshami/Desktop/quantum_forge_workspace"
QF_EXE_PATH = os.path.join(WORKSPACE_DIR, "target/release/quantum_compiler")

def print_fancy_text(text, font='slant', color='cyan', attrs=['bold']):
    f = Figlet(font=font)
    cprint(f.renderText(text), color, attrs=attrs)

def extract_total_time(output, implementation="default"):
    if implementation == "qf":
        match = re.search(r"Total time:\s+(\d+\.\d+)", output)
    else:
        match = re.search(r"Total time for entire program:\s+(\d+\.\d+)", output)
    return float(match.group(1)) if match else None

def format_time(seconds):
    if seconds >= 3600:
        return f"{int(seconds // 3600)} hours, {int((seconds % 3600) // 60)} minutes, {seconds % 60:.1f} seconds"
    elif seconds >= 60:
        return f"{int(seconds // 60)} minutes, {seconds % 60:.1f} seconds"
    else:
        return f"{seconds:.3f} seconds"

def run_python_implementation(name, command, color):
    print_fancy_text(f"Running {name}...", color=color)
    start_time = time.time()

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1
    )

    output, stderr = process.communicate()
    end_time = time.time()

    if stderr:
        print("Warning: stderr output:")
        print(stderr)

    if process.returncode != 0:
        print(f"Warning: Process returned non-zero exit code: {process.returncode}")

    print(output)
    return output, end_time - start_time

def run_rust_implementation(name, command, color):
    print_fancy_text(f"Running {name}...", color=color)
    start_time = time.time()

    qasm_file = os.path.abspath(command[1])
    process = subprocess.Popen(
        [command[0], qasm_file, command[2]], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        universal_newlines=True,
        cwd=WORKSPACE_DIR
    )
    output, stderr = process.communicate()
    end_time = time.time()

    if stderr:
        print("Warning: stderr output:")
        print(stderr)

    if process.returncode != 0:
        print(f"Warning: Process returned non-zero exit code: {process.returncode}")

    print(output)
    return output, end_time - start_time

def main():
    parser = argparse.ArgumentParser(description="Run quantum circuit simulations")
    parser.add_argument("qasm_file", help="Input QASM file")
    parser.add_argument("iterations", help="Number of iterations")
    parser.add_argument("--no-pennylane", action="store_true", help="Skip running PennyLane simulator")
    parser.add_argument("--no-tket", action="store_true", help="Skip running tket simulator")
    args = parser.parse_args()

    qasm_file = args.qasm_file
    iterations = args.iterations

    # Initialize results dictionary to store all implementation results
    results_data = {}

    # Run required implementations
    implementations = {
        "Qiskit (IBM)": {
            "command": ["python3", "qiskit_simulator.py", qasm_file, iterations],
            "color": "light_blue",
            "runner": run_python_implementation,
            "extract_time": "default"
        },
        "Cirq (Google)": {
            "command": ["python3", "cirq_simulator.py", qasm_file, iterations],
            "color": "red",
            "runner": run_python_implementation,
            "extract_time": "default"
        },
        "QF": {
            "command": [QF_EXE_PATH, qasm_file, iterations],
            "color": "magenta",
            "runner": run_rust_implementation,
            "extract_time": "qf"
        }
    }

    if not args.no_pennylane:
        implementations["PennyLane (Xanadu)"] = {
            "command": ["python3", "pennylane_simulator.py", qasm_file, iterations],
            "color": "yellow",
            "runner": run_python_implementation,
            "extract_time": "default"
        }

    if not args.no_tket:
        implementations["tket (Quantinuum)"] = {
            "command": ["python3", "tket_simulator.py", qasm_file, iterations],
            "color": "cyan",
            "runner": run_python_implementation,
            "extract_time": "default"
        }

    # Run implementations and collect results
    for name, impl in implementations.items():
        try:
            output, run_time = impl["runner"](
                name, 
                impl["command"], 
                impl["color"]
            )
            total_time = extract_total_time(output, impl["extract_time"])
            
            if total_time is None:
                print_fancy_text("Error", color='red')
                cprint(f"Could not extract total time from {name}. Output:", 'red', attrs=['bold'])
                cprint(output, 'red')
                input("Press Enter to exit...")
                sys.exit(1)

            results_data[name] = {
                "time": total_time,
                "color": impl["color"]
            }
        except Exception as e:
            print_fancy_text("Error", color='red')
            cprint(f"Failed to run {name}: {str(e)}", 'red', attrs=['bold'])
            input("Press Enter to exit...")
            sys.exit(1)

    # Create table results
    table_results = [
        [name, format_time(data["time"])]
        for name, data in results_data.items()
    ]

    print_fancy_text("Results", color='cyan')
    table = tabulate(table_results, headers=["Compiler", "Total Time"], tablefmt="fancy_grid")
    cprint(table, 'white')

    # Find fastest implementation
    fastest = min(results_data.items(), key=lambda x: x[1]["time"])
    fastest_name = fastest[0]
    fastest_data = fastest[1]

    # Print fastest result
    print_fancy_text("Fastest", color=fastest_data["color"])
    cprint(f"{colored(fastest_name.upper(), fastest_data['color'])} IS FASTEST!", attrs=['bold', 'blink'])

    # Calculate and print comparisons
    for name, data in results_data.items():
        if name != fastest_name:
            speed_ratio = data["time"] / fastest_data["time"]
            comparison = (
                f"{colored(fastest_name, fastest_data['color'])} is "
                f"{colored(f'{speed_ratio:.1f}', data['color'])}x faster than "
                f"{colored(name, data['color'])}."
            )
            cprint(comparison, attrs=['bold'])

    cprint("=" * 50, fastest_data["color"], attrs=['bold'])

if __name__ == "__main__":
    main()