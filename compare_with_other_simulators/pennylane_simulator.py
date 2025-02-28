import pennylane as qml
from pennylane import numpy as np
import time
import sys
from typing import Dict, Tuple

# Import Qiskit for QASM parsing
from qiskit import QuantumCircuit

def run_benchmark(qasm_file: str, iterations: int) -> Tuple[float, Dict[str, int]]:
    # Load the QASM file using Qiskit
    try:
        with open(qasm_file, 'r') as f:
            qasm_str = f.read()
        circuit_qiskit = QuantumCircuit.from_qasm_str(qasm_str)
        num_qubits = circuit_qiskit.num_qubits
    except FileNotFoundError:
        print(f"Error: File '{qasm_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading QASM file: {e}")
        sys.exit(1)

    # Convert the QASM string into a PennyLane quantum function
    try:
        loaded_circuit = qml.from_qasm(qasm_str)
    except Exception as e:
        print(f"Error converting QASM to PennyLane: {e}")
        sys.exit(1)

    # Create the device with the correct number of wires
    dev = qml.device('default.qubit', wires=num_qubits)

    @qml.qnode(dev)
    def qnode():
        # Apply the loaded circuit to the specified wires
        loaded_circuit(wires=range(num_qubits))
        # Return probabilities for all computational basis states
        return qml.probs(wires=range(num_qubits))

    # Initialize variables to measure execution time and track results
    total_execution_time = 0.0
    measurement_counts: Dict[str, int] = {}

    # Run the circuit multiple times, measuring time and accumulating counts
    for _ in range(iterations):
        # Record the start time before execution
        start_time = time.time()

        # Run the quantum circuit and get the probability distribution
        probs = qnode()
        
        # Sample from the probability distribution
        outcome = np.random.choice(2**num_qubits, p=probs)
        outcome_str = format(outcome, f'0{num_qubits}b')
        
        # Update counts
        measurement_counts[outcome_str] = measurement_counts.get(outcome_str, 0) + 1

        # Record the end time after execution
        end_time = time.time()
        total_execution_time += (end_time - start_time)

    # Calculate average execution time per circuit execution
    average_time = total_execution_time / iterations

    return average_time, measurement_counts

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pennylane_simulator.py <input_file.qasm> <iterations>")
        sys.exit(1)

    qasm_file = sys.argv[1]
    try:
        iterations = int(sys.argv[2])
        if iterations <= 0:
            raise ValueError
    except ValueError:
        print("Error: iterations must be a positive integer")
        sys.exit(1)

    start_time = time.time()
    average_time, counts = run_benchmark(qasm_file, iterations)
    end_time = time.time()
    total_time = end_time - start_time

    # Print results
    print(f"\nCircuit Benchmark Results:")
    print(f"Number of iterations: {iterations}")
    print(f"Average time per execution: {average_time:.6f} seconds")
    print("Measurement results:")
    total_counts = sum(counts.values())
    for outcome, count in sorted(counts.items()):
        percentage = (count / total_counts) * 100
        print(f"{outcome}: {count} ({percentage:.2f}%)")
    print(f"Total time for entire program: {total_time:.6f} seconds")