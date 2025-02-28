from cirq.contrib.qasm_import import circuit_from_qasm
import qsimcirq
import numpy as np
import time
import sys
from typing import Dict, Tuple

def run_benchmark(qasm_file: str, iterations: int) -> Tuple[float, Dict[str, int]]:
    # Load the circuit from the QASM file
    with open(qasm_file, 'r') as file:
        qasm_str = file.read()

    # Convert QASM to Cirq circuit
    circuit = circuit_from_qasm(qasm_str)

    # Set up the Cirq simulator
    simulator = qsimcirq.QSimSimulator()

    # Initialize variables to measure execution time and track results
    total_execution_time = 0
    measurement_counts: Dict[str, int] = {}

    # Run the circuit multiple times, measuring time and accumulating counts
    for _ in range(iterations):
        # Record the start time before execution
        start_time = time.time()

        # Run the quantum circuit with 1 shot
        result = simulator.run(circuit, repetitions=1)

        # Record the end time after execution
        end_time = time.time()

        # Calculate the execution time for this iteration and add to the total
        total_execution_time += (end_time - start_time)

        # Accumulate measurement results for this iteration
        outcome = ''.join(str(np.array(v, dtype=int).item()) for v in result.measurements.values())
        measurement_counts[outcome] = measurement_counts.get(outcome, 0) + 1

    # Calculate average execution time per circuit execution
    average_time = total_execution_time / iterations

    # Return the average execution time and accumulated measurement counts
    return average_time, measurement_counts

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file.qasm> <iterations>")
        sys.exit(1)

    qasm_file = sys.argv[1]
    try:
        iterations = int(sys.argv[2])
    except ValueError:
        print("Error: iterations must be an integer")
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
    for outcome, count in counts.items():
        print(f"{outcome}: {count} ({count/sum(counts.values())*100:.2f}%)")
    print(f"Total time for entire program: {total_time:.6f} seconds")