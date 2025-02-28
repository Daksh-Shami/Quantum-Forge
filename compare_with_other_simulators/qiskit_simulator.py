from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import circuit_drawer
import time
import sys

def run_benchmark(qasm_file, iterations):
    # Load the circuit from the QASM file
    qc = QuantumCircuit.from_qasm_file(qasm_file)

    # Set up the Aer simulator
    simulator = AerSimulator()

    transpiled_qc = transpile(qc, simulator)

    measurement_counts = {}

    start_time = time.time()

    result = simulator.run(transpiled_qc, shots=iterations).result()
    counts = result.get_counts(transpiled_qc)

    end_time = time.time()

    total_execution_time = end_time - start_time

    for outcome, count in counts.items():
        measurement_counts[outcome] = measurement_counts.get(outcome, 0) + count


    return total_execution_time, measurement_counts

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