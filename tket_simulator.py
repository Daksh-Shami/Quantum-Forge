from pytket import Circuit
from pytket.qasm import circuit_from_qasm, circuit_to_qasm
from pytket.extensions.qiskit import AerBackend
import time
import sys

def run_benchmark(qasm_file, iterations):
    # Load the circuit from QASM file
    qc = circuit_from_qasm(qasm_file)
    
    # Set up the simulator backend
    backend = AerBackend()
    
    # Compile circuit for the backend
    compiled_circ = backend.get_compiled_circuit(qc)
    
    total_execution_time = 0
    measurement_counts = {}
    
    # Run circuit once (similar to Qiskit example)
    for _ in range(1):
        start_time = time.time()
        result = backend.run_circuit(compiled_circ, n_shots=iterations)
        counts = result.get_counts()
        end_time = time.time()
        total_execution_time += (end_time - start_time)
        
        # Aggregate counts
        for outcome, count in counts.items():
            measurement_counts[outcome] = measurement_counts.get(outcome, 0) + count
    
    average_time = total_execution_time / 1
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