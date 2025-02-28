from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt
import argparse

def draw_circuit_from_qasm(qasm_file):
    """
    Reads a qasm file and generates a circuit drawing with a black background.

    Args:
      qasm_file: Path to the qasm file.
    """
    # Create a QuantumCircuit object from the qasm file
    circuit = QuantumCircuit.from_qasm_file(qasm_file)

    # Draw the circuit with a black background using circuit_drawer
    fig = circuit_drawer(circuit, fold=-1, output='mpl', style={
        'backgroundcolor': 'black',
        'textcolor': 'white',
        'subtextcolor': 'white',
        'linecolor': 'white',
        'creglinecolor': 'white',
        'gatetextcolor': 'black',
        'gatefacecolor': 'white',
    })

    # Display the figure
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Draw quantum circuit from QASM file')
    parser.add_argument('qasm_file', type=str,
                       help='Path to the QASM file')
    
    args = parser.parse_args()
    draw_circuit_from_qasm(args.qasm_file)

if __name__ == '__main__':
    main()
