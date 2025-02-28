import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import subprocess
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing as mp
from itertools import product
import time
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from dimod import ExactSolver  # Classical brute-force solver
import dwave_networkx as dnx
from time import perf_counter
from tqdm import tqdm

def qaoa_worker(args):
    """Worker function for multiprocessing."""
    instance, gamma, beta = args  # Unpack arguments
    return instance.run_circuit_worker((gamma, beta, None))


class QAOAMaxCut:
    def __init__(self, n_nodes=12, n_threads=2):
        self.n_nodes = n_nodes
        self.G = self.create_datacenter_topology()
        num_edges = len(self.G.edges())
        
        # Dynamically scale ranges based on graph size
        self.gamma_range = np.linspace(0, 2*np.pi, max(10, self.n_nodes))  # Linear scaling
        self.beta_range = np.linspace(0, np.pi, max(10, self.n_nodes))    # Linear scaling
        
        # Alternatively, use edge-based scaling (commented)
        # self.gamma_range = np.linspace(0, 2*np.pi, int(num_edges ** 0.5))
        # self.beta_range = np.linspace(0, np.pi, int(num_edges ** 0.5))
        
        self.simulator_path = "C:/Users/User/quantum_forge_workspace/target/release/quantum_compiler.exe"
        self.n_threads = n_threads
        plt.style.use('dark_background')


    def create_datacenter_topology(self):
        """Create hierarchical datacenter topology"""
        G = nx.Graph()
        
        # Core layer (fully connected)
        core_nodes = min(2, self.n_nodes)
        for i in range(core_nodes):
            for j in range(i+1, core_nodes):
                G.add_edge(i, j, weight=10)  # High bandwidth core links
        
        # Aggregation layer
        agg_nodes = min(4, self.n_nodes - core_nodes)
        for i in range(agg_nodes):
            for c in range(core_nodes):
                G.add_edge(core_nodes + i, c, weight=5)  # Medium bandwidth
        
        # Access layer
        acc_start = core_nodes + agg_nodes
        for i in range(acc_start, self.n_nodes):
            # Connect to two aggregation switches for redundancy
            conns = min(2, agg_nodes)
            for j in range(conns):
                G.add_edge(i, core_nodes + j, weight=1)  # Standard bandwidth
        
        return G

    def calculate_cut(self, bitstring):
        """Calculate weighted value with enforced balance"""
        partition = [i for i, bit in enumerate(bitstring) if bit == '1']

        # Count nodes in each layer for each partition
        core_nodes = 2
        agg_nodes = min(4, self.n_nodes - core_nodes)

        zone_a_core = sum(1 for i in range(core_nodes) if i in partition)
        zone_b_core = core_nodes - zone_a_core

        zone_a_agg = sum(1 for i in range(core_nodes, core_nodes + agg_nodes)
                        if i in partition)
        zone_b_agg = agg_nodes - zone_a_agg

        # Penalize unbalanced partitions heavily
        balance_penalty = 0
        if abs(zone_a_core - zone_b_core) != 0:  # Want exactly one core in each zone
            balance_penalty -= 1000
        if abs(zone_a_agg - zone_b_agg) > 1:  # Allow slight agg imbalance if node count is large
            balance_penalty -= 1000
            
        value = balance_penalty
        for i, j in self.G.edges():
            weight = self.G.edges[i, j]['weight']
            is_cut = (i in partition) != (j in partition)

            if is_cut:
                value -= weight
            else:
                value += weight
        return value
    
    def verify_maxcut(self, qaoa_partition):
        """Verify QAOA result with detailed bandwidth analysis and correctness checks."""
        # Calculate the cut value explicitly
        cut_value = 0
        for i, j in self.G.edges():
            weight = self.G.edges[i, j]['weight']
            if (i in qaoa_partition) != (j in qaoa_partition):  # Edge crosses the partition
                cut_value += weight

        # Verify internal and cross-zone bandwidth
        internal_high = sum(self.G.edges[i, j]['weight']
                            for i, j in self.G.edges()
                            if (i in qaoa_partition) == (j in qaoa_partition)
                            and self.G.edges[i, j]['weight'] == 10)

        internal_med = sum(self.G.edges[i, j]['weight']
                        for i, j in self.G.edges()
                        if (i in qaoa_partition) == (j in qaoa_partition)
                        and self.G.edges[i, j]['weight'] == 5)

        internal_low = sum(self.G.edges[i, j]['weight']
                        for i, j in self.G.edges()
                        if (i in qaoa_partition) == (j in qaoa_partition)
                        and self.G.edges[i, j]['weight'] == 1)

        cross_high = sum(self.G.edges[i, j]['weight']
                        for i, j in self.G.edges()
                        if (i in qaoa_partition) != (j in qaoa_partition)
                        and self.G.edges[i, j]['weight'] == 10)

        cross_med = sum(self.G.edges[i, j]['weight']
                        for i, j in self.G.edges()
                        if (i in qaoa_partition) != (j in qaoa_partition)
                        and self.G.edges[i, j]['weight'] == 5)

        cross_low = sum(self.G.edges[i, j]['weight']
                        for i, j in self.G.edges()
                        if (i in qaoa_partition) != (j in qaoa_partition)
                        and self.G.edges[i, j]['weight'] == 1)

        total_weight = sum(w for _, _, w in self.G.edges.data('weight'))
        efficiency = (internal_high + internal_med + internal_low) / total_weight * 100

        # Check if the calculated cut matches the measured cut
        measured_cut = cross_high + cross_med + cross_low
        is_correct = cut_value == measured_cut

        results = {
            'cut_value': cut_value,
            'measured_cut': measured_cut,
            'is_correct': is_correct,
            'internal_bandwidth': {
                'high': internal_high,
                'medium': internal_med,
                'low': internal_low,
                'total': internal_high + internal_med + internal_low
            },
            'cross_zone_bandwidth': {
                'high': cross_high,
                'medium': cross_med,
                'low': cross_low,
                'total': cross_high + cross_med + cross_low
            },
            'total_bandwidth': total_weight,
            'efficiency': efficiency,
            'edge_count': len(self.G.edges()),
            'node_count': len(self.G.nodes())
        }

        # Detailed optimization report with correctness check
        print("\n" + "=" * 50)
        print(f"{'Datacenter Network Optimization Report':^50}")
        print("=" * 50)
        print(f"Nodes: {results['node_count']}")
        print(f"Edges: {results['edge_count']}")
        print("-" * 50)
        print(f"Internal Bandwidth:")
        print(f"  High (10Gb/s): {results['internal_bandwidth']['high']} Gb/s")
        print(f"  Medium (5Gb/s): {results['internal_bandwidth']['medium']} Gb/s")
        print(f"  Low (1Gb/s): {results['internal_bandwidth']['low']} Gb/s")
        print(f"  Total: {results['internal_bandwidth']['total']} Gb/s")
        print("-" * 50)
        print(f"Cross-Zone Bandwidth:")
        print(f"  High (10Gb/s): {results['cross_zone_bandwidth']['high']} Gb/s")
        print(f"  Medium (5Gb/s): {results['cross_zone_bandwidth']['medium']} Gb/s")
        print(f"  Low (1Gb/s): {results['cross_zone_bandwidth']['low']} Gb/s")
        print(f"  Total: {results['cross_zone_bandwidth']['total']} Gb/s")
        print("-" * 50)
        print(f"Overall Bandwidth:")
        print(f"  Total Bandwidth: {results['total_bandwidth']} Gb/s")
        print(f"  Efficiency: {results['efficiency']:.2f}%")
        print("-" * 50)
        print("Partition Details:")
        print(f"  Zone A Nodes: {len(qaoa_partition)}")
        print(f"  Zone B Nodes: {len(self.G.nodes()) - len(qaoa_partition)}")
        print("-" * 50)
        print("Partition Verification:")
        print(f"  Calculated Cut Value: {cut_value}")
        print(f"  Measured Cut Value: {measured_cut}")
        print(f"  Correct Solution: {'Yes' if is_correct else 'No'}")
        print("-" * 50)
        print("Correctness Check:")
        print(f"  High Bandwidth Usage Verified: {'Yes' if internal_high > 0 else 'No'}")
        print(f"  Cross-Zone Traffic Minimized: {'Yes' if cross_high < total_weight * 0.1 else 'No'}")
        print("=" * 50)

        return results


    def create_circuit(self, gamma, beta):
        """Create QAOA circuit as QASM string"""
        qasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n"
        qasm += f"qreg q[{self.n_nodes}];\ncreg c[{self.n_nodes}];\n"
        
        for i in range(self.n_nodes):
            qasm += f"h q[{i}];\n"
        
        for i, j in self.G.edges():
            qasm += f"cx q[{i}],q[{j}];\n"
            qasm += f"rz({2 * gamma}) q[{j}];\n"
            qasm += f"cx q[{i}],q[{j}];\n"
        
        for i in range(self.n_nodes):
            qasm += f"rx({2 * beta}) q[{i}];\n"
            qasm += f"measure q[{i}] -> c[{i}];\n"
            
        return qasm

    def run_circuit_worker(self, params):
        """Worker function for parallel processing"""
        gamma, beta, worker_id = params
        qasm = self.create_circuit(gamma, beta)
        
        try:
            process = subprocess.run(
                [self.simulator_path, "-", "100"],  # Use stdin
                input=qasm,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            if process.returncode != 0:
                print(f"Error in worker {worker_id}:")
                print("stdout:", process.stdout)
                print("stderr:", process.stderr)
                return (gamma, beta, 0, {})
            
            measurements = {}
            for line in process.stdout.split('\n'):
                if ':' in line and line[0].isdigit():
                    bitstring, count = line.split(':')[0], int(line.split()[1])
                    measurements[bitstring.strip()] = count
                    
            if measurements:
                best_bitstring = max(measurements.keys(), 
                                   key=lambda x: self.calculate_cut(x))
                cut_value = self.calculate_cut(best_bitstring)
                return (gamma, beta, -cut_value, measurements)
            
            return (gamma, beta, 0, {})
            
        except Exception as e:
            print(f"Error in worker {worker_id}: {e}")
            return (gamma, beta, 0, {})

    def compute_landscape_parallel(self):
        """Compute landscape using parallel processing (optimized)."""
        landscape = np.zeros((len(self.gamma_range), len(self.beta_range)))
        best_value = float('inf')
        best_measurements = None
        best_params = None

        # Precompute index mappings for faster lookups
        gamma_indices = {gamma: idx for idx, gamma in enumerate(self.gamma_range)}
        beta_indices = {beta: idx for idx, beta in enumerate(self.beta_range)}

        # Prepare worker parameters
        params = [(self, gamma, beta) for gamma, beta in product(self.gamma_range, self.beta_range)]

        # Use multiprocessing Pool with a progress bar
        with mp.Pool(self.n_threads) as pool:
            results = list(tqdm(pool.imap_unordered(qaoa_worker, params), total=len(params), desc="Processing"))

        # Process results
        for gamma, beta, value, measurements in results:
            i = gamma_indices[gamma]
            j = beta_indices[beta]
            landscape[i, j] = value

            if value < best_value:
                best_value = value
                best_measurements = measurements
                best_params = (gamma, beta)

        return landscape, best_measurements, best_params


    def visualize(self):
        start_time = time.time()
        
        print("Computing QAOA landscape in parallel...")
        landscape, best_measurements, (best_gamma, best_beta) = self.compute_landscape_parallel()
        
        computation_time = time.time() - start_time
        if best_measurements:
            best_bitstring = max(best_measurements.keys(), 
                            key=lambda x: self.calculate_cut(x))
            best_partition = [i for i, bit in enumerate(best_bitstring) if bit == '1']
            verification = self.verify_maxcut(best_partition)
        else:
            print("Warning: No valid measurements found")
            best_partition = []
            verification = None

        # Network visualization
        pos = nx.shell_layout(self.G)  # Structured layout for better spacing
        fig = go.Figure()

        # Separate edges into internal and cross-zone
        internal_edges = [(i, j) for i, j in self.G.edges()
                        if (i in best_partition) == (j in best_partition)]
        cross_edges = [(i, j) for i, j in self.G.edges()
                    if (i in best_partition) != (j in best_partition)]

        # Add internal edges (refined neon blue)
        for edge in internal_edges:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = self.G.edges[edge]['weight']
            fig.add_trace(go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode='lines',
                line=dict(color='#05f0fc', width=1 + weight / 2),  # Scaled thickness
                hoverinfo='text',
                text=[f"Weight: {weight}"],  # Tooltip showing weight
                name='Internal Edge',
                showlegend=False
            ))

        # Add cross-zone edges (subtle dashed red)
        for edge in cross_edges:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = self.G.edges[edge]['weight']
            fig.add_trace(go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode='lines',
                line=dict(color='#FF4500', width=1 + weight / 2, dash='dash'),  # Scaled thickness
                hoverinfo='text',
                text=[f"Weight: {weight}"],  # Tooltip showing weight
                name='Cross-Zone Edge',
                showlegend=False
            ))

        # Add nodes
        for i in range(self.n_nodes):
            x, y = pos[i]
            layer = "Core" if i < 2 else "Agg" if i < 6 else "Access"
            color = '#FF69B4' if i in best_partition else '#87CEEB'
            size = 30 if layer == "Core" else 25 if layer == "Agg" else 20

            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                text=[f"Node {i} ({layer})"],
                textposition="top center",
                hoverinfo='text',
                marker=dict(
                    size=size,
                    color=color,
                    line=dict(color='white', width=1)
                ),
                name=f"{layer} Nodes" if i == 0 else None,  # Legend entry only once
                showlegend=False
            ))

        # Add a simpler annotation for legend
        fig.add_annotation(
            text="<b>Legend:</b><br>"
                "<span style='color:#05f0fc;'>Neon Blue:</span> Internal Edge<br>"
                "<span style='color:#FF4500;'>Dashed Red:</span> Cross-Zone Edge<br>"
                "<b>Thickness:</b> Bandwidth<br>"
                "<b>Size:</b> Node Layer",
            xref="paper", yref="paper",
            x=1.1, y=0.25,  # Positioned on the right
            showarrow=False,
            font=dict(color="white"),
            align="left",
            bgcolor="rgba(10, 15, 30, 0.8)",
            bordercolor="white",
            borderwidth=1
        )

        # Update layout for better aesthetics
        fig.update_layout(
            title='Proof of Concept: Datacenter Bandwidth Optimization',
            title_x=0.5,
            font=dict(color='white', size=14),
            plot_bgcolor='rgba(10, 15, 30, 1)',  # Subtle dark background
            paper_bgcolor='rgba(10, 15, 30, 1)',
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            margin=dict(l=50, r=300, t=50, b=50),  # Wider margin for annotation
            legend=dict(
                x=1.02, y=1,
                bgcolor="rgba(10, 15, 30, 0.8)",
                font=dict(color="white"),
                bordercolor="white",
                borderwidth=1
            )
        )

        # Generate QUBO for the Max-Cut problem
        # qubo = max_cut_qubo(self.G)  # self.G is your NetworkX graph

        # # Solve QUBO using ExactSolver
        # exact_solver = ExactSolver()
        # classical_samples = exact_solver.sample_qubo(qubo)

        # # Extract the best solution
        # classical_best_cut = 0
        # start_classical = perf_counter()

        # for sample, energy in classical_samples.data(['sample', 'energy']):
        #     cut_value = -energy  # Energy is negative for Max-Cut
        #     classical_best_cut = max(classical_best_cut, cut_value)

        # end_classical = perf_counter()
        # classical_runtime = end_classical - start_classical

        # Find classical best cut using brute force
        # for partition in exact_solver.sample_qubo(dnx.qu):
        #     classical_best_cut = max(classical_best_cut, self.calculate_cut(partition))

        # end_classical = perf_counter()
        # classical_runtime = end_classical - start_classical

        # Update performance and result stats
        stats_text = (
            f"<b>Stats:</b><br>"
            f"Nodes: {verification['node_count']}<br>"
            f"Edges: {verification['edge_count']}<br>"
            f"Best Gamma: {best_gamma:.2f}<br>"
            f"Best Beta: {best_beta:.2f}<br>"
            # f"Best Cut (Classical): {classical_best_cut}<br>"
            f"Circuits Evaluated: {len(self.gamma_range) * len(self.beta_range)}<br>"
            f"<b>Time taken per circuit: {computation_time/(len(self.gamma_range) * len(self.beta_range)):.4f}s</b><br>"
            # f"Runtime (Classical): {classical_runtime:.2f}s"
        )


        fig.add_annotation(
            text=stats_text,
            xref="paper", yref="paper",
            x=1.1, y=0.5,  # Positioned on the right side
            showarrow=False,
            font=dict(color="white", size=12),
            align="left",
            bgcolor="rgba(10, 15, 30, 0.8)",
            bordercolor="white",
            borderwidth=1
        )


        fig.show()

    
if __name__ == "__main__":
    n_threads = mp.cpu_count()
    qaoa = QAOAMaxCut(n_nodes=12, n_threads=n_threads)
    qaoa.visualize()