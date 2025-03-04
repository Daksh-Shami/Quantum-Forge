use qf_compiler::*;
use std::env;
use std::fs::File;
use std::io::{self, BufReader, Read};

mod circuit_executor;
use circuit_executor::execute_circuit;

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        return Err(
            "Usage: quantum_compiler <input_file or '-' for stdin> <iterations>".to_string(),
        );
    }

    // Pre-allocate buffer with capacity for better performance
    let mut buffer = String::with_capacity(16 * 1024); // 16KB initial capacity

    // Read input either from file or stdin using buffered I/O
    let input = if args[1] == "-" {
        BufReader::new(io::stdin())
            .read_to_string(&mut buffer)
            .map_err(|e| format!("Error reading from stdin: {}", e))?;
        buffer
    } else {
        let file = File::open(&args[1]).map_err(|e| format!("Error opening file: {}", e))?;
        BufReader::new(file)
            .read_to_string(&mut buffer)
            .map_err(|e| format!("Error reading file: {}", e))?;
        buffer
    };

    let (circuit, measurement_order) = QuantumCircuit::from_qasm(&input)?;

    let total_iterations: usize = args[2]
        .parse()
        .map_err(|e| format!("Invalid number of iterations: {}", e))?;

    // Execute circuit with verbose output disabled for better performance
    let results = execute_circuit(&circuit, &measurement_order, total_iterations, false)?;

    // Use a single formatted string to reduce allocations
    println!("\nExecution Summary:\nTotal time: {:.6} seconds\nPeak memory usage: {:.2} MB\nTotal iterations: {}\n\nMeasurement counts:", 
        results.total_time,
        results.peak_memory as f64 / 1_048_576.0,
        results.total_iterations
    );

    // Pre-calculate the divisor for percentage
    let percentage_divisor = total_iterations as f64;
    for (state, count) in results.counts {
        println!(
            "{}: {} ({:.2}%)",
            state,
            count,
            (count as f64 / percentage_divisor) * 100.0
        );
    }

    Ok(())
}
