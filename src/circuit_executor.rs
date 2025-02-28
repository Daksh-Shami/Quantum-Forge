use crate::*;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;
use rayon::{prelude::*, ThreadPoolBuilder};
use std::collections::HashMap;

// Thread-local storage for circuit instances
thread_local! {
    static CIRCUIT_CACHE: std::cell::RefCell<Option<Arc<QuantumCircuit>>> = std::cell::RefCell::new(None);
}

pub struct ExecutionResults {
    pub counts: HashMap<String, usize>,
    pub total_time: f64,
    pub peak_memory: usize,
    pub total_iterations: usize,
}

pub fn bitvec_to_string(bv: &MeasurementResult) -> String {
    let mut result = String::with_capacity(bv.len());
    for i in 0..bv.len() {
        result.push(if bv.get(i) { '1' } else { '0' });
    }
    result
}

pub fn execute_circuit(
    circuit: &QuantumCircuit, 
    measurement_order: &[usize], 
    iterations: usize,
    verbose: bool
) -> Result<ExecutionResults, String> {
    let start_time = Instant::now();

    if verbose {
        println!("Circuit composition:");
        println!("{}", circuit.compose());
        println!("Measurement order: {:?}", measurement_order);
    }

    // Configure thread pool
    let ideal_thread_num = circuit.qubit_count() * iterations / 100000;
    let num_threads = ideal_thread_num.clamp(1, num_cpus::get());
    
    ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .map_err(|e| format!("Failed to build thread pool: {}", e))?;

    if verbose {
        println!("Using {} threads", num_threads);
    }

    // Process in batches
    const BATCH_SIZE: usize = 100000;
    let num_batches = (iterations + BATCH_SIZE - 1) / BATCH_SIZE;
    
    // Process batches in parallel
    let results: Result<HashMap<MeasurementResult, usize>, String> = (0..num_batches).into_par_iter().try_fold(
        || HashMap::default(),
        |mut local_counts, batch| {
            let current_batch_size = if batch == num_batches - 1 {
                iterations - (batch * BATCH_SIZE)
            } else {
                BATCH_SIZE
            };

            // Execute measurements in parallel
            let results: Vec<MeasurementResult> = (0..current_batch_size)
                .into_par_iter()
                .map(|_| circuit.measureall(&measurement_order))
                .collect::<Result<Vec<_>, _>>()?;

            // Count results
            for result in results {
                *local_counts.entry(result).or_insert(0) += 1;
            }

            Ok(local_counts)
        },
    ).try_reduce(
        || HashMap::default(),
        |mut acc, local_counts| {
            for (state, count) in local_counts {
                *acc.entry(state).or_insert(0) += count;
            }
            Ok(acc)
        },
    );

    let counts = results?;

    // Convert MeasurementResult keys to string representation
    let mut string_counts = HashMap::default();
    for (state, count) in counts {
        let state_str = bitvec_to_string(&state);
        string_counts.insert(state_str, count);
    }

    if verbose {
        println!("\nMeasurement results (out of {} runs):", iterations);
        for (state, count) in &string_counts {
            println!("{}: {} ({:.2}%)", 
                    state, 
                    count, 
                    (*count as f64 / iterations as f64) * 100.0);
        }
    }

    let total_time = start_time.elapsed().as_secs_f64();
    let peak_memory = PEAK_MEMORY.load(Ordering::Relaxed);

    if verbose {
        println!("Total time: {:.6} seconds", total_time);
        println!("Peak memory usage: {:.2} MB", peak_memory as f64 / 1_048_576.0);
    }

    Ok(ExecutionResults {
        counts: string_counts,
        total_time,
        peak_memory,
        total_iterations: iterations,
    })
}