use super::QuantumSimulator;
use crate::{Complex, QuantumGate, MeasurementResult, QuantumState, BitVec};
use cust::prelude::*;
use cuda_std::prelude::*;
use cuda_std::float::GpuFloat;
use rand::Rng;
use std::error::Error;

use super::kernels;

const BLOCK_SIZE: u32 = 256;

#[derive(Clone)]
pub struct GPUSimulator {
    state_vector: DeviceBuffer<Complex<f64>>,
    module: Module,
    stream: Stream,
    context: Context,
    device: Device,
    num_qubits: usize,
}

impl GPUSimulator {
    pub fn new(num_qubits: usize) -> Result<Self, Box<dyn Error>> {
        let size = 1 << num_qubits;
        
        // Initialize CUDA
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(0)?;
        let context = Context::new(device)?;
        let stream = Stream::new(StreamFlags::DEFAULT, None)?;
        
        // Load kernels module
        let module = Module::from_ptx(kernels::PTX, &[])?;
        
        // Initialize state vector on host
        let mut host_state = vec![Complex::new(0.0, 0.0); size];
        host_state[0] = Complex::new(1.0, 0.0);  // Initialize |0> state
        
        // Allocate and copy to device
        let state_vector = DeviceBuffer::from_slice(&host_state)?;

        Ok(Self {
            state_vector,
            module,
            stream,
            context,
            device,
            num_qubits,
        })
    }

    fn apply_hadamard(&mut self, qubit: usize) -> Result<(), Box<dyn Error>> {
        if qubit >= self.num_qubits {
            return Err("Qubit index out of range".into());
        }

        let n = 1 << self.num_qubits;
        let num_blocks = (n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

        let func = self.module.get_function("hadamard_kernel")?;
        unsafe {
            let stream = &self.stream;
            launch!(func<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                self.state_vector.as_device_ptr(),
                qubit as u32,
                n as u32
            ))?;
        }

        Ok(())
    }

    fn apply_rx(&mut self, qubit: usize, angle: f64) -> Result<(), Box<dyn Error>> {
        if qubit >= self.num_qubits {
            return Err("Qubit index out of range".into());
        }

        let n = 1 << self.num_qubits;
        let num_blocks = (n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

        let func = self.module.get_function("rx_kernel")?;
        unsafe {
            let stream = &self.stream;
            launch!(func<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                self.state_vector.as_device_ptr(),
                qubit as u32,
                angle,
                n as u32
            ))?;
        }

        Ok(())
    }

    fn apply_rz(&mut self, qubit: usize, angle: f64) -> Result<(), Box<dyn Error>> {
        if qubit >= self.num_qubits {
            return Err("Qubit index out of range".into());
        }

        let n = 1 << self.num_qubits;
        let num_blocks = (n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

        let func = self.module.get_function("rz_kernel")?;
        unsafe {
            let stream = &self.stream;
            launch!(func<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                self.state_vector.as_device_ptr(),
                qubit as u32,
                angle,
                n as u32
            ))?;
        }

        Ok(())
    }

    fn apply_cnot(&mut self, control: usize, target: usize) -> Result<(), Box<dyn Error>> {
        if control >= self.num_qubits || target >= self.num_qubits {
            return Err("Qubit indices out of range".into());
        }

        let n = 1 << self.num_qubits;
        let num_blocks = (n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

        let func = self.module.get_function("cnot_kernel")?;
        unsafe {
            let stream = &self.stream;
            launch!(func<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                self.state_vector.as_device_ptr(),
                control as u32,
                target as u32,
                n as u32
            ))?;
        }

        Ok(())
    }
}

impl QuantumSimulator for GPUSimulator {
    fn apply_gate(&mut self, gate: &QuantumGate) -> Result<(), String> {
        match gate {
            QuantumGate::Hadamard { qubit } => self.apply_hadamard(*qubit)
                .map_err(|e| format!("Hadamard gate error: {}", e)),
            QuantumGate::RX { qubit, angle } => self.apply_rx(*qubit, *angle)
                .map_err(|e| format!("RX gate error: {}", e)),
            QuantumGate::RZ { qubit, angle } => self.apply_rz(*qubit, *angle)
                .map_err(|e| format!("RZ gate error: {}", e)),
            QuantumGate::CNOT { control, target } => self.apply_cnot(*control, *target)
                .map_err(|e| format!("CNOT gate error: {}", e)),
            _ => Err(format!("Gate {:?} not yet implemented for GPU", gate)),
        }
    }

    fn measure(&self, measurement_order: &[usize]) -> Result<MeasurementResult, String> {
        let n = 1 << self.num_qubits;
        let mut host_state = vec![Complex::new(0.0, 0.0); n];
        
        // Copy state vector back to host for measurement
        self.state_vector.copy_to(&mut host_state)
            .map_err(|e| format!("Failed to copy state vector from GPU: {}", e))?;

        let mut rng = rand::thread_rng();
        let mut result = BitVec::new();
        let mut remaining_probability = 1.0;
        
        for &qubit in measurement_order {
            let mut prob_one = 0.0;
            let mask = 1 << qubit;
            
            for (i, &amp) in host_state.iter().enumerate() {
                if i & mask != 0 {
                    prob_one += (amp.re * amp.re + amp.im * amp.im) / remaining_probability;
                }
            }
            
            let random = rng.gen::<f64>();
            let outcome = random < prob_one;
            result.push(outcome);
            
            // Update remaining probability and collapse state
            let mut new_norm = 0.0;
            for i in 0..host_state.len() {
                if (i & mask != 0) != outcome {
                    host_state[i] = Complex::new(0.0, 0.0);
                } else {
                    new_norm += host_state[i].re * host_state[i].re + 
                               host_state[i].im * host_state[i].im;
                }
            }
            
            let scale = (1.0 / new_norm).sqrt();
            for amp in host_state.iter_mut() {
                *amp = Complex::new(amp.re * scale, amp.im * scale);
            }
            
            remaining_probability = new_norm;
        }
        
        // Update GPU state vector with collapsed state
        self.state_vector.copy_from(&host_state)
            .map_err(|e| format!("Failed to copy collapsed state back to GPU: {}", e))?;
            
        Ok(MeasurementResult::new(result))
    }

    fn get_state(&self) -> Result<QuantumState, String> {
        let n = 1 << self.num_qubits;
        let mut host_state = vec![Complex::new(0.0, 0.0); n];
        
        self.state_vector.copy_to(&mut host_state)
            .map_err(|e| format!("Failed to copy state vector from GPU: {}", e))?;
            
        // Convert Complex to your Complex type
        let state = host_state.into_iter()
            .map(|c| Complex::new(c.re, c.im))
            .collect();
            
        Ok(QuantumState::from_vec(state))
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn memory_estimate(&self) -> usize {
        // Each complex number uses 16 bytes (2 * f64)
        16 * (1 << self.num_qubits)
    }
}
