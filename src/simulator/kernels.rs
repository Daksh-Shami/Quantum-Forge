use cuda_std::prelude::*;
use cuda_std::float::GpuFloat;
use crate::Complex;

mod ptx {
    include!(concat!(env!("OUT_DIR"), "/quantum_forge_kernels.ptx.rs"));
}
pub use ptx::PTX;

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn hadamard_kernel(state_vector: *mut Complex<GpuFloat>, qubit: u32, n: u32) {
    let idx = thread::index_1d() as usize;
    if idx >= n as usize {
        return;
    }

    // Only process if this is the lower index of a pair
    if (idx & (1 << qubit)) == 0 {
        let pair_idx = idx | (1 << qubit);
        if pair_idx < n as usize {
            let v0 = *state_vector.add(idx);
            let v1 = *state_vector.add(pair_idx);
            
            // Apply Hadamard transformation
            let inv_sqrt2 = GpuFloat::from(1.0) / GpuFloat::from(2.0).sqrt();
            *state_vector.add(idx) = Complex::new(inv_sqrt2, GpuFloat::from(0.0)) * (v0 + v1);
            *state_vector.add(pair_idx) = Complex::new(inv_sqrt2, GpuFloat::from(0.0)) * (v0 - v1);
        }
    }
}

#[kernel]
pub unsafe fn rx_kernel(state_vector: *mut Complex<GpuFloat>, qubit: u32, angle: GpuFloat, n: u32) {
    let idx = thread::index_1d() as usize;
    if idx >= n as usize {
        return;
    }

    // Only process if this is the lower index of a pair
    if (idx & (1 << qubit)) == 0 {
        let pair_idx = idx | (1 << qubit);
        if pair_idx < n as usize {
            let v0 = *state_vector.add(idx);
            let v1 = *state_vector.add(pair_idx);
            
            let cos = (angle / GpuFloat::from(2.0)).cos();
            let sin = (angle / GpuFloat::from(2.0)).sin();
            
            *state_vector.add(idx) = Complex::new(cos, GpuFloat::from(0.0)) * v0 + 
                                   Complex::new(GpuFloat::from(0.0), -sin) * v1;
            *state_vector.add(pair_idx) = Complex::new(GpuFloat::from(0.0), -sin) * v0 + 
                                        Complex::new(cos, GpuFloat::from(0.0)) * v1;
        }
    }
}

#[kernel]
pub unsafe fn rz_kernel(state_vector: *mut Complex<GpuFloat>, qubit: u32, angle: GpuFloat, n: u32) {
    let idx = thread::index_1d() as usize;
    if idx >= n as usize {
        return;
    }

    if (idx & (1 << qubit)) != 0 {
        let v = *state_vector.add(idx);
        let phase = Complex::new(angle.cos(), angle.sin());
        *state_vector.add(idx) = v * phase;
    }
}

#[kernel]
pub unsafe fn cnot_kernel(state_vector: *mut Complex<GpuFloat>, control: u32, target: u32, n: u32) {
    let idx = thread::index_1d() as usize;
    if idx >= n as usize {
        return;
    }

    if (idx & (1 << control)) != 0 {
        let target_bit = idx & (1 << target) != 0;
        let partner_idx = idx ^ (1 << target);
        
        if !target_bit {
            let temp = *state_vector.add(idx);
            *state_vector.add(idx) = *state_vector.add(partner_idx);
            *state_vector.add(partner_idx) = temp;
        }
    }
}
