use super::QuantumSimulator;
use crate::{BitVec, MeasurementResult, QuantumGate, QuantumState};
use aligned_vec::AVec;
use num_complex::Complex;
use rand::Rng;
use std::arch::x86_64::*;

#[derive(Clone)]
pub struct CPUSimulator {
    state_vector: AVec<Complex<f64>>,
    num_qubits: usize,
}

impl CPUSimulator {
    pub fn new(num_qubits: usize) -> Self {
        let size = 1 << num_qubits;
        let mut state_vector = AVec::new(64); // 64-byte alignment for AVX-512
        state_vector.resize(size, Complex::new(0.0, 0.0));

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            // Initialize with SIMD - 4 complex numbers at a time (8 f64 values)
            let zero = _mm256_setzero_pd();
            let one = _mm256_set_pd(0.0, 0.0, 0.0, 1.0);

            // Set first element to 1.0 + 0.0i, rest to 0.0 + 0.0i
            let ptr = state_vector.as_mut_ptr() as *mut __m256d;
            _mm256_store_pd(ptr as *mut f64, one);

            // Zero out the rest using SIMD
            for i in 1..(size / 4) {
                _mm256_store_pd(ptr.add(i) as *mut f64, zero);
            }
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            state_vector[0] = Complex::new(1.0, 0.0);
        }

        Self {
            state_vector,
            num_qubits,
        }
    }

    pub fn from_state(state_vector: Box<[Complex<f64>]>, num_qubits: usize) -> Self {
        let mut aligned_vec = AVec::new(64);
        aligned_vec.resize(state_vector.len(), Complex::new(0.0, 0.0));

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            // Process 4 complex numbers (8 f64 values) at a time using AVX2
            let src_ptr = state_vector.as_ptr() as *const f64;
            let dst_ptr = aligned_vec.as_mut_ptr() as *mut f64;
            let num_doubles = state_vector.len() * 2; // Each Complex has 2 f64s

            // Copy aligned chunks using SIMD
            for i in (0..num_doubles).step_by(4) {
                if i + 4 <= num_doubles {
                    let v = _mm256_loadu_pd(src_ptr.add(i));
                    _mm256_store_pd(dst_ptr.add(i), v);
                }
            }

            // Handle remaining elements (if any)
            let remaining_start = (num_doubles / 4) * 4;
            for i in remaining_start..num_doubles {
                *dst_ptr.add(i) = *src_ptr.add(i);
            }
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            aligned_vec.extend_from_slice(&state_vector);
        }

        Self {
            state_vector: aligned_vec,
            num_qubits,
        }
    }

    fn apply_hadamard(&mut self, qubit: usize) -> Result<(), String> {
        if qubit >= self.num_qubits {
            return Err(format!("Qubit index {} out of range", qubit));
        }

        let factor = 1.0 / 2.0_f64.sqrt();
        let len = self.state_vector.len();
        let mask = 1 << qubit;

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            let factor_vec = _mm256_set1_pd(factor);
            let ptr = self.state_vector.as_mut_ptr() as *mut f64;

            // Process pairs while preserving phases
            for i in 0..len {
                if i & mask == 0 {
                    let i2 = i | mask;
                    if i2 < len {
                        // Load two complex numbers (real1,imag1,real2,imag2)
                        let v1 = _mm256_loadu_pd(ptr.add(2 * i));
                        let v2 = _mm256_loadu_pd(ptr.add(2 * i2));

                        // Compute sum and difference preserving phases
                        let sum = _mm256_add_pd(v1, v2);
                        let diff = _mm256_sub_pd(v1, v2);

                        // Apply normalization factor
                        let result1 = _mm256_mul_pd(sum, factor_vec);
                        let result2 = _mm256_mul_pd(diff, factor_vec);

                        // Store results back, handling only the real and imaginary parts
                        _mm_storeu_pd(ptr.add(2 * i), _mm256_castpd256_pd128(result1));
                        _mm_storeu_pd(ptr.add(2 * i2), _mm256_castpd256_pd128(result2));
                    }
                }
            }
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            for i in 0..len {
                if i & mask == 0 {
                    let i2 = i | mask;
                    if i2 < len {
                        let v1 = self.state_vector[i];
                        let v2 = self.state_vector[i2];

                        let sum = v1 + v2;
                        let diff = v1 - v2;

                        self.state_vector[i] = Complex::new(sum.re * factor, sum.im * factor);
                        self.state_vector[i2] = Complex::new(diff.re * factor, diff.im * factor);
                    }
                }
            }
        }

        Ok(())
    }

    fn apply_rx(&mut self, qubit: usize, angle: f64) -> Result<(), String> {
        if qubit >= self.num_qubits {
            return Err(format!("Qubit index {} out of range", qubit));
        }

        let theta = angle / 2.0;
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let len = self.state_vector.len();
        let mask = 1 << qubit;

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            let cos_vec = _mm256_set1_pd(cos_theta);
            let sin_vec = _mm256_set1_pd(sin_theta);
            let ptr = self.state_vector.as_mut_ptr() as *mut f64;

            for i in 0..len {
                if i & mask == 0 {
                    let i2 = i | mask;
                    if i2 < len {
                        // Load two complex numbers (real1,imag1,real2,imag2)
                        let v1 = _mm256_loadu_pd(ptr.add(2 * i)); // [re1, im1, re2, im2]
                        let v2 = _mm256_loadu_pd(ptr.add(2 * i2)); // [re3, im3, re4, im4]

                        // Prepare shuffled vectors for the matrix multiplication
                        let v1_re_im = v1; // [re1, im1, re2, im2]
                        let v1_im_re = _mm256_permute_pd(v1, 0b0101); // [im1, re1, im2, re2]
                        let v2_re_im = v2; // [re3, im3, re4, im4]
                        let v2_im_re = _mm256_permute_pd(v2, 0b0101); // [im3, re3, im4, re4]

                        // Compute the real parts: cos(θ)*v1 - sin(θ)*v2.im
                        let cos_terms = _mm256_mul_pd(cos_vec, v1_re_im);
                        let sin_terms = _mm256_mul_pd(sin_vec, v2_im_re);
                        let result1 = _mm256_sub_pd(cos_terms, sin_terms);

                        // Compute the imaginary parts: cos(θ)*v2 + sin(θ)*v1.im
                        let cos_terms2 = _mm256_mul_pd(cos_vec, v2_re_im);
                        let sin_terms2 = _mm256_mul_pd(sin_vec, v1_im_re);
                        let result2 = _mm256_add_pd(cos_terms2, sin_terms2);

                        // Store results back
                        _mm256_storeu_pd(ptr.add(2 * i), result1);
                        _mm256_storeu_pd(ptr.add(2 * i2), result2);
                    }
                }
            }
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            for i in 0..len {
                if i & mask == 0 {
                    let i2 = i | mask;
                    if i2 < len {
                        let v1 = self.state_vector[i];
                        let v2 = self.state_vector[i2];

                        // RX(θ) = [cos(θ/2)    -i*sin(θ/2)]
                        //         [-i*sin(θ/2)   cos(θ/2)  ]
                        self.state_vector[i] = Complex::new(
                            v1.re * cos_theta - v2.im * sin_theta,
                            v1.im * cos_theta + v2.re * sin_theta,
                        );
                        self.state_vector[i2] = Complex::new(
                            v2.re * cos_theta - v1.im * sin_theta,
                            v2.im * cos_theta + v1.re * sin_theta,
                        );
                    }
                }
            }
        }

        Ok(())
    }

    fn apply_rz(&mut self, qubit: usize, angle: f64) -> Result<(), String> {
        if qubit >= self.num_qubits {
            return Err(format!("Qubit index {} out of range", qubit));
        }

        let theta = angle / 2.0;
        let cos_theta = theta.cos();
        let sin_theta = -theta.sin(); // Note the negative sign for RZ gate
        let len = self.state_vector.len();
        let mask = 1 << qubit;

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            let cos_vec = _mm256_set1_pd(cos_theta);
            let sin_vec = _mm256_set1_pd(sin_theta);
            let ptr = self.state_vector.as_mut_ptr() as *mut f64;

            // Process 2 complex numbers at a time using AVX2
            for i in (0..len).step_by(2) {
                if i & mask != 0 {
                    // Load two complex numbers (real1,imag1,real2,imag2)
                    let v = _mm256_loadu_pd(ptr.add(2 * i));

                    // Split into real and imaginary parts
                    let re = v; // [re1, im1, re2, im2]
                    let im = _mm256_permute_pd(v, 0b0101); // [im1, re1, im2, re2]

                    // Compute rotation: (cos(θ) + i*sin(θ))*(re + i*im)
                    // = (cos(θ)*re - sin(θ)*im) + i*(sin(θ)*re + cos(θ)*im)
                    let cos_terms = _mm256_mul_pd(cos_vec, re);
                    let sin_terms = _mm256_mul_pd(sin_vec, im);
                    let real_result = _mm256_sub_pd(cos_terms, sin_terms);

                    let cos_terms_im = _mm256_mul_pd(cos_vec, im);
                    let sin_terms_re = _mm256_mul_pd(sin_vec, re);
                    let imag_result = _mm256_add_pd(sin_terms_re, cos_terms_im);

                    // Combine real and imaginary parts
                    let result = _mm256_blend_pd(real_result, imag_result, 0b1010);

                    // Store the result back
                    _mm256_storeu_pd(ptr.add(2 * i), result);
                }
            }
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            let rotation = Complex::new(cos_theta, sin_theta);
            for i in 0..len {
                if i & mask != 0 {
                    self.state_vector[i] *= rotation;
                }
            }
        }

        Ok(())
    }

    fn apply_cnot(&mut self, control: usize, target: usize) -> Result<(), String> {
        if control >= self.num_qubits || target >= self.num_qubits {
            return Err(format!("Qubit indices out of range"));
        }

        let len = self.state_vector.len();
        let control_mask = 1 << control;
        let target_mask = 1 << target;

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        // Process 2 complex numbers at a time (4 f64s)
        for i in (0..len).step_by(2) {
            // Check first complex number
            if i & control_mask != 0 && i & target_mask == 0 {
                let i2 = i ^ target_mask;
                self.state_vector.swap(i, i2);
            }

            // Check second complex number if it exists
            if i + 1 < len {
                let i_next = i + 1;
                if i_next & control_mask != 0 && i_next & target_mask == 0 {
                    let i2_next = i_next ^ target_mask;
                    self.state_vector.swap(i_next, i2_next);
                }
            }
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            for i in 0..len {
                if (i & control_mask != 0) && (i & target_mask == 0) {
                    let i2 = i ^ target_mask;
                    self.state_vector.swap(i, i2);
                }
            }
        }

        Ok(())
    }

    fn apply_x(&mut self, qubit: usize) -> Result<(), String> {
        if qubit >= self.num_qubits {
            return Err(format!("Qubit index {} out of range", qubit));
        }

        for i in 0..self.state_vector.len() {
            if i & (1 << qubit) == 0 {
                let i2 = i ^ (1 << qubit);
                self.state_vector.swap(i, i2);
            }
        }
        Ok(())
    }

    fn apply_cz(&mut self, control: usize, target: usize) -> Result<(), String> {
        if control >= self.num_qubits || target >= self.num_qubits {
            return Err(format!("Qubit indices out of range"));
        }

        for i in 0..self.state_vector.len() {
            if (i & (1 << control) != 0) && (i & (1 << target) != 0) {
                self.state_vector[i] = -self.state_vector[i];
            }
        }
        Ok(())
    }

    fn apply_toffoli(
        &mut self,
        control1: usize,
        control2: usize,
        target: usize,
    ) -> Result<(), String> {
        if control1 >= self.num_qubits || control2 >= self.num_qubits || target >= self.num_qubits {
            return Err(format!("Qubit indices out of range"));
        }

        for i in 0..self.state_vector.len() {
            if (i & (1 << control1) != 0) && (i & (1 << control2) != 0) {
                let i2 = i ^ (1 << target);
                self.state_vector.swap(i, i2);
            }
        }
        Ok(())
    }

    fn apply_swap(&mut self, qubit1: usize, qubit2: usize) -> Result<(), String> {
        if qubit1 >= self.num_qubits || qubit2 >= self.num_qubits {
            return Err(format!("Qubit indices out of range"));
        }

        for i in 0..self.state_vector.len() {
            if (i & (1 << qubit1) != 0) != (i & (1 << qubit2) != 0) {
                let j = i ^ (1 << qubit1) ^ (1 << qubit2);
                self.state_vector.swap(i, j);
            }
        }
        Ok(())
    }

    fn apply_phase(&mut self, qubit: usize) -> Result<(), String> {
        if qubit >= self.num_qubits {
            return Err(format!("Qubit index {} out of range", qubit));
        }

        let phase = Complex::new(0.0, 1.0);
        let mask = 1 << qubit;

        // Only iterate over states where the qubit is 1
        for i in 0..self.state_vector.len() {
            if i & mask != 0 {
                self.state_vector[i] *= phase;
            }
        }

        Ok(())
    }
}

impl QuantumSimulator for CPUSimulator {
    fn apply_gate(&mut self, gate: &QuantumGate) -> Result<(), String> {
        match gate {
            QuantumGate::Hadamard(qubit) => self.apply_hadamard(*qubit),
            QuantumGate::CNOT(control, target) => self.apply_cnot(*control, *target),
            QuantumGate::X(qubit) => self.apply_x(*qubit),
            QuantumGate::Phase(qubit) => self.apply_phase(*qubit),
            QuantumGate::RZ(qubit, angle) => self.apply_rz(*qubit, *angle),
            QuantumGate::RX(qubit, angle) => self.apply_rx(*qubit, *angle),
            QuantumGate::CZ(control, target) => self.apply_cz(*control, *target),
            QuantumGate::Toffoli(c1, c2, target) => self.apply_toffoli(*c1, *c2, *target),
            QuantumGate::Swap(q1, q2) => self.apply_swap(*q1, *q2),
        }
    }

    fn measure(&self, measurement_order: &[usize]) -> Result<MeasurementResult, String> {
        // Validate measurement order
        if let Some(&invalid_qubit) = measurement_order.iter().find(|&&q| q >= self.num_qubits) {
            return Err(format!(
                "Invalid qubit {} in measurement order",
                invalid_qubit
            ));
        }

        let mut rng = rand::thread_rng();
        let mut cumulative_prob = 0.0;
        let r = rng.gen::<f64>();

        // Create result bitvec with zeros for all qubits
        let mut result = BitVec::zeros(self.num_qubits);

        // Find the collapsed state using the same efficient sampling
        for (i, amplitude) in self.state_vector.iter().enumerate() {
            cumulative_prob += amplitude.norm_sqr();
            if r <= cumulative_prob {
                // Only set the bits for qubits in the measurement order
                for &qubit_idx in measurement_order {
                    result.set(qubit_idx, (i >> qubit_idx) & 1 == 1);
                }
                return Ok(MeasurementResult::new(result));
            }
        }

        Err("Measurement failed - probabilities did not sum to 1".to_string())
    }

    fn get_state(&self) -> Result<QuantumState, String> {
        Ok(QuantumState::from_amplitudes_ref(&self.state_vector))
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn memory_estimate(&self) -> usize {
        // Each complex number uses 16 bytes (2 * f64)
        self.state_vector.len() * 16
    }
}
