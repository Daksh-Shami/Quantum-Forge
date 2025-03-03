use crate::complex::Complex;

pub const FRAC_1_SQRT_2: f64 = 1.0 / 2.0_f64.sqrt(); // 1/√2

pub const H_FACTOR: Complex = Complex {
    re: FRAC_1_SQRT_2,
    im: 0.0,
};

pub const I_UNIT: Complex = Complex { re: 0.0, im: 1.0 };

// Helper function to create RX gate matrix elements
pub fn rx_matrix_elements(angle: f64) -> (Complex, Complex) {
    let cos_half = (angle/2.0).cos();
    let sin_half = (angle/2.0).sin();
    (
        Complex::new(cos_half, 0.0),
        Complex::new(0.0, -sin_half)
    )
}
