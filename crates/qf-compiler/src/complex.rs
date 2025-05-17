use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug)]
pub struct Complex<T = f64>
where
    T: Copy + Debug,
{
    pub re: T,
    pub im: T,
}

// Implement Serialize and Deserialize only for f64 which we know supports them
impl Serialize for Complex<f64> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Complex", 2)?;
        state.serialize_field("re", &self.re)?;
        state.serialize_field("im", &self.im)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for Complex<f64> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct ComplexHelper {
            re: f64,
            im: f64,
        }

        let helper = ComplexHelper::deserialize(deserializer)?;
        Ok(Self {
            re: helper.re,
            im: helper.im,
        })
    }
}

impl<T> Complex<T>
where
    T: Copy + Debug,
{
    pub const fn new(re: T, im: T) -> Self {
        Self { re, im }
    }
}

// It's more efficient to call norm_squared than norm. This is by design as Quantum operations usually use norm squared more often.
impl<T> Complex<T>
where
    T: Copy + Debug + Add<Output = T> + Mul<Output = T>,
{
    pub fn norm_squared(&self) -> T {
        self.re * self.re + self.im * self.im
    }
}

// Implement norm only for f64 since it requires sqrt
impl Complex<f64> {
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }
}

impl<T> Mul for Complex<T>
where
    T: Copy + Debug + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl<T> Mul<Complex<T>> for &Complex<T>
where
    T: Copy + Debug + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    type Output = Complex<T>;

    fn mul(self, rhs: Complex<T>) -> Complex<T> {
        Complex {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl<T> MulAssign for Complex<T>
where
    T: Copy + Debug + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    fn mul_assign(&mut self, rhs: Self) {
        let re = self.re * rhs.re - self.im * rhs.im;
        let im = self.re * rhs.im + self.im * rhs.re;
        self.re = re;
        self.im = im;
    }
}

impl<T> Add for Complex<T>
where
    T: Copy + Debug + Add<Output = T>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl<T> AddAssign for Complex<T>
where
    T: Copy + Debug + AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl<T> Sub for Complex<T>
where
    T: Copy + Debug + Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl<T> Neg for Complex<T>
where
    T: Copy + Debug + Neg<Output = T>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl<T> SubAssign for Complex<T>
where
    T: Copy + Debug + SubAssign,
{
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}
