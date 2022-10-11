#![allow(non_snake_case)]

extern crate pgbart;

use numpy::{PyReadonlyArray1, PyReadonlyArray2};

use pgbart::math::Matrix;
use pgbart::pgbart::ExternalData;

type LogpFunc = unsafe extern "C" fn(*const f64, usize) -> std::os::raw::c_double;

pub struct PythonData {
    X: Matrix<f64>,
    y: Vec<f64>,
    logp: LogpFunc,
}

impl PythonData {
    pub fn new(X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>, logp: usize) -> Self {
        let X = Matrix::from_vec(X.to_vec().unwrap(), X.shape()[0], X.shape()[1]);
        let y = y.to_vec().unwrap();

        let logp: LogpFunc = unsafe { std::mem::transmute(logp as *const std::ffi::c_void) };

        Self { X, y, logp }
    }
}

impl ExternalData for PythonData {
    fn X(&self) -> &Matrix<f64> {
        &self.X
    }

    fn y(&self) -> &Vec<f64> {
        &self.y
    }

    fn model_logp(&self, v: &Vec<f64>) -> f64 {
        let logp = self.logp;
        let value = unsafe { logp(v.as_ptr(), v.len()) };

        value
    }
}
