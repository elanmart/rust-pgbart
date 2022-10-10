#![allow(non_snake_case)]

extern crate pgbart;

use numpy::{PyReadonlyArray1, PyReadonlyArray2};

use pgbart::math::Matrix;
use pgbart::pgbart::ExternalData;

type LogpFunc = unsafe extern "C" fn(*const f32, usize) -> std::os::raw::c_float;

pub struct PythonData {
    X: Matrix<f32>,
    y: Vec<f32>,
    logp: LogpFunc,
}

impl PythonData {
    pub fn new(X: PyReadonlyArray2<f32>, y: PyReadonlyArray1<f32>, logp: usize) -> Self {
        let X = Matrix::from_vec(X.to_vec().unwrap(), X.shape()[0], X.shape()[1]);
        let y = y.to_vec().unwrap();

        let logp: LogpFunc = unsafe { std::mem::transmute(logp as *const std::ffi::c_void) };

        Self { X, y, logp }
    }
}

impl ExternalData for PythonData {
    fn X(&self) -> &Matrix<f32> {
        &self.X
    }

    fn y(&self) -> &Vec<f32> {
        &self.y
    }

    fn model_logp(&self, v: &Vec<f32>) -> f32 {
        let logp = self.logp;
        let value = unsafe { logp(v.as_ptr(), v.len()) };

        value
    }
}
