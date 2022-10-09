#![allow(non_snake_case)]

extern crate pgbart;

use numpy::{PyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyFunction;

use pgbart::math::Matrix;
use pgbart::pgbart::ExternalData;

pub struct PythonData {
    X: Matrix<f32>,
    y: Vec<f32>,
    logp: Py<PyFunction>,
}

impl PythonData {
    pub fn new(X: PyReadonlyArray2<f32>, y: PyReadonlyArray1<f32>, logp: Py<PyFunction>) -> Self {
        let X = Matrix::from_vec(X.to_vec().unwrap(), X.shape()[0], X.shape()[1]);
        let y = y.to_vec().unwrap();

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
        let value = Python::with_gil(|py| {
            let pyarray = PyArray::from_slice(py, v.as_slice());
            let args = (pyarray,);
            let res = self
                .logp
                .call1(py, args)
                .expect("logp did not return a value!!!");
            let val: f32 = res.extract(py).unwrap();
            val
        });

        value
    }
}
