#![allow(non_snake_case)]

mod data;

extern crate pgbart;

use crate::data::PythonData;

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pgbart::pgbart::{PgBartSettings, PgBartState};
use pyo3::prelude::*;

#[pyclass(unsendable)]
struct StateWrapper {
    state: PgBartState,
}

#[pyfunction]
fn initialize(
    X: PyReadonlyArray2<f32>,
    y: PyReadonlyArray1<f32>,
    logp: usize,
    alpha: f32,
    n_trees: usize,
    n_particles: usize,
    kfactor: f32,
    batch: (f32, f32),
    split_covar_prior: PyReadonlyArray1<f32>,
) -> StateWrapper {
    let data = PythonData::new(X, y, logp);
    let data = Box::new(data);
    let params = PgBartSettings::new(
        n_trees,
        n_particles,
        alpha,
        kfactor,
        batch,
        split_covar_prior.to_vec().unwrap(),
    );

    let state = PgBartState::new(params, data);

    StateWrapper { state }
}

#[pyfunction]
fn step<'py>(py: Python<'py>, wrapper: &mut StateWrapper, tune: bool) -> &'py PyArray1<f32> {
    wrapper.state.set_tune(tune);
    wrapper.state.step();
    
    // Can we avoid using clone() here somehow?
    let predictions = wrapper.state.predictions();
    let arr = PyArray1::from_vec(py, predictions.clone());

    arr
}

#[pymodule]
fn rust_pgbart(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(initialize, m)?)?;
    m.add_function(wrap_pyfunction!(step, m)?)?;
    Ok(())
}
