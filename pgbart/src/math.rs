pub fn elemwise(v1: &Vec<f64>, v2: &Vec<f64>, f: fn((&f64, &f64)) -> f64) -> Vec<f64> {
    std::iter::zip(v1, v2).map(f).collect()
}

pub fn add(v1: &Vec<f64>, v2: &Vec<f64>) -> Vec<f64> {
    elemwise(v1, v2, |(x1, x2)| x1 + x2)
}

pub fn sub(v1: &Vec<f64>, v2: &Vec<f64>) -> Vec<f64> {
    elemwise(v1, v2, |(x1, x2)| x1 - x2)
}

pub fn max(v: &Vec<f64>) -> f64 {
    v.iter().fold(f64::MIN, |a, &b| a.max(b))
}

pub fn mean(v: &Vec<f64>) -> f64 {
    let sum: f64 = v.iter().sum();
    sum / v.len() as f64
}

pub fn cumsum(v: &Vec<f64>) -> Vec<f64> {
    let ret: Vec<f64> = v
        .iter()
        .scan(0f64, |state, item| {
            *state += *item;
            let ret = state;
            Some(*ret)
        })
        .collect();

    ret
}

pub fn normalized_cumsum(v: &Vec<f64>) -> Vec<f64> {
    let total: f64 = v.iter().sum();
    let ret: Vec<f64> = v
        .iter()
        .scan(0f64, |state, item| {
            *state += *item;
            let ret = *state / total;
            Some(ret)
        })
        .collect();

    ret
}

pub fn stdev(v: &Vec<f64>) -> f64 {
    let n = v.len() as f64;
    let mu = mean(v);

    let f = |carry: f64, item: &f64| carry + (item - mu).powi(2);
    let var = v.iter().fold(0., f) / n;

    var.sqrt()
}

pub struct Matrix<T>
where
    T: Copy,
{
    data: Vec<T>,
    pub n_rows: usize,
    pub n_cols: usize,
}

impl<T> Matrix<T>
where
    T: Copy,
{
    pub fn from_vec(data: Vec<T>, n_rows: usize, n_cols: usize) -> Self {
        if data.len() != (n_rows * n_cols) {
            panic!("Data size does not match the provided dimensions");
        }

        Matrix {
            data,
            n_rows,
            n_cols,
        }
    }

    pub fn get(&self, i: &usize, j: &usize) -> T {
        self.data[i * self.n_cols + j]
    }

    pub fn select_rows(&self, rows: &Vec<usize>, col: &usize) -> Vec<T> {
        let mut ret = Vec::<T>::with_capacity(rows.len());

        for i in rows {
            ret.push(self.get(i, col));
        }

        ret
    }
}
