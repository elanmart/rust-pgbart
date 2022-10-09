#![allow(non_snake_case)]

use crate::math::{self, Matrix};
use crate::pgbart::PgBartState;
use crate::tree::Tree;

use std::collections::VecDeque;
use std::collections::{HashMap, HashSet};

#[derive(Clone)]
pub struct ParticleParams {
    n_points: usize, // Number of points in the dataset
    n_covars: usize, // Number of covariates in the dataset
    kfactor: f32,    // Standard deviation of noise added during leaf value sampling
}

#[derive(Clone)]
struct Indices {
    leaf_nodes: HashSet<usize>,               // Set of leaf node indices
    expansion_nodes: VecDeque<usize>,         // Nodes that we still can expand
    data_indices: HashMap<usize, Vec<usize>>, // Indicies of points at each node
}

#[derive(Clone)]
pub struct Weight {
    log_w: f32,          // Log weight of this particle
    log_likelihood: f32, // Log-likelihood from the previous iteration
}

#[derive(Clone)]
pub struct Particle {
    params: ParticleParams,
    tree: Tree,
    indices: Indices,
    weight: Weight,
}

impl Weight {
    fn new() -> Self {
        Weight {
            log_w: 0.,
            log_likelihood: 0.,
        }
    }

    // Sets the log-weight and log-likelihood of this particle to a fixed value
    pub fn reset(&mut self, log_likelihood: f32) {
        self.log_w = log_likelihood;
        self.log_likelihood = log_likelihood;
    }

    // Updates the log-weight of this particle, and sets the log-likelohood to a new value
    pub fn update(&mut self, log_likelihood: f32) {
        let log_w = self.log_w + log_likelihood - self.log_likelihood;

        self.log_w = log_w;
        self.log_likelihood = log_likelihood;
    }

    // --- Getters ---
    pub fn log_w(&self) -> f32 {
        self.log_w
    }

    pub fn log_likelihood(&self) -> f32 {
        self.log_likelihood
    }

    // --- Setters ---
    pub fn set_log_w(&mut self, log_w: f32) {
        self.log_w = log_w;
    }
}

impl ParticleParams {
    pub fn new(n_points: usize, n_covars: usize, kfactor: f32) -> Self {
        ParticleParams {
            n_points,
            n_covars,
            kfactor,
        }
    }

    pub fn with_new_kf(&self, kfactor: f32) -> Self {
        ParticleParams {
            n_points: self.n_points,
            n_covars: self.n_covars,
            kfactor,
        }
    }
}

impl Indices {
    // Creates a new struct for a given dataset size
    fn new(n_points: usize) -> Self {
        let data_indices = Vec::from_iter(0..n_points);
        Indices {
            leaf_nodes: HashSet::from([0]),
            expansion_nodes: VecDeque::from([0]),
            data_indices: HashMap::from([(0, data_indices)]),
        }
    }

    // Checks if there are any nodes left to expand
    fn is_empty(&self) -> bool {
        self.expansion_nodes.is_empty()
    }

    // Returns the indices of datapoints stored in node with index `idx`
    fn get_data_indices(&self, idx: usize) -> Result<&Vec<usize>, &str> {
        let ret = self.data_indices.get(&idx);
        ret.ok_or("Index not found in the data_indices map")
    }

    // Removes an index from the list of expansion nodes
    fn pop_expansion_index(&mut self) -> Option<usize> {
        self.expansion_nodes.pop_front()
    }

    // Removes the index from the set of leaves and from the data_indices map
    fn remove_index(&mut self, idx: usize) {
        self.leaf_nodes.remove(&idx);
        self.data_indices.remove(&idx);
    }

    // Adds an index of a new leaf to be expanded
    fn add_index(&mut self, idx: usize, data_rows: Vec<usize>) {
        self.leaf_nodes.insert(idx);
        self.expansion_nodes.push_back(idx);
        self.data_indices.insert(idx, data_rows);
    }

    // Removes everything from the expansion nodes
    fn clear(&mut self) {
        self.expansion_nodes.clear();
    }
}

impl Particle {
    // Creates a new Particle with specified Params and a single-node (root only) Tree
    pub fn new(params: ParticleParams, leaf_value: f32) -> Self {
        let tree = Tree::new(leaf_value);
        let indices = Indices::new(params.n_points);
        let weight = Weight::new();

        Particle {
            params,
            tree,
            indices,
            weight,
        }
    }

    pub fn frozen_copy(&self) -> Particle {
        let mut ret = self.clone();
        ret.indices.clear();

        ret
    }

    // Creates a copy of the particle, but re-samples the values in leaf nodes
    pub fn with_resampled_leaves(&self, state: &PgBartState) -> Self {
        // Make a copy of the tree structure, params etc.
        let mut ret = self.frozen_copy();

        for leaf_idx in &ret.indices.leaf_nodes {
            // We don't resample the root
            if *leaf_idx == 0 {
                continue;
            }

            // Ask the PG to generate a new value
            let data_indices = &ret.indices.data_indices[leaf_idx];
            let value = ret.leaf_value(data_indices, state);

            // Update the tree
            let msg =
                "The indices stored in self.indices are not consistent with the tree structure";
            ret.tree.update_leaf_node(*leaf_idx, value).expect(msg);
        }

        ret
    }

    // Attempts to grow this particle (or, more precisely, the tree inside this particle)
    // Returns a boolean indicating if the tree structure was modified
    pub fn grow(&mut self, X: &Matrix<f32>, state: &PgBartState) -> bool {
        // Check if there are any nodes left to expand
        let idx = match self.indices.pop_expansion_index() {
            Some(value) => value,
            None => {
                return false;
            }
        };

        // Stochastiaclly decide if the node should be split or not
        let msg = "Internal indices are not aligned with the tree";
        let leaf = self.tree.get_node(&idx).expect(msg);
        let expand = state.probabilities().sample_expand_flag(leaf.depth());
        if !expand {
            return false;
        }

        // Get the examples that were routed to this leaf
        let rows = self.indices.get_data_indices(idx).expect(msg);
        let split_idx = state.probabilities().sample_split_index();
        let feature_values = X.select_rows(rows, &split_idx);

        // And see if we can split them into two groups
        match state.probabilities().sample_split_value(&feature_values) {
            None => {
                return false;
            }

            Some(split_value) => {
                // Now we have everything (leaf_idx, split_idx, split_value)
                // So we can split the leaf into an internal node

                // Now route the data points into left / right child
                let data_inds = self.split_data(&rows, &feature_values, &split_value);

                // Ask the sampler to generate values for the new leaves to be added
                let leaf_vals = (
                    self.leaf_value(&data_inds.0, state),
                    self.leaf_value(&data_inds.1, state),
                );

                // Update the tree
                let (left_value, right_value) = leaf_vals;
                let new_inds = {
                    let msg = "Splitting a leaf failed, meaning the indices in particle were not consistent with the tree";
                    self.tree
                        .split_leaf_node(idx, split_idx, split_value, left_value, right_value)
                        .expect(msg)
                };

                // Remove the old index, we won't need it anymore
                self.indices.remove_index(idx);

                // Add the new leaves into expansion nodes etc.
                self.indices.add_index(new_inds.0, data_inds.0);
                self.indices.add_index(new_inds.1, data_inds.1);

                // Signal that the structure was updated
                return true;
            }
        }
    }

    // Generate predictions for this particle.
    // We do not need to traverse the tree, because during training
    // We simply keep track the leaf index where each data points lands
    pub fn predict(&self) -> Vec<f32> {
        let mut y_hat: Vec<f32> = vec![0.; self.params.n_points];

        for idx in &self.indices.leaf_nodes {
            let leaf = self.tree.get_node(idx).unwrap().as_leaf().unwrap();
            let row_inds = &self.indices.data_indices[idx];
            for i in row_inds {
                y_hat[*i] = leaf.value();
            }
        }

        y_hat
    }

    // For each pair of (row_index, feature_value) decide if that row will go to the left or right child
    fn split_data(
        &self,
        row_indices: &Vec<usize>,
        feature_values: &Vec<f32>,
        split_value: &f32,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut left_indices: Vec<usize> = vec![];
        let mut right_indices: Vec<usize> = vec![];

        for (idx, value) in std::iter::zip(row_indices, feature_values) {
            if value <= split_value {
                left_indices.push(*idx);
            } else {
                right_indices.push(*idx);
            }
        }

        (left_indices, right_indices)
    }

    // Returns a new sampled leaf value
    fn leaf_value(&self, data_indices: &Vec<usize>, state: &PgBartState) -> f32 {
        // TODO: This feels a bit off
        // This function takes as input indices of data points that ended in a particular leaf
        // Then calls the Sampler to fetch the predicted values for those data points
        // Calculates the mean
        // And calls the state again to sample a value around that mean
        let node_preds = state.predictions_subset(data_indices);
        let mu = math::mean(&node_preds);
        let value = state
            .probabilities()
            .sample_leaf_value(mu, self.params.kfactor);

        value
    }

    // --- Getters ---
    pub fn finished(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn split_variables(&self) -> Vec<usize> {
        self.tree.get_split_variables()
    }

    pub fn params(&self) -> &ParticleParams {
        &self.params
    }

    pub fn weight(&self) -> &Weight {
        &self.weight
    }

    pub fn weight_mut(&mut self) -> &mut Weight {
        &mut self.weight
    }
}
