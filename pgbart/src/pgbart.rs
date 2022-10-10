#![allow(non_snake_case)]

use rand::distributions::WeightedIndex;
use rand_distr::{Distribution, Normal, Uniform};

use crate::math::{self, Matrix};
use crate::particle::{Particle, ParticleParams};

use rand::{self, Rng};

// Settings for the Particle Gibbs Sampler
pub struct PgBartSettings {
    n_trees: usize,             // Number of trees in the ensemble
    n_particles: usize,         // Number of particles to spawn in each iteration
    alpha: f32,                 // Prior split probability
    default_kf: f32,            // Standard deviation of noise added during leaf value sampling
    batch: (f32, f32),          // How many trees to update in tuning / final phase
    intial_alpha_vec: Vec<f32>, // Prior on covariates to use as splits
}

// Struct with helpers and settings for most (all?) random things in the algorithm
pub struct Probabilities {
    normal: Normal<f32>,      // distro for sampling unit gaussian.
    uniform: Uniform<f32>,    // distro for sampling uniformly from a pre-defined range
    alpha_vec: Vec<f32>,      // prior for variable selection
    spliting_probs: Vec<f32>, // posterior for variable selection
    alpha: f32,               // prior split probability
}

// We had to use the trait here, because otherwise
// it seems impossible to have a clean callback into Python
pub trait ExternalData {
    fn X(&self) -> &Matrix<f32>;
    fn y(&self) -> &Vec<f32>;
    fn model_logp(&self, v: &Vec<f32>) -> f32;
}

// The core of the algorithm
pub struct PgBartState {
    data: Box<dyn ExternalData>,  // dataset we're training on
    params: PgBartSettings,       // hyperparams
    probabilities: Probabilities, // helpers and settings for most (all?) random things in the algorithm
    predictions: Vec<f32>,        // current bart predictions, one per data point
    particles: Vec<Particle>,     // m particles, one per tree
    variable_inclusion: Vec<u32>, // feature importance
    tune: bool,                   // tuning phase indicator
}

impl PgBartSettings {
    // I think we either need to implement this dummy `new`, or make all the fields `pub`?
    pub fn new(
        n_trees: usize,
        n_particles: usize,
        alpha: f32,
        default_kf: f32,
        batch: (f32, f32),
        intial_alpha_vec: Vec<f32>,
    ) -> Self {
        Self {
            n_trees,
            n_particles,
            alpha,
            default_kf,
            batch,
            intial_alpha_vec,
        }
    }
}

impl Probabilities {
    // Sample a boolean flag indicating if a node should be split or not
    pub fn sample_expand_flag(&self, depth: usize) -> bool {
        let mut rng = rand::thread_rng();

        let p = 1. - self.alpha.powi(depth as i32);
        let res = p < rng.gen::<f32>();

        res
    }

    // Sample a new value for a leaf node
    pub fn sample_leaf_value(&self, mu: f32, kfactor: f32) -> f32 {
        let mut rng = rand::thread_rng();

        let norm = self.normal.sample(&mut rng) * kfactor;

        norm + mu
    }

    // Sample the index of a feature to split on
    pub fn sample_split_index(&self) -> usize {
        let mut rng = rand::thread_rng();

        let p = rng.gen::<f32>();
        for (idx, value) in self.spliting_probs.iter().enumerate() {
            if p <= *value {
                return idx;
            }
        }

        self.spliting_probs.len() - 1
    }

    // Sample a boolean flag indicating if a node should be split or not
    pub fn sample_split_value(&self, candidates: &Vec<f32>) -> Option<f32> {
        let mut rng = rand::thread_rng();

        if candidates.len() == 0 {
            None
        } else {
            let dist = Uniform::<usize>::new(0, candidates.len());
            let idx = dist.sample(&mut rng);
            Some(candidates[idx])
        }
    }

    // Sample a new kf
    pub fn sample_kf(&self) -> f32 {
        let mut rng = rand::thread_rng();
        let kf = self.uniform.sample(&mut rng);

        kf
    }

    // Sample an index according to normalized weights
    pub fn select_particle(&self, mut particles: Vec<Particle>, weights: &Vec<f32>) -> Particle {
        let mut rng = rand::thread_rng();

        let dist = WeightedIndex::new(weights).unwrap();
        let idx = dist.sample(&mut rng);
        let selected = particles.swap_remove(idx);

        selected
    }

    // Resample the particles according to the weights vector
    fn resample_particles(&self, particles: Vec<Particle>, weights: &Vec<f32>) -> Vec<Particle> {
        let mut rng = rand::thread_rng();

        let dist = WeightedIndex::new(weights).unwrap();
        let mut ret: Vec<Particle> = Vec::with_capacity(particles.len());

        if weights.len() != (particles.len() - 2) {
            panic!("Weights and particles mismatch");
        }

        // TODO: could this be optimized? Keep in mind that borrow checker
        // will not let us "move" any item out of a vector
        // using "remove" is slow
        // and using "swap_remove" will mess up the alignment between weights and particles
        // so "cloning" everything might be the best choice actually?
        ret.push(particles[0].clone());
        ret.push(particles[1].clone());
        for _ in 2..particles.len() {
            let idx = dist.sample(&mut rng) + 2;
            ret.push(particles[idx].clone());
        }

        ret
    }
}

impl PgBartState {
    // Initialize the Particle Gibbs sampler
    pub fn new(params: PgBartSettings, data: Box<dyn ExternalData>) -> Self {
        // Unpack
        let X = data.X();
        let y = data.y();
        let m = params.n_trees as f32;
        let mu = math::mean(y);
        let leaf_value = mu / m;

        // Standard deviation for binary / real data
        let binary = y.iter().all(|v| (*v == 0.) || (*v == 1.));
        let std = if binary {
            3. / m.powf(0.5)
        } else {
            math::stdev(y) / m.powf(0.5)
        };

        // Initialize the predictions at first iteration. Also initialize feat importance
        let predictions: Vec<f32> = vec![mu; X.n_rows];
        let variable_inclusion: Vec<u32> = vec![0; X.n_cols];

        // Initilize the trees (m trees with root nodes only)
        // We store the trees wrapped with Particle structs since it simplifies the code
        let mut particles: Vec<Particle> = Vec::with_capacity(params.n_trees);
        for _ in 0..params.n_trees {
            let p_params = ParticleParams::new(X.n_rows, X.n_cols, params.default_kf);
            let p = Particle::new(p_params, leaf_value);
            particles.push(p);
        }

        // Sampling probabilities
        let alpha_vec: Vec<f32> = params.intial_alpha_vec.clone(); // We will be updating those, hence the clone
        let spliting_probs: Vec<f32> = math::normalized_cumsum(&alpha_vec);
        let probabilities = Probabilities {
            alpha_vec,
            spliting_probs,
            alpha: params.alpha,
            normal: Normal::new(0., std).unwrap(),
            uniform: Uniform::new(0.33, 0.75), // TODO: parametrize this?
        };

        // Done
        PgBartState {
            params,
            data,
            particles,
            probabilities,
            predictions,
            variable_inclusion,
            tune: true,
        }
    }

    pub fn step(&mut self) {
        // Setup
        let mut rng = rand::thread_rng();

        // Get the indices of the trees we'll be modifying
        let amount = self.num_to_update();
        let length = self.params.n_trees;
        let indices = rand::seq::index::sample(&mut rng, length, amount);

        // Get the default prediction for a new particle
        let y = self.data.y();
        let mu = math::mean(y) / (self.params.n_particles as f32);

        // Modify each tree sequentially
        for particle_index in indices {
            // Fetch the tree to modify. We store the trees wrapped with Particle structs.
            let selected_p = &self.particles[particle_index];
            let local_preds = math::sub(&self.predictions, &selected_p.predict());

            // Initialize local particles
            // note that while self.particles has size n_trees
            // local_particles has size n_particles
            // and all particles in local_particles
            // essentially are modifications of a single tree
            let local_particles = self.initialize_particles(selected_p, &local_preds, mu);

            // Now we run the inner loop
            // where we grow + resample the particles multiple times
            let local_particles = self.grow_particles(local_particles, &local_preds);

            // Normalize all the weights
            let (_, weights) = self.normalize_weights(&local_particles);

            // Sample a single tree (particle) to be kept for the next iteration of the PG sampler
            let mut selected = {
                self.probabilities
                    .select_particle(local_particles, &weights)
            };

            // Line 20 of the algo in the paper
            // "compute particle weight next round"
            let log_n_particles = (self.params.n_particles as f32).ln();
            let log_lik = selected.weight().log_likelihood();

            selected.weight_mut().set_log_w(log_lik - log_n_particles);

            // Update the probabilities of sampling each covariate if we're in the tuning phase
            // Otherwise update the feature importance counter
            self.update_sampling_probs(&selected);

            // Update the predictions
            self.predictions = math::add(&local_preds, &selected.predict());

            // Update the tree
            self.particles[particle_index] = selected;
        }
    }

    fn initialize_particles(&self, p: &Particle, local_preds: &Vec<f32>, mu: f32) -> Vec<Particle> {
        // The first particle is the exact copy of the selected tree
        let p0 = p.frozen_copy();

        // The second particle retains the tree structure, but re-samples the leaf values
        let p1 = p.with_resampled_leaves(self);

        // Initialize the vector
        let mut local_particles = vec![p0, p1];

        // Reset the weights on the first two particles
        for item in &mut local_particles {
            let preds = math::add(&local_preds, &item.predict());
            let log_lik = self.data.model_logp(&preds);
            item.weight_mut().reset(log_lik);
        }

        // The rest of the particles starts as empty trees (root node only);
        for _ in 2..self.params.n_particles {
            // Change the kf if we're in the tuning phase
            let params = if self.tune {
                let kf = self.probabilities.sample_kf();
                p.params().with_new_kf(kf)
            } else {
                p.params().clone()
            };

            // Create and add to the list
            let new_p = Particle::new(params, mu);
            local_particles.push(new_p);
        }

        // Done
        local_particles
    }

    fn grow_particles(
        &self,
        mut particles: Vec<Particle>,
        local_preds: &Vec<f32>,
    ) -> Vec<Particle> {
        // We'll need the data to grow the particles
        let X = self.data.X();

        // Now we can start growing the local_particles
        loop {
            // Break if there is nothing to update anymore
            if particles.iter().all(|p| p.finished()) {
                break;
            }

            // We iterate over to_update, keeping the first two unchanged
            for p in &mut particles[2..] {
                // Update the tree inside it
                let needs_update = p.grow(X, self);

                // Update the weight if needed
                if needs_update {
                    let preds = math::add(&local_preds, &p.predict());
                    let loglik = self.data.model_logp(&preds);
                    p.weight_mut().update(loglik);
                }
            }

            // Normalize the weights of the updatable particles
            // See the normalize_weights for some helpful break down on how it's done
            let (wt, weights) = self.normalize_weights(&particles[2..]);

            // Note: the weights are of size (n_particles - 2)
            // That's because resmple() will keep the first two particles anyway
            particles = self.probabilities.resample_particles(particles, &weights);

            // Set the log-weight of the particles 2.. to wt -- log mean weight
            for p in &mut particles[2..] {
                p.weight_mut().set_log_w(wt);
            }
        }

        // Set the log-weight of each particle to the last log likelihood
        for p in &mut particles {
            let loglik = p.weight().log_likelihood();
            p.weight_mut().set_log_w(loglik);
        }

        // Done
        particles
    }

    fn update_sampling_probs(&mut self, p: &Particle) {
        // Get the indices of covariates used by this particle
        let used_variates = p.split_variables();

        // During tuning phase, we update the probabilities
        if self.tune {
            let probs = math::normalized_cumsum(&self.probabilities.alpha_vec);
            self.probabilities.spliting_probs = probs;
            for idx in used_variates {
                self.probabilities.alpha_vec[idx] += 1.;
            }

        // Otherwise we just record the counts
        } else {
            for idx in used_variates {
                self.variable_inclusion[idx] += 1;
            }
        }
    }

    // Normalize the particle weights
    fn normalize_weights(&self, particles: &[Particle]) -> (f32, Vec<f32>) {
        let log_weights = { particles.iter().map(|p| p.weight().log_w()) };

        let max_log_weight = {
            particles
                .iter()
                .map(|p| p.weight().log_w())
                .fold(f32::MIN, |a, b| a.max(b))
        };

        let scaled_log_weights = log_weights.map(|logw| logw - max_log_weight);

        // At this point we have
        // scaled_weights = weights / weights.max()
        let scaled_weights: Vec<f32> = scaled_log_weights.map(|x| x.exp()).collect();

        // normalized_scaled_weights = weights / weights.sum()
        let sum_scaled_weights: f32 = scaled_weights.iter().sum();
        let normalized_scaled_weights: Vec<f32> = scaled_weights
            .iter()
            .map(|w| (w / sum_scaled_weights) + 1e-12)
            .collect();

        // The un-normalized weight assigned to each updatable particle after single growing cycle
        let log_sum_scaled_weights = sum_scaled_weights.ln();
        let log_n_particles = (self.params.n_particles as f32).ln();
        let log_w: f32 = max_log_weight + log_sum_scaled_weights - log_n_particles;
        // log_w = log ( weights.max() * scaled_weights.mean() )

        (log_w, normalized_scaled_weights)
    }

    // Returns the number of trees we should modify in the current phase
    fn num_to_update(&self) -> usize {
        let fraction = if self.tune {
            self.params.batch.0
        } else {
            self.params.batch.1
        };

        ((self.params.n_trees as f32) * fraction).floor() as usize
    }

    // Get predictions for a subset of data points
    pub fn predictions_subset(&self, indices: &Vec<usize>) -> Vec<f32> {
        let all_preds = &self.predictions;
        let mut ret = Vec::<f32>::new();

        for row_idx in indices {
            ret.push(all_preds[*row_idx]);
        }

        ret
    }

    // --- Getters ---
    pub fn probabilities(&self) -> &Probabilities {
        &self.probabilities
    }

    pub fn predictions(&self) -> &Vec<f32> {
        &self.predictions
    }

    // --- Setters ---
    pub fn set_tune(&mut self, tune: bool) {
        self.tune = tune;
    }
}
