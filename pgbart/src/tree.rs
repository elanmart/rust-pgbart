use std::collections::HashMap;

#[derive(Clone)]
pub struct Leaf {
    index: usize,
    value: f64,
}

#[derive(Clone)]
pub struct Internal {
    index: usize,
    split_idx: usize,
    split_value: f64,
}

#[derive(Clone)]
pub enum Node {
    Leaf(Leaf),
    Internal(Internal),
}

#[derive(Clone)]
pub struct Tree {
    nodes: HashMap<usize, Node>,
}

#[derive(Debug)]
pub enum TreeError {
    NotLeaf(usize),
    NotInternal(usize),
    IndexNotFound(usize),
}

impl Leaf {
    // Creates a new leaf
    pub fn new(index: usize, value: f64) -> Self {
        Leaf { index, value }
    }

    // --- Getters ---
    pub fn value(&self) -> f64 {
        self.value
    }
}

impl Internal {
    // Creates a new internal node
    fn new(index: usize, split_idx: usize, value: f64) -> Self {
        Internal {
            index,
            split_idx,
            split_value: value,
        }
    }

    // Returns the index of the left child for this node
    fn left(&self) -> usize {
        self.index * 2 + 1
    }

    // Returns the index of the right child for this node
    fn right(&self) -> usize {
        self.index * 2 + 2
    }
}

impl Node {
    // Create an internal node
    pub fn leaf(index: usize, value: f64) -> Self {
        Node::Leaf(Leaf::new(index, value))
    }

    // Create a leaf node
    pub fn internal(index: usize, split_idx: usize, split_value: f64) -> Self {
        Node::Internal(Internal::new(index, split_idx, split_value))
    }

    // Unpacks the node enum into a Leaf struct, or returns an Err
    pub fn as_leaf(&self) -> Result<&Leaf, TreeError> {
        match self {
            Node::Internal(n) => Err(TreeError::NotLeaf(n.index)),
            Node::Leaf(n) => Ok(n),
        }
    }

    // Unpacks the node enum into an Internal struct, or returns an Err
    fn as_internal(&self) -> Result<&Internal, TreeError> {
        match self {
            Node::Internal(n) => Ok(n),
            Node::Leaf(n) => Err(TreeError::NotInternal(n.index)),
        }
    }

    // Returns the index of this node
    fn index(&self) -> usize {
        match self {
            Node::Internal(n) => n.index,
            Node::Leaf(n) => n.index,
        }
    }

    // Returns the depth at which this node lives
    pub fn depth(&self) -> usize {
        ((self.index() + 1) as f64).log2().floor() as usize
    }
}

impl Tree {
    // Creates a tree with a single root node
    pub fn new(root_value: f64) -> Self {
        let root = Node::Leaf(Leaf::new(0, root_value));
        let nodes = HashMap::from_iter([(0, root)]);
        Tree { nodes }
    }

    // Returns the root node
    pub fn root(&self) -> &Node {
        self.nodes
            .get(&0)
            .expect("The tree should always have a root node at index 0")
    }

    // Returns the node at given index if it exists
    pub fn get_node(&self, idx: &usize) -> Result<&Node, TreeError> {
        self.nodes.get(idx).ok_or(TreeError::IndexNotFound(*idx))
    }

    // Makes sure that the node at given index is a leaf
    fn check_leaf(&self, idx: usize) -> Result<(), TreeError> {
        self.get_node(&idx)?.as_leaf()?;

        Ok(())
    }

    // Assigns a new node at a given index
    fn add_node(&mut self, node: Node) -> &Self {
        let idx = node.index();
        self.nodes.insert(idx, node);

        self
    }

    // Updates the value of a leaf node
    pub fn update_leaf_node(&mut self, idx: usize, value: f64) -> Result<(), TreeError> {
        self.check_leaf(idx)?;
        self.add_node(Node::leaf(idx, value));

        Ok(())
    }

    // Turns a leaf node into an internal node with two children
    // Returns the indices of newly created leaves
    pub fn split_leaf_node(
        &mut self,
        idx: usize,
        split_idx: usize,
        split_value: f64,
        left_value: f64,
        right_value: f64,
    ) -> Result<(usize, usize), TreeError> {
        self.check_leaf(idx)?;

        // Setup the parent and get the children indices
        let new = Internal::new(idx, split_idx, split_value);
        let (lix, rix) = (new.left(), new.right());

        // Set the nodes
        self.add_node(Node::Internal(new));
        self.add_node(Node::leaf(lix, left_value));
        self.add_node(Node::leaf(rix, right_value));

        // Done
        Ok((lix, rix))
    }

    // Returns a prediction for a given example
    pub fn predict(&self, x: &Vec<f64>) -> f64 {
        // We start with the root node
        let mut node = self.root();

        // And keep searching the tree
        let result = loop {
            match node {
                // Until we find a leaf
                Node::Leaf(n) => {
                    break n.value;
                }

                // Otherwise we go left or right
                Node::Internal(n) => {
                    // Depending on the value of the feature
                    let child_idx = if x[n.split_idx] <= n.split_value {
                        n.left()
                    } else {
                        n.right()
                    };

                    // We should panic if any internal node points to nothing
                    let msg = "Internal node should always point to a valid child";
                    node = self.get_node(&child_idx).expect(msg);
                }
            }
        };

        // done
        result
    }

    // Returns a list of split variables (covariates, features) used by this tree
    pub fn get_split_variables(&self) -> Vec<usize> {
        let mut ret: Vec<usize> = Vec::new();

        for item in self.nodes.values() {
            if let Ok(node) = item.as_internal() {
                ret.push(node.split_idx);
            }
        }

        ret
    }
}
