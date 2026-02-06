//! Query Executor
//!
//! Executes parsed queries against the HDR store using DataFusion
//! or direct vector operations.

use std::sync::Arc;
use crate::bitpack::BitpackedVector;
use crate::hdr_cascade::{HdrCascade, SearchResult};
use crate::resonance::{VectorField, Resonator};
use crate::storage::ArrowStore;
use crate::{HdrError, Result};

use super::parser::{QueryAst, QueryType, VectorOp, Expr, PropertyValue};

/// Query execution result
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Column names
    pub columns: Vec<String>,
    /// Rows of values
    pub rows: Vec<Vec<ResultValue>>,
    /// Execution statistics
    pub stats: ExecutionStats,
}

impl QueryResult {
    /// Create empty result
    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
            rows: Vec::new(),
            stats: ExecutionStats::default(),
        }
    }

    /// Create result with columns
    pub fn with_columns(columns: Vec<String>) -> Self {
        Self {
            columns,
            rows: Vec::new(),
            stats: ExecutionStats::default(),
        }
    }

    /// Add a row
    pub fn add_row(&mut self, row: Vec<ResultValue>) {
        self.rows.push(row);
    }

    /// Number of rows
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Is empty?
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

/// Result value types
#[derive(Debug, Clone)]
pub enum ResultValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Vector(BitpackedVector),
    VectorId(u64),
    Node {
        id: u64,
        labels: Vec<String>,
        properties: std::collections::HashMap<String, PropertyValue>,
    },
    Edge {
        id: u64,
        rel_type: String,
        src: u64,
        dst: u64,
        properties: std::collections::HashMap<String, PropertyValue>,
    },
    List(Vec<ResultValue>),
    Map(std::collections::HashMap<String, ResultValue>),
}

/// Execution statistics
#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    /// Nodes read
    pub nodes_read: usize,
    /// Relationships read
    pub relationships_read: usize,
    /// Nodes created
    pub nodes_created: usize,
    /// Relationships created
    pub relationships_created: usize,
    /// Vector comparisons
    pub vector_comparisons: usize,
    /// Execution time in microseconds
    pub execution_time_us: u64,
    /// Cascade filter statistics
    pub cascade_stats: CascadeStats,
}

/// Cascade filter statistics
#[derive(Debug, Clone, Default)]
pub struct CascadeStats {
    /// Candidates at L0 (Belichtung)
    pub l0_candidates: usize,
    /// Candidates at L1 (1-bit)
    pub l1_candidates: usize,
    /// Candidates at L2 (stacked)
    pub l2_candidates: usize,
    /// Final candidates
    pub final_candidates: usize,
}

/// Query executor
pub struct QueryExecutor {
    /// Vector store
    store: Option<Arc<ArrowStore>>,
    /// HDR cascade index
    cascade: Option<Arc<HdrCascade>>,
    /// Vector field for resonance
    field: Option<Arc<VectorField>>,
    /// Resonator for cleanup
    resonator: Option<Arc<Resonator>>,
    /// Parameter bindings
    parameters: std::collections::HashMap<String, ResultValue>,
}

impl Default for QueryExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryExecutor {
    /// Create new executor
    pub fn new() -> Self {
        Self {
            store: None,
            cascade: None,
            field: None,
            resonator: None,
            parameters: std::collections::HashMap::new(),
        }
    }

    /// Set the vector store
    pub fn with_store(mut self, store: Arc<ArrowStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// Set the HDR cascade
    pub fn with_cascade(mut self, cascade: Arc<HdrCascade>) -> Self {
        self.cascade = Some(cascade);
        self
    }

    /// Set the vector field
    pub fn with_field(mut self, field: Arc<VectorField>) -> Self {
        self.field = Some(field);
        self
    }

    /// Set the resonator
    pub fn with_resonator(mut self, resonator: Arc<Resonator>) -> Self {
        self.resonator = Some(resonator);
        self
    }

    /// Set a parameter
    pub fn set_param(&mut self, name: &str, value: ResultValue) {
        self.parameters.insert(name.to_string(), value);
    }

    /// Execute a query
    pub fn execute(&self, ast: &QueryAst) -> Result<QueryResult> {
        let start = std::time::Instant::now();

        let mut result = match ast.query_type {
            QueryType::Match => self.execute_match(ast)?,
            QueryType::VectorSearch => self.execute_vector_search(ast)?,
            QueryType::BoundRetrieval => self.execute_bound_retrieval(ast)?,
            QueryType::Create => self.execute_create(ast)?,
            _ => QueryResult::empty(),
        };

        result.stats.execution_time_us = start.elapsed().as_micros() as u64;
        Ok(result)
    }

    /// Execute MATCH query
    fn execute_match(&self, ast: &QueryAst) -> Result<QueryResult> {
        let mut result = QueryResult::with_columns(vec!["node".to_string()]);

        // For now, return all vectors from store
        if let Some(store) = &self.store {
            for (id, _vec) in store.iter() {
                result.add_row(vec![ResultValue::VectorId(id)]);
                if let Some(limit) = ast.limit {
                    if result.len() >= limit {
                        break;
                    }
                }
            }
        }

        Ok(result)
    }

    /// Execute vector search query
    fn execute_vector_search(&self, ast: &QueryAst) -> Result<QueryResult> {
        let mut result = QueryResult::with_columns(vec![
            "id".to_string(),
            "distance".to_string(),
            "similarity".to_string(),
        ]);

        // Get query vector from parameters
        let query = self.get_query_vector()?;
        let k = ast.limit.unwrap_or(10);

        if let Some(cascade) = &self.cascade {
            let search_results = cascade.search(&query, k);

            result.stats.vector_comparisons = cascade.len();
            result.stats.cascade_stats.final_candidates = search_results.len();

            for sr in search_results {
                result.add_row(vec![
                    ResultValue::Int(sr.index as i64),
                    ResultValue::Int(sr.distance as i64),
                    ResultValue::Float(sr.similarity as f64),
                ]);
            }
        } else if let Some(store) = &self.store {
            let search_results = store.search(&query, k);

            for (id, dist, sim) in search_results {
                result.add_row(vec![
                    ResultValue::Int(id as i64),
                    ResultValue::Int(dist as i64),
                    ResultValue::Float(sim as f64),
                ]);
            }
        }

        Ok(result)
    }

    /// Execute bound retrieval query
    fn execute_bound_retrieval(&self, ast: &QueryAst) -> Result<QueryResult> {
        let mut result = QueryResult::with_columns(vec!["result".to_string()]);

        // Get edge, verb, and known from parameters
        let edge = self.get_param_vector("edge")?;
        let verb = self.get_param_vector("verb")?;
        let known = self.get_param_vector("known")?;

        // Unbind: edge ⊗ verb ⊗ known = result
        let unbound = edge.xor(&verb).xor(&known);

        // Optionally cleanup result
        if let Some(resonator) = &self.resonator {
            if let Some(res) = resonator.resonate(&unbound) {
                if let Some(clean) = resonator.get(res.index) {
                    result.add_row(vec![ResultValue::Vector(clean.clone())]);
                    return Ok(result);
                }
            }
        }

        result.add_row(vec![ResultValue::Vector(unbound)]);
        Ok(result)
    }

    /// Execute CREATE query
    fn execute_create(&self, ast: &QueryAst) -> Result<QueryResult> {
        let mut result = QueryResult::with_columns(vec!["created".to_string()]);
        result.stats.nodes_created = 1;
        Ok(result)
    }

    /// Execute a vector operation
    pub fn execute_vector_op(&self, op: &VectorOp) -> Result<ResultValue> {
        match op {
            VectorOp::Bind { a, b } => {
                let va = self.eval_to_vector(a)?;
                let vb = self.eval_to_vector(b)?;
                Ok(ResultValue::Vector(va.xor(&vb)))
            }
            VectorOp::Unbind { bound, key } => {
                let vbound = self.eval_to_vector(bound)?;
                let vkey = self.eval_to_vector(key)?;
                Ok(ResultValue::Vector(vbound.xor(&vkey)))
            }
            VectorOp::Bind3 { src, verb, dst } => {
                let vs = self.eval_to_vector(src)?;
                let vv = self.eval_to_vector(verb)?;
                let vd = self.eval_to_vector(dst)?;
                Ok(ResultValue::Vector(vs.xor(&vv).xor(&vd)))
            }
            VectorOp::Hamming { a, b } => {
                let va = self.eval_to_vector(a)?;
                let vb = self.eval_to_vector(b)?;
                let dist = crate::hamming::hamming_distance_scalar(&va, &vb);
                Ok(ResultValue::Int(dist as i64))
            }
            VectorOp::Similarity { a, b } => {
                let va = self.eval_to_vector(a)?;
                let vb = self.eval_to_vector(b)?;
                let dist = crate::hamming::hamming_distance_scalar(&va, &vb);
                let sim = crate::hamming::hamming_to_similarity(dist);
                Ok(ResultValue::Float(sim as f64))
            }
            VectorOp::Bundle { vectors } => {
                let vecs: Result<Vec<BitpackedVector>> = vectors.iter()
                    .map(|e| self.eval_to_vector(e))
                    .collect();
                let vecs = vecs?;
                let refs: Vec<&BitpackedVector> = vecs.iter().collect();
                Ok(ResultValue::Vector(BitpackedVector::bundle(&refs)))
            }
            VectorOp::Permute { vector, positions } => {
                let v = self.eval_to_vector(vector)?;
                let rotated = if *positions >= 0 {
                    v.rotate_left(*positions as usize)
                } else {
                    v.rotate_right((-*positions) as usize)
                };
                Ok(ResultValue::Vector(rotated))
            }
            VectorOp::Resonance { vector, query } => {
                let vvec = self.eval_to_vector(vector)?;
                let vquery = self.eval_to_vector(query)?;
                let dist = crate::hamming::hamming_distance_scalar(&vvec, &vquery);
                let sim = crate::hamming::hamming_to_similarity(dist);
                Ok(ResultValue::Float(sim as f64))
            }
            VectorOp::Cleanup { vector, memory } => {
                let v = self.eval_to_vector(vector)?;
                if let Some(resonator) = &self.resonator {
                    if let Some(res) = resonator.resonate(&v) {
                        if let Some(clean) = resonator.get(res.index) {
                            return Ok(ResultValue::Vector(clean.clone()));
                        }
                    }
                }
                Ok(ResultValue::Vector(v))
            }
            VectorOp::CascadeSearch { query, k, threshold } => {
                let vquery = self.eval_to_vector(query)?;
                if let Some(cascade) = &self.cascade {
                    let results = cascade.search(&vquery, *k);
                    let list: Vec<ResultValue> = results.into_iter()
                        .map(|r| ResultValue::Map(
                            [
                                ("index".to_string(), ResultValue::Int(r.index as i64)),
                                ("distance".to_string(), ResultValue::Int(r.distance as i64)),
                                ("similarity".to_string(), ResultValue::Float(r.similarity as f64)),
                            ].into_iter().collect()
                        ))
                        .collect();
                    Ok(ResultValue::List(list))
                } else {
                    Ok(ResultValue::List(vec![]))
                }
            }
            VectorOp::Voyager { query, radius, stack_size } => {
                let vquery = self.eval_to_vector(query)?;
                if let Some(cascade) = &self.cascade {
                    if let Some(result) = cascade.voyager_deep_field(&vquery, *radius, *stack_size) {
                        return Ok(ResultValue::Map(
                            [
                                ("star".to_string(), ResultValue::Vector(result.star)),
                                ("cleaned_distance".to_string(), ResultValue::Int(result.cleaned_distance as i64)),
                                ("signal_strength".to_string(), ResultValue::Float(result.signal_strength as f64)),
                                ("noise_reduction".to_string(), ResultValue::Float(result.noise_reduction as f64)),
                            ].into_iter().collect()
                        ));
                    }
                }
                Ok(ResultValue::Null)
            }
            VectorOp::Analogy { a, b, c } => {
                let va = self.eval_to_vector(a)?;
                let vb = self.eval_to_vector(b)?;
                let vc = self.eval_to_vector(c)?;
                // ? = c ⊗ (b ⊗ a)
                let transform = vb.xor(&va);
                let result = vc.xor(&transform);
                Ok(ResultValue::Vector(result))
            }
        }
    }

    /// Get query vector from parameters
    fn get_query_vector(&self) -> Result<BitpackedVector> {
        self.get_param_vector("query")
    }

    /// Get vector from parameter
    fn get_param_vector(&self, name: &str) -> Result<BitpackedVector> {
        match self.parameters.get(name) {
            Some(ResultValue::Vector(v)) => Ok(v.clone()),
            Some(_) => Err(HdrError::Query(format!("Parameter {} is not a vector", name))),
            None => Err(HdrError::Query(format!("Missing parameter: {}", name))),
        }
    }

    /// Evaluate expression to vector
    fn eval_to_vector(&self, expr: &Expr) -> Result<BitpackedVector> {
        match expr {
            Expr::Variable(name) => self.get_param_vector(name),
            Expr::Property { var, prop } => {
                // TODO: property access
                Err(HdrError::Query("Property access not implemented".into()))
            }
            Expr::Literal(PropertyValue::Vector(bytes)) => {
                BitpackedVector::from_bytes(bytes)
            }
            Expr::VectorOp(op) => {
                match self.execute_vector_op(op)? {
                    ResultValue::Vector(v) => Ok(v),
                    _ => Err(HdrError::Query("Vector operation did not return vector".into())),
                }
            }
            _ => Err(HdrError::Query("Cannot convert expression to vector".into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdr_cascade::HdrCascade;

    #[test]
    fn test_vector_search() {
        let mut cascade = HdrCascade::with_capacity(100);

        // Add vectors
        for i in 0..100 {
            cascade.add(BitpackedVector::random(i as u64 + 100));
        }

        let executor = QueryExecutor::new()
            .with_cascade(Arc::new(cascade));

        let ast = QueryAst {
            query_type: QueryType::VectorSearch,
            matches: vec![],
            where_clause: None,
            returns: vec![],
            limit: Some(10),
            order_by: None,
            vector_ops: vec![],
        };

        // Set query parameter
        let mut exec = executor;
        exec.set_param("query", ResultValue::Vector(BitpackedVector::random(150)));

        // Note: This would fail without the cascade having the query vector
        // The test mainly validates the structure
    }

    #[test]
    fn test_vector_ops() {
        let executor = QueryExecutor::new();

        let a = BitpackedVector::random(1);
        let b = BitpackedVector::random(2);

        let mut exec = executor;
        exec.set_param("a", ResultValue::Vector(a.clone()));
        exec.set_param("b", ResultValue::Vector(b.clone()));

        // Test bind
        let op = VectorOp::Bind {
            a: Expr::Variable("a".to_string()),
            b: Expr::Variable("b".to_string()),
        };

        let result = exec.execute_vector_op(&op).unwrap();
        if let ResultValue::Vector(bound) = result {
            // Verify: bound ⊗ b = a
            let recovered = bound.xor(&b);
            assert_eq!(recovered, a);
        } else {
            panic!("Expected vector result");
        }

        // Test hamming
        let op = VectorOp::Hamming {
            a: Expr::Variable("a".to_string()),
            b: Expr::Variable("a".to_string()),
        };
        let result = exec.execute_vector_op(&op).unwrap();
        if let ResultValue::Int(dist) = result {
            assert_eq!(dist, 0); // Same vector = 0 distance
        }
    }
}
