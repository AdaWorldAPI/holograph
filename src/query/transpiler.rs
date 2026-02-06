//! Query Transpilers
//!
//! Convert GQL Alchemy syntax to DataFusion SQL and RedisGraph Cypher.

use super::parser::{QueryAst, QueryType, VectorOp, Expr, PropertyValue};

/// Transpile to DataFusion SQL
pub struct CypherTranspiler {
    /// Parameter bindings
    parameters: std::collections::HashMap<String, String>,
}

impl Default for CypherTranspiler {
    fn default() -> Self {
        Self::new()
    }
}

impl CypherTranspiler {
    pub fn new() -> Self {
        Self {
            parameters: std::collections::HashMap::new(),
        }
    }

    /// Set a parameter binding
    pub fn bind(&mut self, name: &str, value: &str) {
        self.parameters.insert(name.to_string(), value.to_string());
    }

    /// Transpile AST to DataFusion SQL
    pub fn to_sql(&self, ast: &QueryAst) -> String {
        match ast.query_type {
            QueryType::Match | QueryType::VectorSearch => {
                self.match_to_sql(ast)
            }
            QueryType::Create => {
                self.create_to_sql(ast)
            }
            QueryType::BoundRetrieval => {
                self.bound_retrieval_to_sql(ast)
            }
            _ => String::new()
        }
    }

    fn match_to_sql(&self, ast: &QueryAst) -> String {
        let mut sql = String::new();

        // Convert MATCH patterns to SQL JOINs
        sql.push_str("SELECT ");

        // Return clause
        if ast.returns.is_empty() {
            sql.push_str("*");
        } else {
            sql.push_str("*"); // Simplified
        }

        sql.push_str(" FROM nodes");

        // WHERE clause with vector operations
        if let Some(where_clause) = &ast.where_clause {
            sql.push_str(" WHERE ");
            // Convert predicates to SQL
        }

        // Vector operations become UDF calls
        for op in &ast.vector_ops {
            match op {
                VectorOp::Hamming { a, b } => {
                    sql.push_str(" /* hamming_distance(...) */");
                }
                VectorOp::Similarity { a, b } => {
                    sql.push_str(" /* vector_similarity(...) */");
                }
                VectorOp::CascadeSearch { query, k, threshold } => {
                    sql.push_str(&format!(
                        " /* cascade_search(query, {}, {:?}) */",
                        k, threshold
                    ));
                }
                _ => {}
            }
        }

        // Limit
        if let Some(limit) = ast.limit {
            sql.push_str(&format!(" LIMIT {}", limit));
        }

        sql
    }

    fn create_to_sql(&self, ast: &QueryAst) -> String {
        "INSERT INTO nodes DEFAULT VALUES".to_string()
    }

    fn bound_retrieval_to_sql(&self, ast: &QueryAst) -> String {
        // Bound retrieval becomes a function call
        "SELECT unbind_vector(edge, verb, known) AS result FROM edges".to_string()
    }

    /// Transpile vector operation to SQL UDF call
    pub fn vector_op_to_sql(&self, op: &VectorOp) -> String {
        match op {
            VectorOp::Bind { a, b } => {
                format!("vector_bind({}, {})",
                    self.expr_to_sql(a),
                    self.expr_to_sql(b))
            }
            VectorOp::Unbind { bound, key } => {
                format!("vector_unbind({}, {})",
                    self.expr_to_sql(bound),
                    self.expr_to_sql(key))
            }
            VectorOp::Bind3 { src, verb, dst } => {
                format!("vector_bind3({}, {}, {})",
                    self.expr_to_sql(src),
                    self.expr_to_sql(verb),
                    self.expr_to_sql(dst))
            }
            VectorOp::Resonance { vector, query } => {
                format!("vector_resonance({}, {})",
                    self.expr_to_sql(vector),
                    self.expr_to_sql(query))
            }
            VectorOp::Cleanup { vector, memory } => {
                let mem = memory.as_deref().unwrap_or("default");
                format!("vector_cleanup({}, '{}')",
                    self.expr_to_sql(vector), mem)
            }
            VectorOp::Bundle { vectors } => {
                let args: Vec<_> = vectors.iter()
                    .map(|v| self.expr_to_sql(v))
                    .collect();
                format!("vector_bundle({})", args.join(", "))
            }
            VectorOp::Hamming { a, b } => {
                format!("hamming_distance({}, {})",
                    self.expr_to_sql(a),
                    self.expr_to_sql(b))
            }
            VectorOp::Similarity { a, b } => {
                format!("vector_similarity({}, {})",
                    self.expr_to_sql(a),
                    self.expr_to_sql(b))
            }
            VectorOp::Permute { vector, positions } => {
                format!("vector_permute({}, {})",
                    self.expr_to_sql(vector), positions)
            }
            VectorOp::CascadeSearch { query, k, threshold } => {
                let thresh = threshold.map_or("NULL".to_string(), |t| t.to_string());
                format!("cascade_search({}, {}, {})",
                    self.expr_to_sql(query), k, thresh)
            }
            VectorOp::Voyager { query, radius, stack_size } => {
                format!("voyager_search({}, {}, {})",
                    self.expr_to_sql(query), radius, stack_size)
            }
            VectorOp::Analogy { a, b, c } => {
                format!("vector_analogy({}, {}, {})",
                    self.expr_to_sql(a),
                    self.expr_to_sql(b),
                    self.expr_to_sql(c))
            }
        }
    }

    fn expr_to_sql(&self, expr: &Expr) -> String {
        match expr {
            Expr::Literal(v) => self.value_to_sql(v),
            Expr::Variable(name) => name.clone(),
            Expr::Property { var, prop } => format!("{}.{}", var, prop),
            Expr::Function { name, args } => {
                let arg_strs: Vec<_> = args.iter()
                    .map(|a| self.expr_to_sql(a))
                    .collect();
                format!("{}({})", name, arg_strs.join(", "))
            }
            Expr::VectorOp(op) => self.vector_op_to_sql(op),
            Expr::Arithmetic { left, op, right } => {
                let op_str = match op {
                    super::parser::ArithOp::Add => "+",
                    super::parser::ArithOp::Sub => "-",
                    super::parser::ArithOp::Mul => "*",
                    super::parser::ArithOp::Div => "/",
                    super::parser::ArithOp::Mod => "%",
                    super::parser::ArithOp::Pow => "^",
                };
                format!("({} {} {})",
                    self.expr_to_sql(left), op_str, self.expr_to_sql(right))
            }
            Expr::Case { whens, else_expr } => {
                let mut sql = "CASE".to_string();
                for (pred, expr) in whens {
                    sql.push_str(&format!(" WHEN ... THEN {}", self.expr_to_sql(expr)));
                }
                if let Some(e) = else_expr {
                    sql.push_str(&format!(" ELSE {}", self.expr_to_sql(e)));
                }
                sql.push_str(" END");
                sql
            }
        }
    }

    fn value_to_sql(&self, value: &PropertyValue) -> String {
        match value {
            PropertyValue::Null => "NULL".to_string(),
            PropertyValue::Bool(b) => if *b { "TRUE" } else { "FALSE" }.to_string(),
            PropertyValue::Int(i) => i.to_string(),
            PropertyValue::Float(f) => f.to_string(),
            PropertyValue::String(s) => format!("'{}'", s.replace('\'', "''")),
            PropertyValue::List(items) => {
                let strs: Vec<_> = items.iter().map(|i| self.value_to_sql(i)).collect();
                format!("ARRAY[{}]", strs.join(", "))
            }
            PropertyValue::Map(m) => {
                // JSON object
                let pairs: Vec<_> = m.iter()
                    .map(|(k, v)| format!("'{}': {}", k, self.value_to_sql(v)))
                    .collect();
                format!("{{{}}}", pairs.join(", "))
            }
            PropertyValue::Vector(bytes) => {
                // Hex encode vector
                let hex: String = bytes.iter()
                    .map(|b| format!("{:02x}", b))
                    .collect();
                format!("X'{}'", hex)
            }
            PropertyValue::Parameter(name) => {
                self.parameters.get(name)
                    .cloned()
                    .unwrap_or_else(|| format!("${}", name))
            }
        }
    }
}

/// Transpile GQL Alchemy to standard Cypher
pub struct GqlTranspiler {
    /// Vector function mappings
    vector_functions: std::collections::HashMap<String, String>,
}

impl Default for GqlTranspiler {
    fn default() -> Self {
        Self::new()
    }
}

impl GqlTranspiler {
    pub fn new() -> Self {
        let mut vector_functions = std::collections::HashMap::new();

        // Map GQL Alchemy functions to Cypher procedures
        vector_functions.insert("BIND".to_string(), "hdr.bind".to_string());
        vector_functions.insert("UNBIND".to_string(), "hdr.unbind".to_string());
        vector_functions.insert("RESONANCE".to_string(), "hdr.resonance".to_string());
        vector_functions.insert("CLEANUP".to_string(), "hdr.cleanup".to_string());
        vector_functions.insert("HAMMING".to_string(), "hdr.hamming".to_string());
        vector_functions.insert("SIMILARITY".to_string(), "hdr.similarity".to_string());
        vector_functions.insert("CASCADE_SEARCH".to_string(), "hdr.cascadeSearch".to_string());
        vector_functions.insert("VOYAGER".to_string(), "hdr.voyagerSearch".to_string());
        vector_functions.insert("BUNDLE".to_string(), "hdr.bundle".to_string());
        vector_functions.insert("ANALOGY".to_string(), "hdr.analogy".to_string());

        Self { vector_functions }
    }

    /// Transpile AST to Cypher
    pub fn to_cypher(&self, ast: &QueryAst) -> String {
        match ast.query_type {
            QueryType::Match => self.match_to_cypher(ast),
            QueryType::VectorSearch => self.vector_search_to_cypher(ast),
            QueryType::BoundRetrieval => self.bound_retrieval_to_cypher(ast),
            QueryType::Create => self.create_to_cypher(ast),
            _ => String::new()
        }
    }

    fn match_to_cypher(&self, ast: &QueryAst) -> String {
        let mut cypher = String::from("MATCH ");

        // Patterns would go here
        cypher.push_str("(n)");

        if ast.where_clause.is_some() {
            cypher.push_str("\nWHERE ");
            // Convert predicates
        }

        cypher.push_str("\nRETURN ");
        if ast.returns.is_empty() {
            cypher.push_str("n");
        }

        if let Some(limit) = ast.limit {
            cypher.push_str(&format!("\nLIMIT {}", limit));
        }

        cypher
    }

    fn vector_search_to_cypher(&self, ast: &QueryAst) -> String {
        let mut cypher = String::new();

        // Convert vector search to Cypher with procedure calls
        cypher.push_str("CALL hdr.search($query, $k)\n");
        cypher.push_str("YIELD node, distance, similarity\n");
        cypher.push_str("RETURN node, distance, similarity");

        if let Some(limit) = ast.limit {
            cypher.push_str(&format!("\nLIMIT {}", limit));
        }

        cypher
    }

    fn bound_retrieval_to_cypher(&self, ast: &QueryAst) -> String {
        // Convert bound retrieval to Cypher function call
        let mut cypher = String::new();

        cypher.push_str("RETURN hdr.unbind($edge, $verb, $known) AS result");

        cypher
    }

    fn create_to_cypher(&self, ast: &QueryAst) -> String {
        "CREATE (n) RETURN n".to_string()
    }

    /// Transpile vector operation to Cypher procedure call
    pub fn vector_op_to_cypher(&self, op: &VectorOp) -> String {
        match op {
            VectorOp::Bind { a, b } => {
                format!("hdr.bind({}, {})",
                    self.expr_to_cypher(a),
                    self.expr_to_cypher(b))
            }
            VectorOp::Unbind { bound, key } => {
                format!("hdr.unbind({}, {})",
                    self.expr_to_cypher(bound),
                    self.expr_to_cypher(key))
            }
            VectorOp::CascadeSearch { query, k, threshold } => {
                let thresh = threshold.map_or("null".to_string(), |t| t.to_string());
                format!("hdr.cascadeSearch({}, {}, {})",
                    self.expr_to_cypher(query), k, thresh)
            }
            _ => "/* unsupported vector op */".to_string()
        }
    }

    fn expr_to_cypher(&self, expr: &Expr) -> String {
        match expr {
            Expr::Literal(v) => self.value_to_cypher(v),
            Expr::Variable(name) => name.clone(),
            Expr::Property { var, prop } => format!("{}.{}", var, prop),
            Expr::VectorOp(op) => self.vector_op_to_cypher(op),
            _ => "...".to_string()
        }
    }

    fn value_to_cypher(&self, value: &PropertyValue) -> String {
        match value {
            PropertyValue::Null => "null".to_string(),
            PropertyValue::Bool(b) => if *b { "true" } else { "false" }.to_string(),
            PropertyValue::Int(i) => i.to_string(),
            PropertyValue::Float(f) => f.to_string(),
            PropertyValue::String(s) => format!("'{}'", s.replace('\'', "\\'")),
            PropertyValue::Parameter(name) => format!("${}", name),
            _ => "null".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cypher_transpiler() {
        let transpiler = CypherTranspiler::new();

        let ast = QueryAst {
            query_type: QueryType::Match,
            matches: vec![],
            where_clause: None,
            returns: vec![],
            limit: Some(10),
            order_by: None,
            vector_ops: vec![],
        };

        let sql = transpiler.to_sql(&ast);
        assert!(sql.contains("SELECT"));
        assert!(sql.contains("LIMIT 10"));
    }

    #[test]
    fn test_vector_op_to_sql() {
        let transpiler = CypherTranspiler::new();

        let op = VectorOp::Hamming {
            a: Expr::Variable("a".to_string()),
            b: Expr::Variable("b".to_string()),
        };

        let sql = transpiler.vector_op_to_sql(&op);
        assert!(sql.contains("hamming_distance"));
    }

    #[test]
    fn test_gql_transpiler() {
        let transpiler = GqlTranspiler::new();

        let ast = QueryAst {
            query_type: QueryType::VectorSearch,
            matches: vec![],
            where_clause: None,
            returns: vec![],
            limit: Some(5),
            order_by: None,
            vector_ops: vec![],
        };

        let cypher = transpiler.to_cypher(&ast);
        assert!(cypher.contains("hdr.search"));
        assert!(cypher.contains("LIMIT 5"));
    }
}
