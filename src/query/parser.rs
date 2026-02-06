//! Query Parser for Cypher and GQL Alchemy
//!
//! Parses both RedisGraph-compatible Cypher and extended GQL syntax
//! with vector operations.

use std::collections::HashMap;

/// Parsed query AST
#[derive(Debug, Clone)]
pub struct QueryAst {
    /// Query type
    pub query_type: QueryType,
    /// MATCH patterns
    pub matches: Vec<MatchClause>,
    /// WHERE predicates
    pub where_clause: Option<WhereClause>,
    /// RETURN expressions
    pub returns: Vec<ReturnExpr>,
    /// Optional LIMIT
    pub limit: Option<usize>,
    /// Optional ORDER BY
    pub order_by: Option<OrderBy>,
    /// Vector operations to apply
    pub vector_ops: Vec<VectorOp>,
}

/// Query type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    /// Standard read query
    Match,
    /// Create nodes/edges
    Create,
    /// Merge (create if not exists)
    Merge,
    /// Delete nodes/edges
    Delete,
    /// Vector similarity search
    VectorSearch,
    /// Bound retrieval (O(1) edge lookup)
    BoundRetrieval,
}

/// MATCH clause
#[derive(Debug, Clone)]
pub struct MatchClause {
    /// Node patterns in this match
    pub nodes: Vec<NodePattern>,
    /// Relationship patterns
    pub relationships: Vec<RelationPattern>,
    /// Is this an optional match?
    pub optional: bool,
}

/// Node pattern: (alias:Label {props})
#[derive(Debug, Clone)]
pub struct NodePattern {
    /// Variable name
    pub alias: Option<String>,
    /// Labels
    pub labels: Vec<String>,
    /// Property filters
    pub properties: HashMap<String, PropertyValue>,
    /// Vector binding (for HDR operations)
    pub vector_binding: Option<String>,
}

/// Relationship pattern: -[alias:TYPE {props}]->
#[derive(Debug, Clone)]
pub struct RelationPattern {
    /// Variable name
    pub alias: Option<String>,
    /// Relationship type
    pub rel_type: Option<String>,
    /// Source node alias
    pub from_node: String,
    /// Target node alias
    pub to_node: String,
    /// Direction
    pub direction: RelDirection,
    /// Property filters
    pub properties: HashMap<String, PropertyValue>,
    /// Variable length path: *min..max
    pub var_length: Option<(usize, Option<usize>)>,
}

/// Relationship direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelDirection {
    /// ->
    Outgoing,
    /// <-
    Incoming,
    /// -
    Both,
}

/// Property value
#[derive(Debug, Clone)]
pub enum PropertyValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    List(Vec<PropertyValue>),
    Map(HashMap<String, PropertyValue>),
    /// Vector literal (hex or base64)
    Vector(Vec<u8>),
    /// Parameter reference: $param
    Parameter(String),
}

/// WHERE clause
#[derive(Debug, Clone)]
pub struct WhereClause {
    pub predicates: Vec<Predicate>,
    pub logic: LogicOp,
}

/// Predicate expression
#[derive(Debug, Clone)]
pub enum Predicate {
    /// Standard comparison: a.prop = value
    Comparison {
        left: Expr,
        op: CompareOp,
        right: Expr,
    },
    /// Vector similarity: a.vec ~> b.vec < threshold
    VectorSimilarity {
        left: Expr,
        right: Expr,
        threshold: f32,
    },
    /// Resonance check: RESONANCE(vec, query) > threshold
    Resonance {
        vector: Expr,
        query: Expr,
        threshold: f32,
    },
    /// Negation
    Not(Box<Predicate>),
    /// Compound predicate
    Compound {
        left: Box<Predicate>,
        op: LogicOp,
        right: Box<Predicate>,
    },
    /// Exists subquery
    Exists(Box<QueryAst>),
}

/// Comparison operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    Eq,       // =
    Ne,       // <>
    Lt,       // <
    Le,       // <=
    Gt,       // >
    Ge,       // >=
    Contains, // CONTAINS
    StartsWith, // STARTS WITH
    EndsWith,   // ENDS WITH
    In,         // IN
}

/// Logic operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicOp {
    And,
    Or,
    Xor,
}

/// Expression
#[derive(Debug, Clone)]
pub enum Expr {
    /// Literal value
    Literal(PropertyValue),
    /// Variable reference
    Variable(String),
    /// Property access: a.prop
    Property { var: String, prop: String },
    /// Function call: func(args)
    Function { name: String, args: Vec<Expr> },
    /// Vector operation
    VectorOp(Box<VectorOp>),
    /// Arithmetic
    Arithmetic { left: Box<Expr>, op: ArithOp, right: Box<Expr> },
    /// Case expression
    Case { whens: Vec<(Predicate, Expr)>, else_expr: Option<Box<Expr>> },
}

/// Arithmetic operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
}

/// RETURN expression
#[derive(Debug, Clone)]
pub struct ReturnExpr {
    pub expr: Expr,
    pub alias: Option<String>,
}

/// ORDER BY clause
#[derive(Debug, Clone)]
pub struct OrderBy {
    pub items: Vec<(Expr, bool)>, // (expr, is_descending)
}

/// Vector operation (GQL Alchemy extensions)
#[derive(Debug, Clone)]
pub enum VectorOp {
    /// BIND(a, b) - XOR binding
    Bind { a: Expr, b: Expr },

    /// UNBIND(bound, key) - XOR unbinding (A⊗B⊗B=A)
    Unbind { bound: Expr, key: Expr },

    /// BIND3(src, verb, dst) - Triple binding for edges
    Bind3 { src: Expr, verb: Expr, dst: Expr },

    /// RESONANCE(vec, query_vec) - Find similarity
    Resonance { vector: Expr, query: Expr },

    /// CLEANUP(vec) - Map to nearest clean concept
    Cleanup { vector: Expr, memory: Option<String> },

    /// BUNDLE(vec1, vec2, ...) - Majority vote bundling
    Bundle { vectors: Vec<Expr> },

    /// HAMMING(a, b) - Compute Hamming distance
    Hamming { a: Expr, b: Expr },

    /// SIMILARITY(a, b) - Compute similarity (0-1)
    Similarity { a: Expr, b: Expr },

    /// PERMUTE(vec, n) - Rotate vector by n positions
    Permute { vector: Expr, positions: i32 },

    /// CASCADE_SEARCH(query, k, threshold) - HDR cascade search
    CascadeSearch {
        query: Expr,
        k: usize,
        threshold: Option<f32>,
    },

    /// VOYAGER(query, radius, stack_size) - Deep field search
    Voyager {
        query: Expr,
        radius: u32,
        stack_size: usize,
    },

    /// ANALOGY(a, b, c) - a:b::c:? analogy completion
    Analogy { a: Expr, b: Expr, c: Expr },
}

/// Query parser
pub struct QueryParser {
    /// Input text
    input: String,
    /// Current position
    pos: usize,
    /// Registered parameters
    parameters: HashMap<String, PropertyValue>,
}

impl QueryParser {
    /// Create new parser
    pub fn new(input: &str) -> Self {
        Self {
            input: input.to_string(),
            pos: 0,
            parameters: HashMap::new(),
        }
    }

    /// Set parameter value
    pub fn set_parameter(&mut self, name: &str, value: PropertyValue) {
        self.parameters.insert(name.to_string(), value);
    }

    /// Parse the query
    pub fn parse(&mut self) -> Result<QueryAst, ParseError> {
        self.skip_whitespace();

        let query_type = self.parse_query_type()?;

        let mut ast = QueryAst {
            query_type,
            matches: Vec::new(),
            where_clause: None,
            returns: Vec::new(),
            limit: None,
            order_by: None,
            vector_ops: Vec::new(),
        };

        // Parse clauses based on query type
        match query_type {
            QueryType::Match | QueryType::VectorSearch => {
                self.parse_match_clauses(&mut ast)?;
                self.parse_optional_where(&mut ast)?;
                self.parse_return(&mut ast)?;
                self.parse_optional_order(&mut ast)?;
                self.parse_optional_limit(&mut ast)?;
            }
            QueryType::Create => {
                self.parse_create_pattern(&mut ast)?;
                self.parse_optional_return(&mut ast)?;
            }
            QueryType::BoundRetrieval => {
                self.parse_bound_retrieval(&mut ast)?;
            }
            _ => {}
        }

        Ok(ast)
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.input.len() {
            let c = self.input.chars().nth(self.pos).unwrap();
            if c.is_whitespace() {
                self.pos += 1;
            } else if self.input[self.pos..].starts_with("//") {
                // Skip line comment
                while self.pos < self.input.len() &&
                      self.input.chars().nth(self.pos) != Some('\n') {
                    self.pos += 1;
                }
            } else if self.input[self.pos..].starts_with("/*") {
                // Skip block comment
                self.pos += 2;
                while self.pos < self.input.len() - 1 &&
                      !self.input[self.pos..].starts_with("*/") {
                    self.pos += 1;
                }
                self.pos += 2;
            } else {
                break;
            }
        }
    }

    fn peek_keyword(&self) -> Option<&str> {
        let start = self.pos;
        let mut end = start;

        while end < self.input.len() {
            let c = self.input.chars().nth(end).unwrap();
            if c.is_alphanumeric() || c == '_' {
                end += 1;
            } else {
                break;
            }
        }

        if end > start {
            Some(&self.input[start..end])
        } else {
            None
        }
    }

    fn consume_keyword(&mut self, expected: &str) -> Result<(), ParseError> {
        self.skip_whitespace();

        if let Some(kw) = self.peek_keyword() {
            if kw.eq_ignore_ascii_case(expected) {
                self.pos += expected.len();
                return Ok(());
            }
        }

        Err(ParseError::ExpectedKeyword(expected.to_string()))
    }

    fn try_consume_keyword(&mut self, expected: &str) -> bool {
        self.skip_whitespace();

        if let Some(kw) = self.peek_keyword() {
            if kw.eq_ignore_ascii_case(expected) {
                self.pos += expected.len();
                return true;
            }
        }

        false
    }

    fn parse_query_type(&mut self) -> Result<QueryType, ParseError> {
        self.skip_whitespace();

        if let Some(kw) = self.peek_keyword() {
            let kw_upper = kw.to_uppercase();
            match kw_upper.as_str() {
                "MATCH" => {
                    self.pos += 5;
                    Ok(QueryType::Match)
                }
                "CREATE" => {
                    self.pos += 6;
                    Ok(QueryType::Create)
                }
                "MERGE" => {
                    self.pos += 5;
                    Ok(QueryType::Merge)
                }
                "DELETE" => {
                    self.pos += 6;
                    Ok(QueryType::Delete)
                }
                "VECTOR" | "SEARCH" => {
                    self.pos += kw.len();
                    Ok(QueryType::VectorSearch)
                }
                "UNBIND" | "RETRIEVE" => {
                    self.pos += kw.len();
                    Ok(QueryType::BoundRetrieval)
                }
                _ => Ok(QueryType::Match) // Default
            }
        } else {
            Err(ParseError::UnexpectedEnd)
        }
    }

    fn parse_match_clauses(&mut self, ast: &mut QueryAst) -> Result<(), ParseError> {
        loop {
            self.skip_whitespace();

            let optional = self.try_consume_keyword("OPTIONAL");
            if optional {
                self.consume_keyword("MATCH")?;
            }

            // Parse node and relationship patterns
            let clause = self.parse_match_pattern(optional)?;
            ast.matches.push(clause);

            // Check for another MATCH
            self.skip_whitespace();
            if !self.try_consume_keyword("MATCH") &&
               !self.peek_keyword().map_or(false, |k| k.eq_ignore_ascii_case("OPTIONAL")) {
                break;
            }
        }

        Ok(())
    }

    fn parse_match_pattern(&mut self, optional: bool) -> Result<MatchClause, ParseError> {
        let mut clause = MatchClause {
            nodes: Vec::new(),
            relationships: Vec::new(),
            optional,
        };

        // Simplified pattern parsing
        // Full implementation would handle complex Cypher patterns
        self.skip_whitespace();

        // For now, just skip to next clause keyword
        while self.pos < self.input.len() {
            if let Some(kw) = self.peek_keyword() {
                let kw_upper = kw.to_uppercase();
                if matches!(kw_upper.as_str(),
                    "WHERE" | "RETURN" | "WITH" | "ORDER" | "LIMIT" | "MATCH" | "OPTIONAL"
                ) {
                    break;
                }
            }
            self.pos += 1;
        }

        Ok(clause)
    }

    fn parse_optional_where(&mut self, ast: &mut QueryAst) -> Result<(), ParseError> {
        if self.try_consume_keyword("WHERE") {
            // Simplified WHERE parsing
            // Skip to next clause
            while self.pos < self.input.len() {
                if let Some(kw) = self.peek_keyword() {
                    let kw_upper = kw.to_uppercase();
                    if matches!(kw_upper.as_str(), "RETURN" | "WITH" | "ORDER" | "LIMIT") {
                        break;
                    }
                }
                self.pos += 1;
            }
        }
        Ok(())
    }

    fn parse_return(&mut self, ast: &mut QueryAst) -> Result<(), ParseError> {
        self.consume_keyword("RETURN")?;

        // Simplified RETURN parsing
        while self.pos < self.input.len() {
            if let Some(kw) = self.peek_keyword() {
                let kw_upper = kw.to_uppercase();
                if matches!(kw_upper.as_str(), "ORDER" | "LIMIT") {
                    break;
                }
            }
            self.pos += 1;
        }

        Ok(())
    }

    fn parse_optional_return(&mut self, ast: &mut QueryAst) -> Result<(), ParseError> {
        if self.try_consume_keyword("RETURN") {
            while self.pos < self.input.len() {
                if let Some(kw) = self.peek_keyword() {
                    let kw_upper = kw.to_uppercase();
                    if matches!(kw_upper.as_str(), "ORDER" | "LIMIT") {
                        break;
                    }
                }
                self.pos += 1;
            }
        }
        Ok(())
    }

    fn parse_optional_order(&mut self, ast: &mut QueryAst) -> Result<(), ParseError> {
        if self.try_consume_keyword("ORDER") {
            self.consume_keyword("BY")?;
            // Skip ORDER BY clause
            while self.pos < self.input.len() {
                if let Some(kw) = self.peek_keyword() {
                    if kw.eq_ignore_ascii_case("LIMIT") {
                        break;
                    }
                }
                self.pos += 1;
            }
        }
        Ok(())
    }

    fn parse_optional_limit(&mut self, ast: &mut QueryAst) -> Result<(), ParseError> {
        if self.try_consume_keyword("LIMIT") {
            // Parse limit number
            self.skip_whitespace();
            let start = self.pos;
            while self.pos < self.input.len() &&
                  self.input.chars().nth(self.pos).unwrap().is_ascii_digit() {
                self.pos += 1;
            }
            if self.pos > start {
                let num_str = &self.input[start..self.pos];
                ast.limit = num_str.parse().ok();
            }
        }
        Ok(())
    }

    fn parse_create_pattern(&mut self, ast: &mut QueryAst) -> Result<(), ParseError> {
        // Simplified CREATE parsing
        while self.pos < self.input.len() {
            if let Some(kw) = self.peek_keyword() {
                if kw.eq_ignore_ascii_case("RETURN") {
                    break;
                }
            }
            self.pos += 1;
        }
        Ok(())
    }

    fn parse_bound_retrieval(&mut self, ast: &mut QueryAst) -> Result<(), ParseError> {
        // Parse bound retrieval query:
        // UNBIND edge USING verb, known -> result
        // or
        // RETRIEVE target FROM edge USING verb, source

        self.skip_whitespace();

        // For now, collect as vector op
        ast.query_type = QueryType::BoundRetrieval;

        // Skip to end or RETURN
        while self.pos < self.input.len() {
            if let Some(kw) = self.peek_keyword() {
                if kw.eq_ignore_ascii_case("RETURN") {
                    break;
                }
            }
            self.pos += 1;
        }

        Ok(())
    }
}

/// Parse error
#[derive(Debug, Clone)]
pub enum ParseError {
    UnexpectedEnd,
    UnexpectedToken(String),
    ExpectedKeyword(String),
    InvalidSyntax(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnexpectedEnd => write!(f, "Unexpected end of input"),
            Self::UnexpectedToken(t) => write!(f, "Unexpected token: {}", t),
            Self::ExpectedKeyword(k) => write!(f, "Expected keyword: {}", k),
            Self::InvalidSyntax(s) => write!(f, "Invalid syntax: {}", s),
        }
    }
}

impl std::error::Error for ParseError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_match() {
        let mut parser = QueryParser::new(
            "MATCH (n:Person) RETURN n"
        );
        let ast = parser.parse().unwrap();
        assert_eq!(ast.query_type, QueryType::Match);
    }

    #[test]
    fn test_vector_search() {
        let mut parser = QueryParser::new(
            "VECTOR SEARCH (n) WHERE n.embedding ~> $query < 0.3 RETURN n LIMIT 10"
        );
        let ast = parser.parse().unwrap();
        assert_eq!(ast.query_type, QueryType::VectorSearch);
        assert_eq!(ast.limit, Some(10));
    }

    #[test]
    fn test_bound_retrieval() {
        let mut parser = QueryParser::new(
            "UNBIND edge USING verb, known RETURN result"
        );
        let ast = parser.parse().unwrap();
        assert_eq!(ast.query_type, QueryType::BoundRetrieval);
    }
}
