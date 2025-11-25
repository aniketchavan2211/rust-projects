use std::f64;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x,y)| x * y).sum()
}

fn add_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x,y)| x + y).collect()
}

fn sub_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x,y)| x - y).collect()
}

fn scalar_mul(a: &[f64], s: f64) -> Vec<f64> {
    a.iter().map(|x| x * s).collect()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn add_matrix(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    a.iter().zip(b.iter())
        .map(|(ra, rb)| ra.iter().zip(rb.iter())
        .map(|(x, y)| x + y).collect())
        .collect()
}

fn transpose(a: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = a[0].len();

    let mut t = vec![vec![0.0; rows]; cols];

    for i in 0..rows {
        for j in 0..cols {
            t[j][i] = a[i][j];
        }
    }
    t
}

fn matmul(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = b[0].len();
    let mut result = vec![vec![0.0; cols]; rows];

    for i in 0..rows {
        for j in 0..cols {
            for k in 0..b.len() {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

fn mat_vec_mul(a: &Vec<Vec<f64>>, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.len()];

    for i in 0..a.len() {
        for j in 0..a[0].len() {
            out[i] += a[i][j] * v[j];
        }
    }
    out
}

/// -----------------------------
/// Added: Cosine similarity
/// -----------------------------
/// Cosine similarity = (A · B) / (|A| * |B|)
/// returns a value in [-1, 1]. If either vector has zero magnitude, returns 0.0 to avoid div-by-zero.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");

    let dotp = dot(a, b);
    let na = norm(a);
    let nb = norm(b);

    if na == 0.0 || nb == 0.0 {
        // undefined mathematically; return 0.0 as neutral similarity
        return 0.0;
    }

    dotp / (na * nb)
}

/// helper showing the same computation using precomputed dot and norms
fn cosine_from_dot_norms(dotp: f64, na: f64, nb: f64) -> f64 {
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dotp / (na * nb)
    }
}

/// -----------------------------
/// Added: determinant for 2x2 matrix
/// matrix is represented as Vec<Vec<f64>> with size 2x2
/// | a  b |
/// | c  d |
/// det = a*d - b*c
fn determinant_2x2(m: &Vec<Vec<f64>>) -> f64 {
    assert!(m.len() == 2 && m[0].len() == 2 && m[1].len() == 2, "Matrix must be 2x2");
    m[0][0] * m[1][1] - m[0][1] * m[1][0]
}

fn main() {
    // vectors
    let v1 = vec![1.0, 2.0, 3.0];
    let v2 = vec![4.0, 5.0, 6.0];

    // matrices
    let m1 = vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0]
    ];

    let m2 = vec![
        vec![5.0, 6.0],
        vec![7.0, 8.0]
    ];

    println!("--- Vector Operations ---");
    println!("v1 = {:?}", v1);
    println!("v2 = {:?}", v2);
    println!("dot(v1, v2) = {}", dot(&v1, &v2));
    println!("v1 + v2 = {:?}", add_vec(&v1, &v2));
    println!("v1 - v2 = {:?}", sub_vec(&v1, &v2));
    println!("2 * v1 = {:?}", scalar_mul(&v1, 2.0));
    println!("||v1|| = {}", norm(&v1));

    // cosine similarity: both direct and via helper
    println!("\n--- Cosine Similarity ---");
    let cos = cosine_similarity(&v1, &v2);
    println!("cosine_similarity(v1, v2) = {}", cos);

    let pre_dot = dot(&v1, &v2);
    let pre_na = norm(&v1);
    let pre_nb = norm(&v2);
    println!("cosine_from_dot_norms(...) = {}", cosine_from_dot_norms(pre_dot, pre_na, pre_nb));


    println!("\n--- Matrix Operations ---");
    println!("m1 = {:?}", m1);
    println!("m2 = {:?}", m2);

    println!("m1 + m2 = {:?}", add_matrix(&m1, &m2));
    println!("transpose(m1) = {:?}", transpose(&m1));
    println!("m1 × m2 = {:?}", matmul(&m1, &m2));

    println!("m1 × v1(only first 2 values) = {:?}", mat_vec_mul(&m1, &v1[..2]));

    // Determinant 2x2    
    println!("\n--- 2x2 Determinant ---");
    println!("determinant(m1) = {}", determinant_2x2(&m1));
    println!("determinant(m2) = {}", determinant_2x2(&m2));
}
