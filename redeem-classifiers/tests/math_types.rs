//! Integration tests for the custom Array1 and Array2 math types.

use redeem_classifiers::math::{Array1, Array2};

// ---------------------------------------------------------------------------
// Array1 basics
// ---------------------------------------------------------------------------

#[test]
fn array1_from_vec_and_len() {
    let a = Array1::from_vec(vec![1.0f32, 2.0, 3.0]);
    assert_eq!(a.len(), 3);
    assert!(!a.is_empty());
}

#[test]
fn array1_empty() {
    let a: Array1<f32> = Array1::from_vec(vec![]);
    assert!(a.is_empty());
    assert_eq!(a.len(), 0);
}

#[test]
fn array1_from_elem() {
    let a = Array1::from_elem(5, 42i32);
    assert_eq!(a.len(), 5);
    for v in a.iter() {
        assert_eq!(*v, 42);
    }
}

#[test]
fn array1_zeros() {
    let a: Array1<f32> = Array1::zeros(4);
    assert_eq!(a.len(), 4);
    for v in a.iter() {
        assert_eq!(*v, 0.0);
    }
}

#[test]
fn array1_indexing() {
    let a = Array1::from_vec(vec![10, 20, 30]);
    assert_eq!(a[0], 10);
    assert_eq!(a[1], 20);
    assert_eq!(a[2], 30);
}

#[test]
fn array1_select() {
    let a = Array1::from_vec(vec![10, 20, 30, 40, 50]);
    let selected = a.select(&[0, 2, 4]);
    assert_eq!(selected.to_vec(), vec![10, 30, 50]);
}

#[test]
fn array1_mapv() {
    let a = Array1::from_vec(vec![1.0f32, 2.0, 3.0]);
    let doubled = a.mapv(|x| x * 2.0);
    assert_eq!(doubled.to_vec(), vec![2.0, 4.0, 6.0]);
}

#[test]
fn array1_shape() {
    let a = Array1::from_vec(vec![1, 2, 3]);
    assert_eq!(a.shape(), (3,));
}

// ---------------------------------------------------------------------------
// Array2 basics
// ---------------------------------------------------------------------------

#[test]
fn array2_from_shape_vec() {
    let a = Array2::from_shape_vec((2, 3), vec![1, 2, 3, 4, 5, 6]).unwrap();
    assert_eq!(a.nrows(), 2);
    assert_eq!(a.ncols(), 3);
    assert_eq!(a.shape(), (2, 3));
}

#[test]
fn array2_shape_mismatch_errors() {
    let result = Array2::<f32>::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0]);
    assert!(result.is_err());
}

#[test]
fn array2_indexing() {
    let a = Array2::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap();
    assert_eq!(a[(0, 0)], 1);
    assert_eq!(a[(0, 1)], 2);
    assert_eq!(a[(1, 0)], 3);
    assert_eq!(a[(1, 1)], 4);
}

#[test]
fn array2_row_slice() {
    let a = Array2::from_shape_vec((2, 3), vec![1, 2, 3, 4, 5, 6]).unwrap();
    assert_eq!(a.row_slice(0), &[1, 2, 3]);
    assert_eq!(a.row_slice(1), &[4, 5, 6]);
}

#[test]
fn array2_column() {
    let a = Array2::from_shape_vec((3, 2), vec![1, 2, 3, 4, 5, 6]).unwrap();
    let col0 = a.column(0);
    assert_eq!(col0.to_vec(), vec![1, 3, 5]);
    let col1 = a.column(1);
    assert_eq!(col1.to_vec(), vec![2, 4, 6]);
}

#[test]
fn array2_select_rows() {
    let a = Array2::from_shape_vec((4, 2), vec![1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    let selected = a.select_rows(&[0, 3]);
    assert_eq!(selected.nrows(), 2);
    assert_eq!(selected.row_slice(0), &[1, 2]);
    assert_eq!(selected.row_slice(1), &[7, 8]);
}

#[test]
fn array2_select_columns() {
    let a = Array2::from_shape_vec((2, 4), vec![1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    let sub = a.select_columns(1..3);
    assert_eq!(sub.shape(), (2, 2));
    assert_eq!(sub.row_slice(0), &[2, 3]);
    assert_eq!(sub.row_slice(1), &[6, 7]);
}

#[test]
fn array2_mapv() {
    let a = Array2::from_shape_vec((2, 2), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let neg = a.mapv(|x| -x);
    assert_eq!(neg[(0, 0)], -1.0);
    assert_eq!(neg[(1, 1)], -4.0);
}
