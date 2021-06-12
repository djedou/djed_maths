use std::{
    ops::{Add, Sub, Mul}, 
    iter::Sum
};
use num::{One, Zero, cast, NumCast};
use std::fmt::Debug;
use crate::linear_algebra::matrix::{Matrix};
use std::ops::FnMut;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};


pub type CallBack<T,P> = Arc<Mutex<dyn FnMut(P) -> T + Send + Sync>>;
pub type ZipCallBack<T,P> = Arc<Mutex<dyn FnMut(P,P) -> T + Send + Sync>>;

/// A Vector here is seen as a row matrix or row vector, so size of 1 x n.
#[derive(Debug, Clone)]
pub struct Vector<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T: Debug + Clone + Default + Send + Sync> Vector<T> {

    /// new Empty Vector 
    pub fn new() -> Vector<T> {
        let data: Vec<T> = Vec::new();

        Vector {
            rows: 0,
            cols: 0,
            data
        }
    }

    /// new Vector from Vec
    pub fn new_from_vec(value: &[T]) -> Vector<T> {

        Vector {
            rows: 1,
            cols: value.len(),
            data: value.to_vec()
        }
    }
    
    /// new Vector fill with zeros or default T type
    pub fn new_with_zeros(cols: usize) -> Vector<T> {

        let data: Vec<T> = (0..cols).into_par_iter().map(|_| T::default()).collect();

        Vector {
            rows: 1,
            cols: data.len(),
            data: data.to_vec()
        }
    }

    /// new Vector fill with a function
    pub fn new_from_fn(cols: usize, f: CallBack<T,usize>) -> Vector<T> {
        let mut new_vector: Vector<T> = Vector::new_with_zeros(cols);

        let data: Vec<T> = (0..cols).into_par_iter()
                            .map(|x| {
                                let mut cb = f.lock().unwrap();
                                cb(x)
                            })
                            .collect();

        new_vector.data = data;

        new_vector
    }

    pub fn apply(&self, f: CallBack<T, T>) -> Vector<T> {
        let data = self.get_data().into_par_iter()
                                .clone()
                                .map(|x| {
                                    let mut cb = f.lock().unwrap();
                                    cb(x)
                                })
                                .collect();
        
        Vector {
            rows: self.nrows(),
            cols: self.ncols(),
            data
        }
    }

    pub fn zip_apply(&self, second: &Vector<T>, f: ZipCallBack<T, T>) -> Result<Vector<T>, String> {
        if self.nrows() == second.nrows() && self.ncols() == second.ncols() {
            let data = self.get_data().into_par_iter()
                                .clone()
                                .zip(second.get_data())
                                .map(|(a,b)| {
                                    let mut cb = f.lock().unwrap();
                                    cb(a,b)
                                })
                                .collect();
        
            Ok(Vector {
                rows: self.nrows(),
                cols: self.ncols(),
                data
            })
        }
        else {
            Err("self and rhs should have same size".to_owned())
        }
        
    }

    /// Get Data of the Vector
    pub fn get_data(&self) -> Vec<T> {
        self.data.clone()
    }

    /// Get the numbers of rows, the size
    pub fn nrows(&self) -> usize {
        self.rows
    }

    /// Get the numbers of cols, the size
    pub fn ncols(&self) -> usize {
        self.cols
    }

    /// print the Vector into console
    pub fn view(&self) {
        println!("rows: {}", self.rows);
        println!("cols: {}", self.cols);
        println!("data: {:?}", self.data);
    }

    /// cast a Vector into a Matrix
    pub fn into_matrix(&self, cols: usize) -> Matrix<T> {
        Matrix::new_from_vec(cols, &self.data)
    }

    /// cast a Matrix into Vector
    pub fn from_matrix(matrix: Matrix<T>) -> Vector<T> {

        let vector = Arc::new(Mutex::new(Vec::new()));
        matrix.get_data().par_iter().for_each(|v| {
            vector.lock().unwrap().extend_from_slice(v.get_data().as_slice())
        });
        
        let res = vector.lock().unwrap().clone();
        Vector::new_from_vec(&res)

    }
    
}


impl<T: Debug + Clone + Copy + One + Zero + Default + NumCast + Add<T, Output = T> + Mul<T, Output = T> + Sub<T, Output = T> + Send + Sync + Sum<T>> Vector<T> {
    /// the Norm or Magnitude ||v|| of a vector
    pub fn norm(&self) -> f64 {
        let sum = self.get_data()
                    .par_iter().
                    fold(|| T::default(), |acc, d| acc + ((*d) * (*d)))
                    .sum::<T>();

        cast::<T, f64>(sum).unwrap().sqrt()
    }

    /// check if norm = 1.0
    pub fn is_unit_vector(&self) -> bool {
        self.norm() == 1.0
    }

    /// multiply a Vector by a scalar
    pub fn mul_by_scalar(&self, scalar: T) -> Vector<T> {
        let data: Vec<T> = self.data
            .par_iter()
            .cloned()
            .map(|a| a * scalar)
            .collect();
    
        Vector {
            rows: 1,
            cols: data.len(),
            data: data
        }

    }

    /// add a scalar to a Vector
    pub fn add_scalar(&self, scalar: T) -> Vector<T> {
        let data: Vec<T> = self.data
            .par_iter()
            .map(|a| *a + scalar)
            .collect();
    
            Vector {
                rows: 1,
                cols: data.len(),
                data: data
            }
    }
  
    /// get the unit vector of a Vector, the normalized vector
    pub fn get_unit_vector(&self) -> Vector<T> {

        let norm_inverse: T = cast(1.0 / (self.norm())).unwrap();
        
        self.mul_by_scalar(norm_inverse)
        
    }

    /// add a vector to other vector
    pub fn add_vector(&self, rhs: &Vector<T>) -> Vector<T> {
        let data: Vec<T> = self.data
                        .par_iter()
                        .zip(
                            rhs.data
                                        .par_iter()
                        )
                        .map(|(a, b)| *a + *b).collect();

        Vector {
            rows: 1,
            cols: data.len(),
            data: data
        }
    }

    /// subtract a vector from other vector
    pub fn sub_vector(&self, rhs: &Vector<T>) -> Vector<T> {
        let data: Vec<T> = self.data
                                .par_iter()
                                .zip(rhs.data
                                                .par_iter()
                                )
                                .map(|(a,b)| (*a) - (*b))
                                .collect();

        Vector {
            rows: 1,
            cols: data.len(),
            data: data
        }
    }

    pub fn push_scalar(&mut self, scalar: T) {
        self.data.push(scalar);
        self.cols = self.cols + 1;
    }

    /// a . b = T. The result is a scalar
    pub fn dot_product(&self, rhs: &Vector<T>) -> T {
        let sum: T = self.get_data()
                        .par_iter()
                        .zip(rhs.get_data()
                                        .par_iter())
                        .fold(|| T::default(), |acc,(a,b)| acc + ((*a) * (*b)))
                        .sum::<T>();
        sum
    }
}
/*

impl<T: Debug + Clone + Copy + One + Zero + Default + Send + Sync + Add<T, Output = T>> Vector<T> {
    pub fn add_matrix(&mut self, rhs: &Matrix<T>) {
        matrix_internal_op_mut(&mut self.get_data(), &rhs.get_data(), |mut x,y| {
            matrix_internal_op_mut(&mut x, &y, |x,&y| {
                *x = *x + y
            });
        });
    }
}*/
/*
impl<T: Debug + Clone + Copy + One + Zero + Sub<T, Output = T>> Matrix<T> {
    pub fn substract_matrix(&mut self, rhs: &Matrix<T>) {
        matrix_internal_op_mut(&mut self.data, &rhs.data, |mut x,y| {
            matrix_internal_op_mut(&mut x, &y, |x,&y| {
                *x = *x - y
            });
        });
    }
}
*/


#[cfg(test)]
mod vector_tests {
    use super::{Vector};
    use std::sync::{Arc, Mutex};
    use crate::linear_algebra::matrix::Matrix;
    #[test]
    fn new() {
        let a = Vector::<i32>::new_from_vec(&vec![2,-1,-7,4]);
        a.view();
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn new_with_zeros() {
        let a = Vector::<f64>::new_with_zeros(100);
        a.view();
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn new_from_fn() {
        let callback = Arc::new(Mutex::new(|x| x * 3));
        let a = Vector::new_from_fn(100, callback);
        a.view();
        assert_eq!(2 + 2, 4);
    }
    
    #[test]
    fn apply() {
        let callback = Arc::new(Mutex::new(|x| x * 3));
        let a = Vector::new_from_fn(10, callback);
        a.view();
        let callback2 = Arc::new(Mutex::new(|x| x * 5));
        let b = a.apply(callback2);
        b.view();
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn zip_apply() {
        let callback = Arc::new(Mutex::new(|x| x * 3));
        let a = Vector::new_from_fn(10, callback);
        a.view();
        let callback2 = Arc::new(Mutex::new(|x| x * 5));
        let b = a.apply(callback2);
        b.view();
        let zipcallback = Arc::new(Mutex::new(|a,b| a + b));
        let c = a.zip_apply(&b, zipcallback).unwrap();
        c.view();
        assert_eq!(2 + 2, 4);
    }
  /*
    #[test]
    fn into_matrix() {
        let a = Vector::<i32>::new_from_vec(&vec![2,-1,-7,4]);
        a.view();
        let mat = a.into_matrix(2);
        mat.view();
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn from_matrix() {
        let mat = Matrix::<i32>::new_from_vec(2,&vec![2,-1,-7,4]);
        mat.view();
        let vector = Vector::from_matrix(mat);
        vector.view();
        assert_eq!(2 + 2, 4);
    }
*/
    #[test]
    fn norm() {
        let a = Vector::<i32>::new_from_vec(&vec![2,-2,3,-4]);
        a.view();
        let norm = a.norm();
        println!("norm: {}", norm);
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn mul_by_scalar() {
        let a = Vector::<i32>::new_from_vec(&vec![2,-1,-7,4]);
        a.view();
        let vector2 = a.mul_by_scalar(2);
        vector2.view();
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn add_scalar() {
        let a = Vector::<i32>::new_from_vec(&vec![2,-1,-7,4]);
        a.view();
        let vector = a.add_scalar(2);
        vector.view();
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn get_unit_vector() {
        let a = Vector::<f64>::new_from_vec(&vec![2.0,-1.0,-7.0,4.0]);
        a.view();
        let unit_vector = a.get_unit_vector();
        unit_vector.view();
        assert_eq!(2 + 2, 4);
    }
    #[test]
    fn add_vector() {
        let a = Vector::<f64>::new_from_vec(&vec![2.0,-1.0,-7.0,4.0]);
        let b = Vector::<f64>::new_from_vec(&vec![2.0,-1.0,-7.0,4.0]);
        a.view();
        b.view();
        let vector = a.add_vector(&b);
        vector.view();
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn sub_vector() {
        let a = Vector::<f64>::new_from_vec(&vec![2.0,-1.0,-7.0,4.0]);
        let b = Vector::<f64>::new_from_vec(&vec![2.0,-1.0,-7.0,4.0]);
        a.view();
        b.view();
        let vector = a.sub_vector(&b);
        vector.view();
        assert_eq!(2 + 2, 4);
    }
    
    #[test]
    fn dot_product() {
        let a = Vector::<i32>::new_from_vec(&vec![1,2,3]);
        let b = Vector::<i32>::new_from_vec(&vec![4,-5,6]);
        a.view();
        b.view();
        let dot_product = a.dot_product(&b);
        println!("dot_product: {:?}", dot_product);
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn push_scalar() {
        let mut a = Vector::<i32>::new_from_vec(&vec![1,2,3]);
        a.view();
        a.push_scalar(5);
        a.view();
        assert_eq!(2 + 2, 4);
    }
/*
    #[test]
    fn add() {
        let mut a = Matrix::<i32>::new_from_vec(2,vec![2,-1,-7,4]);
        let b = Matrix::<i32>::new_from_vec(2,vec![-3,0,7,-4]);
        a.add_matrix(&b);
        a.view();
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn sub() {
        let mut a = Matrix::<i32>::new_from_vec(2,vec![2,6,4,8]);
        let b = Matrix::<i32>::new_from_vec(2,vec![3,-5,-7,9]);
        a.substract_matrix(&b);
        println!("{:?}", a);
        assert_eq!(2 + 2, 4);
    }

    */
}
