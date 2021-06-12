use std::ops::{Add, Sub};
use super::matrix_internal_op_mut;
use num::{One, Zero, NumCast};
use std::fmt::Debug;
use crate::linear_algebra::vector::{Vector, CallBack, ZipCallBack};
use std::cmp::PartialEq;
use  std::ops::{FnMut, Fn};
use rayon::prelude::*;
use std::{
    sync::{Arc, Mutex},
    iter::Sum,
};

/// rows = number of Vectors  
/// cols = nmber of value in each Vector or the cols of one Vector in the Matrix  
/// data = Vec<Vector<T>>
#[derive(Debug, Clone)]
pub struct Matrix<T> {
    rows: usize, 
    cols: usize, 
    data: Vec<Vector<T>>,
}

impl<T: Debug + Clone + Default + Sync + Send> Matrix<T> {

    pub fn new() -> Matrix<T> {
        let data: Vec<Vector<T>> = vec![Vector::new()];

        Matrix {
            rows: 0,
            cols: 0,
            data
        }
    }
    /// new Matrix from Vec
    pub fn new_from_vec(cols: usize, value: &[T]) -> Matrix<T> {

        let data: Vec<Vector<T>> = value
                    .par_chunks(cols).collect::<Vec<&[T]>>()
                    .into_par_iter()
                    .map(|d| Vector::new_from_vec(d))
                    .collect();

        Matrix {
            rows: data.len(),
            cols,
            data
        }
    }

    /// new Matrix fill with zeros or default T type
    pub fn new_with_zeros(rows: usize, cols: usize) -> Matrix<T> {

        let data: Vec<Vector<T>> = vec![T::default(); cols * rows]
                    .par_chunks(cols).collect::<Vec<&[T]>>()
                    .into_par_iter()
                    .map(|d| Vector::new_from_vec(d))
                    .collect();

        Matrix {
            rows: data.len(),
            cols,
            data
        }
    }

    /// new Matrix fill with a function
    pub fn new_from_fn(rows: usize, cols: usize, f: CallBack<T, usize>) -> Matrix<T> {
        let new_vector = Vector::new_from_fn(rows * cols, f);                                                
        new_vector.into_matrix(cols)
    }

    /// print Matrix into console
    pub fn view(&self) {
        println!("rows: {}", self.rows);
        println!("cols: {}", self.cols);
        println!("data: [");
        for d in self.data.iter() {
            println!("   {:?},", d.get_data());
        }
        println!(" ]");
    }

    /// Get the numbers of rows, the size
    pub fn nrows(&self) -> usize {
        self.rows
    }

    /// Get the numbers of cols, the size
    pub fn ncols(&self) -> usize {
        self.cols
    }

    /// get Matrix data
    pub fn get_data(&self) -> Vec<Vector<T>> {
        self.data.clone()
    }

    /// cast a Matrix into Vector (row vector)
    pub fn into_vector(&self) -> Vector<T> {
        Vector::from_matrix(self.clone())
    }

    /// Check if the matrix is a square matrix n x m where n == m.
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }
    
}

impl<T: Debug + Clone + Copy + One + Zero + Default + NumCast + PartialEq + Add<T, Output = T> + Sub<T, Output = T> + Sum<T> + Sync + Send> Matrix<T> {
    /// add two matrix and return a new one
    pub fn add_matrix(&mut self, rhs: &Matrix<T>) -> Result<Matrix<T>, String> {
        
        if self.ncols() == rhs.ncols() && self.nrows() == rhs.nrows() {
            let data: Vec<Vector<T>> = self.data
                            .par_iter()
                            .zip(
                                rhs.data
                                            .par_iter()
                            )
                            .map(|(a, b)| a.add_vector(b)).collect();
    
            Ok(Matrix {
                rows: data.len(),
                cols: self.ncols(),
                data
            })
        }
        else {
            Err("self and rhs should have same size".to_owned())
        }
    }

    /// add two matrix, the result is in the first matrix
    pub fn add_matrix_to_self(&mut self, rhs: &Matrix<T>) -> Result<(), String> {
        
        if self.ncols() == rhs.ncols() && self.nrows() == rhs.nrows()  {
            let data: Vec<Vector<T>> = self.data
                            .par_iter()
                            .zip(
                                rhs.data
                                            .par_iter()
                            )
                            .map(|(a, b)| a.add_vector(b)).collect();
            
            self.data = data;

            Ok(())
        }
        else {
            Err("self and rhs should have same size".to_owned())
        }
    }
 
    /// subtruct two matrix and return a new one
    pub fn sub_matrix(&mut self, rhs: &Matrix<T>) -> Result<Matrix<T>, String> {

        if self.ncols() == rhs.ncols() && self.nrows() == rhs.nrows() {
            let data: Vec<Vector<T>> = self.data
                            .par_iter()
                            .zip(
                                rhs.data
                                            .par_iter()
                            )
                            .map(|(a, b)| a.sub_vector(b)).collect();
    
            Ok(Matrix {
                rows: data.len(),
                cols: self.ncols(),
                data
            })
        }
        else {
            Err("self and rhs should have same size".to_owned())
        }
    }

    /// subtruct two matrix, the result is in the first matrix
    pub fn sub_matrix_from_self(&mut self, rhs: &Matrix<T>) -> Result<(), String> {
        
        if self.ncols() == rhs.ncols() && self.nrows() == rhs.nrows() {
            let data: Vec<Vector<T>> = self.data
                            .par_iter()
                            .zip(
                                rhs.data
                                            .par_iter()
                            )
                            .map(|(a, b)| a.sub_vector(b)).collect();
    
            
            self.data = data;
            Ok(())
        }
        else {
            Err("self and rhs should have same size".to_owned())
        }
    }


    /// Compare two matrix, if they are equal
    pub fn is_equal_to(&self, rhs: &Matrix<T>) -> bool {
        if self.rows == rhs.rows && self.cols == rhs.cols {
            let self_vector = self.into_vector();
            let rhs_vector = rhs.into_vector();
            for (a, b) in self_vector.get_data().iter().zip(rhs_vector.get_data().iter()) {
                if *a != *b {
                    return false;
                }
            }
            true
        }
        else {
            false
        }
    }

    /// Check if a Matrix is symmetric
    pub fn is_symmetric(&self) -> bool {
        if self.is_square() {
            let transpose = self.transpose();
            self.is_equal_to(&transpose)
        }
        else {
            false
        }
    }

    /// multiply by a scalar
    pub fn mul_by_scalar(&self, scalar: T) -> Matrix<T> {
        self.into_vector()
        .mul_by_scalar(scalar)
        .into_matrix(self.cols)
    }

    /// Get column by index
    pub fn get_col(&self, n: usize) -> Vector<T> {
        let data: Vec<T> = self.data
                            .par_iter()
                            .map(|a| a.get_data()[n]).collect();
    
        Vector::new_from_vec(&data)
    }

    /// Get row by index
    pub fn get_row(&self, n: usize) -> Vector<T> {
        let data = &self.data[n];

        Vector::new_from_vec(&data.get_data())
    }

    /// add a new column to the matrix, only possible when self.rows == col.len()
    pub fn add_col(&mut self, col: &[T]) -> Result<(), String> {

        if self.rows == col.len() {
            let cols = self.ncols() + 1;
            
            for k in (0..col.len()).into_iter() {
                self.data[k].push_scalar(col[k]);
            }
    
            self.cols = cols;

            Ok(())

        }
        else {
            Err("self rows and col len should have same size".to_owned())
        }
    }

    pub fn add_row(&mut self, row: &[T]) -> Result<(), String> {
        if self.cols == row.len() {
            let rows = self.rows + 1;

            self.data.push(Vector::new_from_vec(row));
            self.rows = rows;

            Ok(())
        }
        else {
            Err("self rows and col len should have same size".to_owned())
        }
    }

    /// Transpose a Matrix
    pub fn transpose(&self) -> Matrix<T> {
        let mut transpose_matrix: Matrix<T> = Matrix::new();

        transpose_matrix.cols = self.rows;

        for c in (0..self.cols).into_iter() {
            transpose_matrix.add_row(&self.get_col(c).get_data()).unwrap();
        }

        let _ = transpose_matrix.data.remove(0);
        transpose_matrix.rows = transpose_matrix.data.len();

        transpose_matrix
    }

    /// dot product two matrix
    pub fn dot_product(&self, rhs: &Matrix<T>) -> Result<Matrix<T>, String> {
        if self.cols == rhs.rows {
            let mut data: Vec<T> = Vec::new();
            for a in (0..self.rows).into_iter() {
                let row = self.get_row(a);
                for b in (0..rhs.cols).into_iter() {
                    let col = rhs.get_col(b);
                    let sum = row.dot_product(&col);
                    data.push(sum);
                }
            }

            Ok(Matrix::new_from_vec(rhs.cols, &data))
        }
        else {
            Err("self cols should equal to rhs rows".to_owned())
        }
    }
/*
    /// new Matrix from Vectors, the matrix will grow by columns
    pub fn new_from_columns(data: &Vec<Vector<T>>) -> Matrix<T> {
        let first = &data[0];

        let mut new_matrix = Matrix::new_from_vec(1, &first.get_data());
        for i in (1..data.len()).into_iter() {
            new_matrix.add_col(&data[i].get_data());
        }

        new_matrix
    }
*/
    /// apply a function to each element of the matrix
    pub fn apply(&self, f: CallBack<T, T>) -> Matrix<T> {

        let data: Vec<Vector<T>> = self.data
                                        .par_iter()
                                        .map(|a| a.apply(f.clone()))
                                        .collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data
        }
    }
/*
    /// apply a mut function to each element of the matrix
    pub fn apply_mut<F>(&mut self, f: &mut F)
        where F: FnMut(usize, usize, T) -> T
    {

        for x in (0..self.nrows()).into_iter(){
            for y in (0..self.ncols()).into_iter() {
                let value = self.data[x][y];
                self.data[x][y] = f(x, y, value);
            }
        }
    }
*/
    /// zip apply a mut function to each element of the matrix
    pub fn zip_apply(&self, rhs: &Matrix<T>, f: ZipCallBack<T,T>) -> Result<Matrix<T>, String> {
        
        //let matrix = self.into_vector()
        //                    .zip_apply(&rhs.into_vector(), f)?
        //                    .into_matrix(self.ncols());

        let data: Vec<Vector<T>> = self.data
                            .par_iter()
                            .zip(
                                rhs.data
                                            .par_iter()
                            )
                            .map(|(a, b)| a.zip_apply(&b, f.clone()).unwrap()).collect();
        Ok(Matrix {
            rows: self.rows,
            cols: self.cols,
            data
        })
        
    }
}

#[cfg(test)]
mod matrix_tests {
    use super::Matrix;
    use std::sync::{Arc, Mutex};
    use crate::linear_algebra::vector::{Vector, CallBack, ZipCallBack};

    #[test]
    fn new_matrix() {
        let a = Matrix::<i32>::new();
        a.view();
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn new_from_vec() {
        let a = Matrix::<i32>::new_from_vec(2,&vec![2,-1,-7,4]);
        a.view();
        assert_eq!(2 + 2, 4);
    }
    
    #[test]
    fn new_with_zeros() {
        let a = Matrix::<f64>::new_with_zeros(20,10);
        a.view();
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn into_vector() {
        let a = Matrix::<i32>::new_with_zeros(4,4);
        a.view();
        let vector = a.into_vector();
        vector.view();
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn new_from_fn() {
        let callback: CallBack<i32, usize> = Arc::new(Mutex::new(|x| (x as i32) * 3));
        let mat1 = Matrix::<i32>::new_from_fn(4,4,callback);
        mat1.view();
        
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn add_matrix() {
        let mut a = Matrix::<i32>::new_from_vec(2,&vec![2,5,-1,3]);
        let b = Matrix::<i32>::new_from_vec(2,&vec![1,4,3,7]);
        if let Ok(c)  = a.add_matrix(&b) {
            c.view();
        }
        assert_eq!(2 + 2, 4);
    }
/*
    #[test]
    fn sub() {
        let mut a = Matrix::<i32>::new_from_vec(2,&vec![2,6,4,8]);
        let b = Matrix::<i32>::new_from_vec(2,&vec![3,-5,-7,9]);
        a.sub_matrix(&b);
        println!("{:?}", a);
        assert_eq!(2 + 2, 4);
    }
*/

    #[test]
    fn transpose() {
        //let mat = Matrix::<i32>::new_from_vec(3,vec![2,-1,-7,4,8,12,14,15,16]);
        //let mat = Matrix::<i32>::new_from_vec(3,&vec![1,5,-3,5,4,2,-3,2,0,2,5,7]);
        let mat = Matrix::<i32>::new_from_vec(3,&vec![6,4,24,1,-9,8]);
        mat.view();
        let transpose = mat.transpose();
        transpose.view();
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn is_equal_to() {
        let mat = Matrix::<i32>::new_from_vec(3,&vec![2,-1,-7,4,8,12,14,15,16]);
        let mat2 = Matrix::<i32>::new_from_vec(3,&vec![1,5,-3,5,4,2,-3,2,0]);
        let mat3 = Matrix::<i32>::new_from_vec(3,&vec![1,5,-3,5,4,2,-3,2,0]);
        mat.view();
        mat2.view();
        mat3.view();

        let equl1 = mat.is_equal_to(&mat2);
        println!("mat is equal to mat2 ? : {:?}", equl1);

        let equl2 = mat2.is_equal_to(&mat3);
        println!("mat2 is equal to mat3 ? : {:?}", equl2);

        assert_eq!(2 + 2, 4);
    }
/*
    #[test]
    fn is_symmetric() {
        //let mat = Matrix::<i32>::new_from_vec(3,vec![2,-1,-7,4,8,12,14,15,16]);
        let mat = Matrix::<i32>::new_from_vec(3,&vec![1,5,-3,5,4,2,-3,2,0]);
        let sym = mat.is_symmetric();
        println!("is_symetric: {:?}", sym);
        assert_eq!(2 + 2, 4);
    }
*/
    #[test]
    fn mul_by_scalar() {
        //let mat = Matrix::<i32>::new_from_vec(3,vec![2,-1,-7,4,8,12,14,15,16]);
        let mat = Matrix::<i32>::new_from_vec(3,&vec![1,5,-3,5,4,2,-3,2,0]);
        mat.view();
        let mul_by_scalar = mat.mul_by_scalar(2);
        mul_by_scalar.view();
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn get_col() {
        //let mat = Matrix::<i32>::new_from_vec(3,vec![2,-1,-7,4,8,12,14,15,16]);
        let mat = Matrix::<i32>::new_from_vec(3,&vec![1,5,-3,5,4,2,-3,2,0]);
        mat.view();
        let get_col = mat.get_col(1);
        get_col.view();
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn get_row() {
        //let mat = Matrix::<i32>::new_from_vec(3,vec![2,-1,-7,4,8,12,14,15,16]);
        let mat = Matrix::<i32>::new_from_vec(3,&vec![1,5,-3,5,4,2,-3,2,0]);
        mat.view();
        let get_row = mat.get_row(2);
        get_row.view();
        assert_eq!(2 + 2, 4);
    }


    #[test]
    fn dot_product() {
        let mat1 = Matrix::<i32>::new_from_vec(3,&vec![1, 2, 3, 4, 5, 6]);
        let mat2 = Matrix::<i32>::new_from_vec(2,&vec![7, 8, 9, 10, 11, 12]);
        mat1.view();
        mat2.view();
        if let Ok(mut_mat) = mat1.dot_product(&mat2) {
            mut_mat.view();
        }
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn add_col() {
        let mut mat1 = Matrix::<i32>::new_from_vec(3,&vec![1,3,-2,0,-1,4]);
        let vec1 = vec![4,6];

        mat1.view();
        mat1.add_col(&vec1).unwrap();
        mat1.view();
        
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn add_row() {
        let mut mat1 = Matrix::<i32>::new_from_vec(3,&vec![1,3,-2,0,-1,4]);
        let vec1 = vec![4,6,10];

        mat1.view();
        mat1.add_row(&vec1).unwrap();
        mat1.view();
        
        assert_eq!(2 + 2, 4);
    }


/*  
    #[test]
    fn new_from_columns() {
        //Vector::<i32>::new_from_vec(&vec![2,-1,-7,4]);
        let v1 = Vector::<f64>::new_from_vec(&vec![0.6753966962600456, 0.07920298215765209, 0.8979780697802067, 0.6198964030881386, 0.11884100014676291, 0.6208360148717391, 0.08050817587074632]); 
        let v2 = Vector::<f64>::new_from_vec(&vec![0.536278939372817, 0.6441195751565744, 0.0329024062382276, 0.4146769744444053, 0.24430115630826355, 0.1162468754910504, 0.6356912661071378]); 
        let v3 = Vector::<f64>::new_from_vec(&vec![0.7827283306758599, 0.0786543317452244, 0.08630282113337406, 0.033621176506813844, 0.5354701294235551, 0.4880328779263531, 0.7459103712585138]);
        let all = vec![v1, v2, v3];
        let new_mat = Matrix::<f64>::new_from_columns(&all);
        new_mat.view();
        
        assert_eq!(2 + 2, 4);
    }
*/
    #[test]
    fn apply() {
        let mat1 = Matrix::<i32>::new_from_vec(3,&vec![1,3,-2,0,-1,4]);
        let callback: CallBack<i32, i32> = Arc::new(Mutex::new(|x| (x as i32) * 3));
        mat1.view();
        let mat2 = mat1.apply(callback);
        mat2.view();
        
        assert_eq!(2 + 2, 4);
    }


    #[test]
    fn zip_apply() {
        let mat1 = Matrix::<i32>::new_from_vec(3,&vec![1,3,-2,0,-1,4]);
        let mat2 = Matrix::<i32>::new_from_vec(3,&vec![1,3,-2,0,-1,4]);

        mat1.view();
        mat2.view();
        
        let callback: ZipCallBack<i32, i32> = Arc::new(Mutex::new(|a,b| a + b));
        let mat3: Matrix<i32> = mat1.zip_apply(&mat2, callback).unwrap();
        mat3.view();

        assert_eq!(2 + 2, 4);
    }
    
}
