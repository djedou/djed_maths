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

#[derive(Debug, Clone)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<Vec<T>>,
}

impl<T: Debug + Clone + Default + Sync + Send> Matrix<T> {
    /// new Matrix from Vec
    pub fn new_from_vec(cols: usize, value: &[T]) -> Matrix<T> {
        let data_slices: Vec<&[T]> = value.par_chunks(cols).collect();
        let data: Vec<Vec<T>> = data_slices.into_par_iter().map(|d| d.to_vec()).collect();

        Matrix {
            rows: data.len(),
            cols,
            data
        }
    }

    /// new Matrix fill with zeros or default T type
    pub fn new_with_zeros(rows: usize, cols: usize) -> Matrix<T> {
        let zeros = vec![T::default(); cols];
        let data: Vec<Vec<T>> = (0..rows).into_par_iter().map(|_| zeros.clone()).collect();

        Matrix {
            rows,
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
            println!("   {:?},", d);
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
    pub fn get_data(&self) -> Vec<Vec<T>> {
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
    /// add two matrix, the result is in the first matrix
    pub fn add_matrix(&mut self, rhs: &Matrix<T>) {
        matrix_internal_op_mut(&mut self.data, &rhs.data, |mut x,y| {
            matrix_internal_op_mut(&mut x, &y, |x,&y| {
                *x = *x + y
            });
        });
    }
/* */
    /// subtruct two matrix, the result is in the first matrix
    pub fn sub_matrix(&mut self, rhs: &Matrix<T>) {
        matrix_internal_op_mut(&mut self.data, &rhs.data, |mut x,y| {
            matrix_internal_op_mut(&mut x, &y, |x,&y| {
                *x = *x - y
            });
        });
    }

    /// Transpose a Matrix
    pub fn transpose(&self) -> Matrix<T> {
        let mut transpose_matrix: Matrix<T> = Matrix::new_with_zeros(self.cols,self.rows);

        for a in (0..self.rows).into_iter() {
            for b in (0..self.cols).into_iter() {
                transpose_matrix.data[b][a] = self.data[a][b];
            }
        }

        transpose_matrix
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
        let vector = self.into_vector();
        let vector = vector.mul_by_scalar(scalar);

        vector.into_matrix(self.cols)
    }

    /// Get column by index
    pub fn get_col(&self, n: usize) -> Vector<T> {
        let mut data: Vec<T> = Vec::new();
        for a in self.data.iter() {
            data.push(a[n]);
        };

        Vector::new_from_vec(&data)
    }

    /// Get row by index
    pub fn get_row(&self, n: usize) -> Vector<T> {
        let data = &self.data[n];

        Vector::new_from_vec(&data)
    }

    /// add a new column to the matrix, only possible when self.rows == col.len()
    pub fn add_col(&mut self, col: &Vec<T>) {
        let cols = self.ncols() + 1;
        if self.rows == col.len() {
            for k in (0..col.len()).into_iter() {
                self.data[k].push(col[k]);
            }
        }

        self.cols = cols;
    }

    pub fn add_row(&mut self, row: &Vec<T>) {
        let rows = self.nrows() + 1;
        if self.cols == row.len() {
            self.data.push(row.to_vec());
        }

        self.rows = rows;
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

    /// new Matrix from Vectors, the matrix will grow by columns
    pub fn new_from_columns(data: &Vec<Vector<T>>) -> Matrix<T> {
        let first = &data[0];

        let mut new_matrix = Matrix::new_from_vec(1, &first.get_data());
        for i in (1..data.len()).into_iter() {
            new_matrix.add_col(&data[i].get_data());
        }

        new_matrix
    }

    /// apply a function to each element of the matrix
    pub fn apply(&self, f: CallBack<T, T>) -> Matrix<T> {

        let new_matrix = self.into_vector()
                        .apply(f)
                        .into_matrix(self.ncols());
        new_matrix
    }

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

    /// zip apply a mut function to each element of the matrix
    pub fn zip_apply(&self, rhs: &Matrix<T>, f: ZipCallBack<T,T>) -> Result<Matrix<T>, String> {
        
        let matrix = self.into_vector()
                            .zip_apply(&rhs.into_vector(), f)?
                            .into_matrix(self.ncols());
        Ok(matrix)
        
    }



}



#[cfg(test)]
mod matrix_tests {
    use super::Matrix;
    use std::sync::{Arc, Mutex};
    use crate::linear_algebra::vector::{Vector, CallBack, ZipCallBack};
    #[test]
    fn new_from_vec() {
        let a = Matrix::<i32>::new_from_vec(2,&vec![2,-1,-7,4]);
        a.view();
        assert_eq!(2 + 2, 4);
    }
    
    #[test]
    fn new_with_zeros() {
        let a = Matrix::<i32>::new_with_zeros(20,10);
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
    fn add() {
        let mut a = Matrix::<i32>::new_from_vec(2,&vec![2,-1,-7,4]);
        let b = Matrix::<i32>::new_from_vec(2,&vec![-3,0,7,-4]);
        a.add_matrix(&b);
        a.view();
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn sub() {
        let mut a = Matrix::<i32>::new_from_vec(2,&vec![2,6,4,8]);
        let b = Matrix::<i32>::new_from_vec(2,&vec![3,-5,-7,9]);
        a.sub_matrix(&b);
        println!("{:?}", a);
        assert_eq!(2 + 2, 4);
    }

    


    #[test]
    fn transpose() {
        //let mat = Matrix::<i32>::new_from_vec(3,vec![2,-1,-7,4,8,12,14,15,16]);
        let mat = Matrix::<i32>::new_from_vec(3,&vec![1,5,-3,5,4,2,-3,2,0]);
        mat.view();
        let transpose = mat.transpose();
        transpose.view();
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn is_equal_to() {
        let mat = Matrix::<i32>::new_from_vec(3,&vec![2,-1,-7,4,8,12,14,15,16]);
        //let mat = Matrix::<i32>::new_from_vec(3,vec![1,5,-3,5,4,2,-3,2,0]);
        mat.view();
        let transpose = mat.transpose();
        let equl = mat.is_equal_to(&transpose);
        println!("they are equals: {:?}", equl);
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn is_symmetric() {
        //let mat = Matrix::<i32>::new_from_vec(3,vec![2,-1,-7,4,8,12,14,15,16]);
        let mat = Matrix::<i32>::new_from_vec(3,&vec![1,5,-3,5,4,2,-3,2,0]);
        let sym = mat.is_symmetric();
        println!("is_symetric: {:?}", sym);
        assert_eq!(2 + 2, 4);
    }

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
        let get_col = mat.get_col(0);
        get_col.view();
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn get_row() {
        //let mat = Matrix::<i32>::new_from_vec(3,vec![2,-1,-7,4,8,12,14,15,16]);
        let mat = Matrix::<i32>::new_from_vec(3,&vec![1,5,-3,5,4,2,-3,2,0]);
        mat.view();
        let get_col = mat.get_row(1);
        get_col.view();
        assert_eq!(2 + 2, 4);
    }
    #[test]
    fn dot_product() {
        let mat1 = Matrix::<i32>::new_from_vec(3,&vec![1, 3, -2, 0, -1, 4]);
        let mat2 = Matrix::<i32>::new_from_vec(2,&vec![2, -2, 1, 5, -3, 4]);
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
        mat1.add_col(&vec1);
        mat1.view();
        
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn add_row() {
        let mut mat1 = Matrix::<i32>::new_from_vec(3,&vec![1,3,-2,0,-1,4]);
        let vec1 = vec![4,6,10];

        mat1.view();
        mat1.add_row(&vec1);
        mat1.view();
        
        assert_eq!(2 + 2, 4);
    }


    
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
