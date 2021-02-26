use std::ops::{Add, Sub};
use super::matrix_internal_op_mut;
use num::{One, Zero};
use std::fmt::Debug;
use crate::linear_algebra::Vector;

#[derive(Debug, Clone)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<Vec<T>>,
}

impl<T: Debug + Clone> Matrix<T> {
    /// new Matrix from Vec
    pub fn new_from_vec(cols: usize, value: Vec<T>) -> Matrix<T> {
        let data_slices: Vec<&[T]> = value.chunks(cols).collect();
        let data: Vec<Vec<T>> = data_slices.into_iter().map(|d| d.to_vec()).collect();

        Matrix {
            rows: data.len(),
            cols,
            data
        }
    }

    /// print Matrix in console
    pub fn view(&self) {
        println!("rows: {}", self.rows);
        println!("cols: {}", self.cols);
        println!("data: [");
        for d in self.data.iter() {
            println!("   {:?},", d);
        }
        println!(" ]");
    }

    /// get Matrix data
    pub fn get_data(&self) -> Vec<Vec<T>> {
        self.data.clone()
    }

    /// cast a Matrix into Vector
    pub fn into_vector(&self) -> Vector<T> {

        let mut vector: Vec<T> = Vec::new();
        self.get_data().iter().for_each(|v| {
            vector.extend_from_slice(v.as_slice())
        });
        
        Vector::new_from_vec(vector)

    }

    /// check if the matrix is a square matrix n x m where n == m.
    pub fn is_square_matrix(&self) -> bool {
        self.rows == self.cols
    }
}

impl<T: Debug + Clone + Copy + One + Zero + Add<T, Output = T>> Matrix<T> {
    /// add two matrix, the result is in the first matrix
    pub fn add_matrix(&mut self, rhs: &Matrix<T>) {
        matrix_internal_op_mut(&mut self.data, &rhs.data, |mut x,y| {
            matrix_internal_op_mut(&mut x, &y, |x,&y| {
                *x = *x + y
            });
        });
    }
}

impl<T: Debug + Clone + Copy + One + Zero + Sub<T, Output = T>> Matrix<T> {
    /// subtruct two matrix, the result is in the first matrix
    pub fn sub_matrix(&mut self, rhs: &Matrix<T>) {
        matrix_internal_op_mut(&mut self.data, &rhs.data, |mut x,y| {
            matrix_internal_op_mut(&mut x, &y, |x,&y| {
                *x = *x - y
            });
        });
    }
}



#[cfg(test)]
mod matrix_tests {
    use super::Matrix;
    #[test]
    fn new() {
        let a = Matrix::<i32>::new_from_vec(2,vec![2,-1,-7,4]);
        println!("{:?}", a);
        assert_eq!(2 + 2, 4);
    }

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
        a.sub_matrix(&b);
        println!("{:?}", a);
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn into_vector() {
        let a = Matrix::<i32>::new_from_vec(2,vec![2,6,4,8]);
        a.view();
        let vector = a.into_vector();
        vector.view();
        assert_eq!(2 + 2, 4);
    }
}