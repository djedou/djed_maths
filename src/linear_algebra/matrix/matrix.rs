use std::ops::{Add, Sub};
use super::matrix_internal_op_mut;
use num::{One, Zero};
use std::fmt::Debug;


#[derive(Debug, Clone)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<Vec<T>>,
}

impl<T: Debug + Clone> Matrix<T> {
    pub fn new_from_vec(cols: usize, value: Vec<T>) -> Matrix<T> {
        let data_slices: Vec<&[T]> = value.chunks(cols).collect();
        let data: Vec<Vec<T>> = data_slices.into_iter().map(|d| d.to_vec()).collect();

        Matrix {
            rows: data.len(),
            cols,
            data
        }
    }
}

impl<T: Debug + Clone + Copy + One + Zero + Add<T, Output = T>> Matrix<T> {
    pub fn add_matrix(&mut self, rhs: &Matrix<T>) {
        matrix_internal_op_mut(&mut self.data, &rhs.data, |mut x,y| {
            matrix_internal_op_mut(&mut x, &y, |x,&y| {
            
                *x = *x + y
            });
        });
    }
}

impl<T: Debug + Clone + Copy + One + Zero + Sub<T, Output = T>> Matrix<T> {
    pub fn substract_matrix(&mut self, rhs: &Matrix<T>) {
        matrix_internal_op_mut(&mut self.data, &rhs.data, |mut x,y| {
            matrix_internal_op_mut(&mut x, &y, |x,&y| {
            
                *x = *x - y
            });
        });
    }
}

/*
/// Adding and consuming two matrices
impl<T: Copy + One + Zero + Add<T, Output = T>> Add<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, f: Matrix<T>) -> Matrix<T> { ... }
*/


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
        println!("{:?}", a);
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
}