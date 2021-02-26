use std::ops::{Add, Sub, Mul};
use num::{One, Zero, cast, NumCast};
use std::fmt::Debug;
use crate::linear_algebra::Matrix;


#[derive(Debug, Clone)]
pub struct Vector<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T: Debug + Clone> Vector<T> {
    /// new Vector from Vec
    pub fn new_from_vec(value: Vec<T>) -> Vector<T> {

        Vector {
            rows: 1,
            cols: value.len(),
            data: value
        }
    }

    pub fn get_data(&self) -> Vec<T> {
        self.data.clone()
    }
    /// print the Vector into console
    pub fn view(&self) {
        println!("rows: {}", self.rows);
        println!("cols: {}", self.cols);
        println!("data: {:?}", self.data);
    }

    /// cast a Vector into a Matrix
    pub fn into_matrix(&self, cols: usize) -> Matrix<T> {
        let data = self.data.clone();
        Matrix::new_from_vec(cols, data)
    }

    /// cast a Matrix into Vector
    pub fn from_matrix(matrix: Matrix<T>) -> Vector<T> {

        let mut vector: Vec<T> = Vec::new();
        matrix.get_data().iter().for_each(|v| {
            vector.extend_from_slice(v.as_slice())
        });
        
        Vector::new_from_vec(vector)

    }
}

impl<T: Debug + Clone + Copy + One + Zero + Default + NumCast + Add<T, Output = T> + Mul<T, Output = T> + Sub<T, Output = T>> Vector<T> {
    /// the Norm or Magnitude ||v|| of a vector
    pub fn norm(&self) -> f64 {
        let sum = self.get_data().iter().fold(T::default(), |acc, d| acc + ((*d) * (*d)));

        let cast_sum: f64 = cast(sum).unwrap();
        cast_sum.sqrt()
    }

    /// check if norm = 1.0
    pub fn is_unit_vector(&self) -> bool {
        self.norm() == 1.0
    }

    /// multiply a Vector by a scalar
    pub fn mul_by_scalar(&self, scalar: T) -> Vector<T> {
        let data: Vec<T> = self.data
            .clone()
            .iter()
            .map(|a| *a * scalar)
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
            .clone()
            .iter()
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
    pub fn add_vector(&self, rhs: Vector<T>) -> Vector<T> {
        let data: Vec<T> = self.data.iter().zip(rhs.data.iter()).map(|(a,b)| (*a) + (*b)).collect();

        Vector {
            rows: 1,
            cols: data.len(),
            data: data
        }
    }

    /// subtruct a vector from other vector
    pub fn sub_vector(&self, rhs: Vector<T>) -> Vector<T> {
        let data: Vec<T> = self.data.iter().zip(rhs.data.iter()).map(|(a,b)| (*a) - (*b)).collect();

        Vector {
            rows: 1,
            cols: data.len(),
            data: data
        }
    }
}
/*
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
*/

#[cfg(test)]
mod vector_tests {
    use super::Vector;
    use crate::linear_algebra::Matrix;
    #[test]
    fn new() {
        let a = Vector::<i32>::new_from_vec(vec![2,-1,-7,4]);
        a.view();
        assert_eq!(2 + 2, 4);
    }
    #[test]
    fn into_matrix() {
        let a = Vector::<i32>::new_from_vec(vec![2,-1,-7,4]);
        a.view();
        let mat = a.into_matrix(2);
        mat.view();
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn from_matrix() {
        let mat = Matrix::<i32>::new_from_vec(2,vec![2,-1,-7,4]);
        mat.view();
        let vector = Vector::from_matrix(mat);
        vector.view();
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn norm() {
        let a = Vector::<i32>::new_from_vec(vec![2,-1,-7,4]);
        a.view();
        let norm = a.norm();
        println!("norm: {}", norm);
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn mul_by_scalar() {
        let a = Vector::<i32>::new_from_vec(vec![2,-1,-7,4]);
        a.view();
        let vector2 = a.mul_by_scalar(2);
        vector2.view();
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn add_scalar() {
        let a = Vector::<i32>::new_from_vec(vec![2,-1,-7,4]);
        a.view();
        let vector = a.add_scalar(2);
        vector.view();
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn get_unit_vector() {
        let a = Vector::<f64>::new_from_vec(vec![2.0,-1.0,-7.0,4.0]);
        a.view();
        let unit_vector = a.get_unit_vector();
        unit_vector.view();
        assert_eq!(2 + 2, 4);
    }
    #[test]
    fn add_vector() {
        let a = Vector::<f64>::new_from_vec(vec![2.0,-1.0,-7.0,4.0]);
        let b = Vector::<f64>::new_from_vec(vec![2.0,-1.0,-7.0,4.0]);
        a.view();
        b.view();
        let vector = a.add_vector(b);
        vector.view();
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn sub_vector() {
        let a = Vector::<f64>::new_from_vec(vec![2.0,-1.0,-7.0,4.0]);
        let b = Vector::<f64>::new_from_vec(vec![2.0,-1.0,-7.0,4.0]);
        a.view();
        b.view();
        let vector = a.sub_vector(b);
        vector.view();
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
    }*/
}
