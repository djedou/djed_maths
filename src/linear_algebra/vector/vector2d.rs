use std::fmt::Debug;
use super::vector::Vector;

/// It has all functions and methods of Vector<T>
pub type Vector2D<T> = Vector<T>;


impl<T: Debug + Clone + Default> Vector2D<T> {
    /// new Vector2D
    pub fn new_vec2d(value: &[T; 2]) -> Vector2D<T> {

        Vector2D::new_from_vec(&value.to_vec())
    }
}


#[cfg(test)]
mod vector2d_tests {
    use super::Vector2D;
 

    #[test]
    fn new() {
        let a = Vector2D::<i32>::new_vec2d(&[2,-1]);
        a.view();
        assert_eq!(2 + 2, 4);
    }
}