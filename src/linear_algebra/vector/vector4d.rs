use std::fmt::Debug;
use super::vector::Vector;

/// It has all functions and methods of Vector<T>
pub type Vector4D<T> = Vector<T>;

/*
impl<T: Debug + Clone + Default> Vector4D<T> {
    /// new Vector4D
    pub fn new_vec4d(value: &[T; 4]) -> Vector4D<T> {

        Vector4D::new_from_vec(&value.to_vec())
    }
}


#[cfg(test)]
mod vector4d_tests {
    use super::Vector4D;
 

    #[test]
    fn new() {
        let a = Vector4D::<i32>::new_vec4d(&[2,-1,3,5]);
        a.view();
        assert_eq!(2 + 2, 4);
    }
}
*/