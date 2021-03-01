use std::fmt::Debug;
use super::vector::Vector;

/// It has all functions and methods of Vector<T>
pub type Vector3D<T> = Vector<T>;


impl<T: Debug + Clone + Default> Vector3D<T> {
    /// new Vector3D
    pub fn new_vec3d(value: &[T; 3]) -> Vector3D<T> {

        Vector3D::new_from_vec(&value.to_vec())
    }
}


#[cfg(test)]
mod vector3d_tests {
    use super::Vector3D;
 

    #[test]
    fn new() {
        let a = Vector3D::<i32>::new_vec3d(&[2,-1,3]);
        a.view();
        assert_eq!(2 + 2, 4);
    }
}