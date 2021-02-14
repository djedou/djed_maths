use std::cmp::min;
use crate::linear_algebra::Matrix;

pub(crate) fn matrix_internal_op_mut<F, T: Clone>(u: &mut [T], v: &[T], mut f: F)
where F: FnMut(&mut T, &T) 
{
    let len = min(u.len(), v.len());
    
    let xs = &mut u[..len];
    let ys = &v[..len];

    for i in 0..len {
        f(&mut xs[i], &ys[i]);
    }
}

pub(crate) fn matrix_internal_op<F, T: Clone>(u: &[T], v: &[T], f: F)
where F: Fn(&T, &T) 
{
    let len = min(u.len(), v.len());
    
    let xs = &u[..len];
    let ys = &v[..len];

    for i in 0..len {
        f(&xs[i], &ys[i])
    }
}
