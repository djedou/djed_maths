use std::cmp::{PartialEq, PartialOrd};
use std::fmt::Debug;
use super::{Inclusion, Complement};

#[derive(Debug, Clone)]
pub struct Set<T> {
    elements: Vec<T>,
    sub_sets: Option<Box<Set<T>>>
}

impl<T: Debug + Clone> Set<T> {
    pub fn new_from_vec(value: Vec<T>) -> Set<T> {
        Set {  
            elements: value,
            sub_sets: None
        }
    }
}

impl<T: PartialOrd + PartialEq + Debug + Clone + Copy> Set<T> {
    /// Check if one Set is included in another one
    pub fn is_included_in(&self, rhs: &Set<T>) -> Inclusion<T> {

        if self.elements.len() <= rhs.elements.len() {

            let mut not_proper_include: Vec<T> = Vec::new();
    
            // check if every element of self is in rhs
            for a in  self.elements.iter() {
                if !rhs.elements.contains(&a) {
                    not_proper_include.push(*a);
                }
            }
    
            if not_proper_include.len() > 0 {
                Inclusion::NotIncluded(not_proper_include)
            }
            else {
                Inclusion::ProperlyIncluded
            }
        }
        else {
            Inclusion::NotAllowed
        }
    }

    /// Check if two Sets are identic
    pub fn is_identical_to(&self, rhs: &Set<T>) -> bool {
        if self.elements.len() != rhs.elements.len() {
            false
        }
        else {
            for a in  self.elements.iter() {
                if !rhs.elements.contains(&a) {
                    return false;
                }
            }
            true
        }
    }

    /// Check if a Set is empty
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Check if two Sets have not elements together
    pub fn is_disjoint_with(&self, rhs: &Set<T>) -> bool {
        let mut count = 0;
        for a in self.elements.iter() {
            if rhs.elements.contains(&a) {
                count = count + 1;
            }
        }

        if count == 0 {
            true
        }
        else {
            false
        }
    }

    /// Get the intersection (meet) of two sets
    pub fn get_intersection_with(&self, rhs: &Set<T>) -> Set<T> {
        let mut elements = Vec::new();
        for a in self.elements.iter() {
            if rhs.elements.contains(&a) {
                elements.push(*a);
            }
        }

        Set {
            elements,
            sub_sets: None
        }
    }

    /// Get the union of two sets
    pub fn get_union_with(&self, rhs: &Set<T>) -> Set<T> {
        let mut elements = self.elements.clone();
        let mut right_set = rhs.elements.clone();
        elements.append(&mut right_set);
        elements.sort_by(|a, b| a.partial_cmp(b).unwrap());
        elements.dedup();

        Set {
            elements,
            sub_sets: None
        }
    }

    /// Set of all elements of set A that are not elements of set B
    pub fn get_difference_from(&self, rhs: &Set<T>) -> Set<T> {
        let mut elements = Vec::new();
        for a in self.elements.iter() {
            if !rhs.elements.contains(&a) {
                elements.push(*a);
            }
        }

        Set {
            elements,
            sub_sets: None
        }
    }


    /// Get the complementation (-A) of set A from the universal Set U.
    /// Here all elements of A are in U, complement -A is a set of elements that 
    /// are in U but not in A.
    pub fn get_complement_from_u(&self, rhs: &Set<T>) -> Complement<T> {

        if self.elements.len() <= rhs.elements.len() {

            match self.is_included_in(&rhs) {
                Inclusion::ProperlyIncluded => {
                    let mut complement: Vec<T> = Vec::new();
    
                    for a in  rhs.elements.iter() {
                        if !self.elements.contains(&a) {
                            complement.push(*a);
                        }
                    }

                    Complement::Value(complement)

                },
                _ => Complement::NotAllowed
            }
        }
        else {
            Complement::NotAllowed
        }
    }

}

/*

#[cfg(test)]
mod set_trait_tests {

    use super::{Set, Inclusion};
    #[test]
    fn test_inclusion_set() {

        let set_a = Set::new_from_vec(vec![0.2, 6.0]);
        let set_b = Set::new_from_vec(vec![0.2, 6.0, 3.5]);
        
        let a_in_b = set_a.is_included_in(&set_b);
        let b_in_a = set_b.is_included_in(&set_a);

        assert_eq!(a_in_b, Inclusion::ProperlyIncluded);
        assert_eq!(b_in_a, Inclusion::NotAllowed);
    }

    #[test]
    fn test_identical_set() {

        let set_a = Set::new_from_vec(vec![0.2, 6.0]);
        let set_b = Set::new_from_vec(vec![0.2, 6.0, 3.5]);
        
        let set_c = Set::new_from_vec(vec![0.2, 6.0, 4.5]);
        let set_d = Set::new_from_vec(vec![0.2, 6.0, 4.5]);

        let a_identic_to_b = set_a.is_identical_to(&set_b);
        let c_identic_to_d = set_c.is_identical_to(&set_d);
        /*let b_in_c = set_b.is_included_in(&set_c);
        let c_in_d = set_c.is_included_in(&set_d);*/

        assert_eq!(a_identic_to_b, false);
        assert_eq!(c_identic_to_d, true);
        /*assert_eq!(b_in_a, Err("this set size must be smaller than the one in argument"));
        assert_eq!(b_in_c, Ok(false));
        assert_eq!(c_in_d, Ok(true));*/
    }

    #[test]
    fn get_intersection_with() {

        let set_a = Set::new_from_vec(vec![0.2, 5.0, 6.0, 7.58]);
        let set_b = Set::new_from_vec(vec![0.2, 6.0, 3.5]);
        
        let set_c = set_a.get_intersection_with(&set_b);

        println!("intersection: {:?}", set_c);
    }

    #[test]
    fn get_union_with() {

        let set_a = Set::new_from_vec(vec![0.2, 5.0, 6.0, 7.58]);
        let set_b = Set::new_from_vec(vec![0.2, 6.0, 3.5]);
        
        let set_c = set_a.get_union_with(&set_b);

        println!("union: {:?}", set_c);
    }

    #[test]
    fn get_difference_from() {

        let set_a = Set::new_from_vec(vec![0.2, 5.0, 6.0, 7.58]);
        let set_b = Set::new_from_vec(vec![0.2, 6.0, 3.5]);
        
        let set_c = set_a.get_difference_from(&set_b);

        println!("difference: {:?}", set_c);
    }

    #[test]
    fn get_complement_from_u() {

        let set_u = Set::new_from_vec(vec![0.2, 5.0, 6.0, 7.58]);
        let set_a = Set::new_from_vec(vec![0.2, 6.0]);
        
        let complement = set_a.get_complement_from_u(&set_u);

        // [5.0, 7.58]
        println!("complement: {:?}", complement);

    }
}
*/