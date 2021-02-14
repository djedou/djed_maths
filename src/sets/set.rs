//use super::{SetT, SetElementT};
//use std::cmp::PartialEq;s

#[derive(Debug, Clone)]
pub struct Set {
    element: Option<Vec<f64>>,
    sub_sets: Option<Box<Set>>
}

impl Set {
    /// Set inclusion relation
    pub fn is_included_in(&self, other: &Self) -> Result<bool, &str> {
        match &self.element {
            Some(self_el) => {
                match &other.element {
                    Some(other_el) => {
                        if self_el.len() > other_el.len() {
                            Err("this set size must be smaller than the one in argument")
                        }
                        else {
                            for value in self_el.iter() {
                                if !other_el.contains(&value) {
                                    return Ok(false)
                                }
                            }
                            Ok(true)
                        }
                    },
                    None => Err("the Set you pass as argument is empty")
                }
            },
            None => Err("the Set is empty")
        }
    }

    /// Build a new Set from a vector
    pub fn from_vec(value: Vec<f64>) -> Self {
        Self {
            element: Some(value),
            sub_sets: None
        }
    }
}

#[cfg(test)]
mod set_trait_tests {

    use super::Set;
    #[test]
    fn test_inclusion_set() {

        let set_a = Set::from_vec(vec![0.2, 6.0]);
        let set_b = Set::from_vec(vec![0.2, 6.0, 3.5]);
        let set_c = Set::from_vec(vec![0.2, 6.0, 4.5]);
        let set_d = Set::from_vec(vec![0.2, 6.0, 4.5]);

        let a_in_b = set_a.is_included_in(&set_b);
        let b_in_a = set_b.is_included_in(&set_a);
        let b_in_c = set_b.is_included_in(&set_c);
        let c_in_d = set_c.is_included_in(&set_d);

        assert_eq!(a_in_b, Ok(true));
        assert_eq!(b_in_a, Err("this set size must be smaller than the one in argument"));
        assert_eq!(b_in_c, Ok(false));
        assert_eq!(c_in_d, Ok(true));
    }
}