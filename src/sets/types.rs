

#[derive(Debug, PartialEq)]
pub enum Inclusion<T> {
    ProperlyIncluded,
    NotIncluded(Vec<T>),
    NotAllowed
}

#[derive(Debug, PartialEq)]
pub enum Complement<T> {
    Value(Vec<T>),
    NotAllowed
}