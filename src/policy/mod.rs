pub mod epsilon_greedy;

use std::borrow::Borrow;
use std::{error::Error, fmt::Display};

#[derive(Debug)]
pub struct PolicyError {
    message: String,
}

impl PolicyError {
    pub fn new(message: String) -> Self {
        Self { message }
    }
}

impl Display for PolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "An error occurred on a Policy.\n\t-> {}", self.message)
    }
}

impl Error for PolicyError {}

pub trait Policy
where
    Self: Sized,
{
    type Action: crate::action::Action;
    type Observation: crate::observation::Observation;

    fn act(&self, observation: impl Borrow<Self::Observation>) -> Self::Action;

    fn policy_improvemnt(
        &mut self,
        value_function: impl Fn(&Self::Observation, &Self::Action) -> f64,
    );
}
