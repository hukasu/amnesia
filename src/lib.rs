pub mod action;
pub mod agent;
pub mod environment;
pub mod observation;
pub mod policy;
pub mod random_number_generator;
pub mod reinforcement_learning;
pub mod trajectory;

pub type ValueFunction<'a, S, A> = dyn Fn(&S, &A) -> f64 + 'a;
