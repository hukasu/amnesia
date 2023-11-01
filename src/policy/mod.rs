pub mod epsilon_greedy;

use std::borrow::Borrow;

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
