use std::borrow::Borrow;

use crate::ValueFunction;

pub trait Agent
where
    Self: Sized,
{
    type Action: crate::action::Action;
    type Observation: crate::observation::Observation;

    fn act(&self, observation: impl Borrow<Self::Observation>) -> Self::Action;

    fn policy_improvemnt(
        &mut self,
        value_function: &ValueFunction<Self::Observation, Self::Action>,
    );
}
