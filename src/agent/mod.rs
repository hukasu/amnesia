use std::borrow::Borrow;

pub trait Agent
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
