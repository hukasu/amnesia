use crate::{action::DiscreteAction, observation::DiscreteObservation, ValueFunction};

pub trait Agent
where
    Self: Sized,
{
    type Action: crate::action::Action;
    type Observation: crate::observation::Observation;

    fn act(&self, observation: &Self::Observation) -> Self::Action;

    fn policy_improvemnt(
        &mut self,
        value_function: &ValueFunction<Self::Observation, Self::Action>,
    );
}

pub trait DiscreteAgent<AC: DiscreteAction, S: DiscreteObservation>:
    Agent<Action = AC, Observation = S>
{
    fn action_probability(&self, action: &AC, observation: &S) -> f64;
}
