use std::marker::PhantomData;

use crate::{
    action::DiscreteAction, observation::DiscreteObservation,
    random_number_generator::RandomNumberGeneratorFacade,
};

pub mod epsilon_greedy;

pub trait Policy
where
    Self: Sized,
{
    type Action: crate::action::Action;
    type Observation: crate::observation::Observation;

    fn act(&self, observation: &Self::Observation) -> Self::Action;

    fn policy_improvemnt(
        &mut self,
        action: &Self::Action,
        observation: &Self::Observation,
        value: f64,
    );
}

pub struct Greedy<A: DiscreteAction, S: DiscreteObservation> {
    observation_action_mapping: Vec<(A, f64)>,
    observation_phantom: PhantomData<S>,
}

impl<A: DiscreteAction, S: DiscreteObservation> Greedy<A, S> {
    #[must_use]
    pub fn new(start_state_randomizer: &dyn RandomNumberGeneratorFacade) -> Self {
        let random_start = S::OBSERVATIONS
            .iter()
            .map(|_| {
                (
                    A::ACTIONS
                        [(start_state_randomizer.random() * A::ACTIONS.len() as f64) as usize],
                    0.,
                )
            })
            .collect();
        Self {
            observation_action_mapping: random_start,
            observation_phantom: PhantomData,
        }
    }
}

impl<A: DiscreteAction, S: DiscreteObservation> Policy for Greedy<A, S> {
    type Action = A;
    type Observation = S;

    fn act(&self, observation: &Self::Observation) -> Self::Action {
        self.observation_action_mapping[observation.index()].0
    }

    fn policy_improvemnt(
        &mut self,
        action: &Self::Action,
        observation: &Self::Observation,
        value: f64,
    ) {
        let observation_index = observation.index();
        if self.observation_action_mapping[observation_index]
            .0
            .eq(action)
        {
            self.observation_action_mapping[observation_index].1 = value;
        } else if self.observation_action_mapping[observation_index].1 < value {
            self.observation_action_mapping[observation_index] = (*action, value);
        }
    }
}
