use std::marker::PhantomData;

use crate::{
    action::DiscreteAction, observation::DiscreteObservation,
    random_number_generator::RandomNumberGeneratorFacade,
};

use super::{Policy, PolicyError};

pub struct EpsilonGreedyPolicy<A: DiscreteAction, S: DiscreteObservation> {
    epsilon: f64,
    rng_facade: Box<dyn RandomNumberGeneratorFacade>,
    observation_action_mapping: Vec<A>,
    observation_phantom: PhantomData<S>,
}

impl<A: DiscreteAction, S: DiscreteObservation> EpsilonGreedyPolicy<A, S> {
    pub fn new(
        epsilon: f64,
        rng_facade: Box<dyn RandomNumberGeneratorFacade>,
    ) -> Result<Self, PolicyError> {
        if (0.0f64..1.0).contains(&epsilon) {
            let random_start = S::OBSERVATIONS
                .iter()
                .map(|_| A::ACTIONS[(rng_facade.random() * A::ACTIONS.len() as f64) as usize])
                .collect();
            Ok(Self {
                epsilon,
                rng_facade,
                observation_action_mapping: random_start,
                observation_phantom: PhantomData,
            })
        } else {
            Err(PolicyError::new(format!(
                "EpsilonGreedyPolicy requires an `epsilon` between `0.0` and `1.0`. Received {epsilon}"
            )))
        }
    }
}

impl<A: DiscreteAction, S: DiscreteObservation> Policy for EpsilonGreedyPolicy<A, S> {
    type Observation = S;
    type Action = A;

    fn act(&self, observation: impl std::borrow::Borrow<Self::Observation>) -> Self::Action {
        if self.rng_facade.random().lt(&self.epsilon) {
            A::ACTIONS[(self.rng_facade.random() * A::ACTIONS.len() as f64) as usize]
        } else {
            S::OBSERVATIONS
                .iter()
                .zip(self.observation_action_mapping.iter())
                .find_map(|(s, action)| {
                    Some(action).copied().filter(|_| s.eq(observation.borrow()))
                })
                .expect("All observations must have a corresponding action.")
        }
    }

    fn policy_improvemnt(
        &mut self,
        value_function: impl Fn(&Self::Observation, &Self::Action) -> f64,
    ) {
        self.observation_action_mapping
            .iter_mut()
            .zip(Self::Observation::OBSERVATIONS.iter())
            .for_each(|(policy_action, observation)| {
                let max_action = Self::Action::ACTIONS.iter().max_by(|lhs, rhs| {
                    value_function(observation, lhs).total_cmp(&value_function(observation, rhs))
                });
                if let Some(best_action) = max_action {
                    *policy_action = *best_action;
                } else {
                    panic!("Could not determinate best action for Observtion {observation:?}")
                }
            });
    }
}
