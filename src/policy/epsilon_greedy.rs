use std::{error::Error, fmt::Display, marker::PhantomData};

use crate::{
    action::DiscreteAction, observation::DiscreteObservation, policy::Policy,
    random_number_generator::RandomNumberGeneratorFacade,
};

#[derive(Debug)]
pub enum EpsilonGreedyPolicyError {
    EpsilonOutOfRange,
}

impl Display for EpsilonGreedyPolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let m = match self {
            Self::EpsilonOutOfRange => "Epsilon must be between [0, 1).",
        };
        write!(f, "{m}")
    }
}

impl Error for EpsilonGreedyPolicyError {}

pub struct EpsilonGreedyPolicy<
    A: DiscreteAction,
    S: DiscreteObservation,
    RNG: RandomNumberGeneratorFacade,
> {
    epsilon: f64,
    rng_facade: RNG,
    observation_action_mapping: Vec<A>,
    observation_phantom: PhantomData<S>,
}

impl<A: DiscreteAction, S: DiscreteObservation, RNG: RandomNumberGeneratorFacade>
    EpsilonGreedyPolicy<A, S, RNG>
{
    pub fn new(epsilon: f64, rng_facade: RNG) -> Result<Self, EpsilonGreedyPolicyError> {
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
            Err(EpsilonGreedyPolicyError::EpsilonOutOfRange)
        }
    }
}

impl<A: DiscreteAction, S: DiscreteObservation, RNG: RandomNumberGeneratorFacade> Policy
    for EpsilonGreedyPolicy<A, S, RNG>
{
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
                    Some(action).filter(|_| s.eq(observation.borrow())).copied()
                })
                .expect("All observations must map to an action.")
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
