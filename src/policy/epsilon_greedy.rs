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
    observation_action_mapping: Vec<(A, f64)>,
    observation_phantom: PhantomData<S>,
}

impl<A: DiscreteAction, S: DiscreteObservation, RNG: RandomNumberGeneratorFacade>
    EpsilonGreedyPolicy<A, S, RNG>
{
    pub fn new(epsilon: f64, rng_facade: RNG) -> Result<Self, EpsilonGreedyPolicyError> {
        if (0.0f64..1.0).contains(&epsilon) {
            let random_start = S::OBSERVATIONS
                .iter()
                .map(|_| {
                    (
                        A::ACTIONS[(rng_facade.random() * A::ACTIONS.len() as f64) as usize],
                        0.,
                    )
                })
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

    fn act(&self, observation: &Self::Observation) -> Self::Action {
        if self.rng_facade.random().lt(&self.epsilon) {
            A::ACTIONS[(self.rng_facade.random() * A::ACTIONS.len() as f64) as usize]
        } else {
            let obs_index = S::OBSERVATIONS
                .iter()
                .position(|obs| obs.eq(observation))
                .expect("All observations must map to an action.");
            self.observation_action_mapping[obs_index].0
        }
    }

    fn policy_improvemnt(
        &mut self,
        action: &Self::Action,
        observation: &Self::Observation,
        value: f64,
    ) {
        let observation_index = S::OBSERVATIONS
            .iter()
            .position(|discrete_obs| discrete_obs.eq(observation))
            .expect("Observation should exist in Discrete observations.");
        if self.observation_action_mapping[observation_index].1 < value {
            self.observation_action_mapping[observation_index] = (*action, value);
        }
    }
}
