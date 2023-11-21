use std::marker::PhantomData;

use crate::{
    action::DiscreteAction,
    agent::{Agent, DiscreteAgent},
    environment::{Environment, EpisodicEnvironment},
    observation::DiscreteObservation,
    reinforcement_learning::{
        temporal_difference::{TemporalDifference, TemporalDifferenceConfiguration},
        DiscretePolicyEstimator, PolicyEstimator,
    },
};

pub struct ExpectedSARSA<E: EpisodicEnvironment> {
    episode_limit: usize,
    learning_rate: f64,
    discount_factor: f64,
    phantom_env: PhantomData<E>,
}

impl<
        AC: DiscreteAction,
        S: DiscreteObservation,
        AG: DiscreteAgent<AC, S> + Agent<Action = AC, Observation = S>,
        E: EpisodicEnvironment<Agent = AG>,
    > ExpectedSARSA<E>
{
    pub fn new(episode_limit: usize, alpha: f64, discount_factor: f64) -> Self {
        Self {
            episode_limit,
            learning_rate: alpha,
            discount_factor,
            phantom_env: PhantomData,
        }
    }
}

impl<
        AC: DiscreteAction,
        S: DiscreteObservation,
        AG: DiscreteAgent<AC, S> + Agent<Action = AC, Observation = S>,
        E: EpisodicEnvironment<Agent = AG>,
    > PolicyEstimator for ExpectedSARSA<E>
{
    type Environment = E;

    fn policy_search(
        self,
        environment: &mut Self::Environment,
        agent: &mut <Self::Environment as Environment>::Agent,
    ) {
        self.temporal_difference_policy_search(
            environment,
            agent,
            &TemporalDifferenceConfiguration {
                episode_limit: self.episode_limit,
                temporal_difference_step: 1,
                learning_rate: self.learning_rate,
                discount_factor: self.discount_factor,
            },
        )
    }
}

impl<
        AC: DiscreteAction,
        S: DiscreteObservation,
        AG: DiscreteAgent<AC, S> + Agent<Action = AC, Observation = S>,
        E: EpisodicEnvironment<Agent = AG>,
    > TemporalDifference<AC, S, AG, E> for ExpectedSARSA<E>
{
    fn algorithm_specific_evaluation(
        &self,
        agent: &AG,
        action_value: &mut [f64],
        next_step: Option<(&S, &AC)>,
    ) -> f64 {
        match next_step {
            Some((next_state, _next_action)) => AC::ACTIONS
                .iter()
                .map(|action| {
                    let cur_index = Self::tabular_index(action, next_state);
                    agent.action_probability(action, next_state) * action_value[cur_index]
                })
                .sum(),
            None => 0.,
        }
    }
}
