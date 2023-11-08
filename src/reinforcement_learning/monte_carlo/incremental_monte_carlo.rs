use std::marker::PhantomData;

use crate::{
    action::DiscreteAction,
    agent::Agent,
    environment::EpisodicEnvironment,
    observation::DiscreteObservation,
    reinforcement_learning::{monte_carlo::MonteCarlo, PolicyEstimator},
    trajectory::Trajectory,
};

pub struct IncrementalMonteCarlo<E: EpisodicEnvironment> {
    return_discount: f64,
    episodes: usize,
    phantom_environment: PhantomData<E>,
}

impl<
        AC: DiscreteAction,
        S: DiscreteObservation,
        AG: Agent<Action = AC, Observation = S>,
        E: EpisodicEnvironment<Agent = AG>,
    > IncrementalMonteCarlo<E>
{
    #[must_use]
    pub fn new(return_discount: f64, episodes: usize) -> Self {
        Self {
            return_discount,
            episodes,
            phantom_environment: PhantomData,
        }
    }
}

impl<
        AC: DiscreteAction,
        S: DiscreteObservation,
        AG: Agent<Action = AC, Observation = S>,
        E: EpisodicEnvironment<Agent = AG>,
    > PolicyEstimator for IncrementalMonteCarlo<E>
{
    type Environment = E;

    fn policy_search(self, environment: &mut Self::Environment, agent: &mut E::Agent) {
        self.monte_carlo_policy_search(environment, agent, self.return_discount, self.episodes)
    }
}

impl<
        AC: DiscreteAction,
        S: DiscreteObservation,
        AG: Agent<Action = AC, Observation = S>,
        E: EpisodicEnvironment<Agent = AG>,
    > MonteCarlo<AC, S, AG, E> for IncrementalMonteCarlo<E>
{
    fn step_update(
        &self,
        step: &Trajectory<S, AC>,
        step_return: &f64,
        _visited: &mut [bool],
        visit_count: &mut [usize],
        _total_returns: &mut [f64],
        observation_values: &mut [f64],
    ) -> f64 {
        let markov_reward_process_index =
            Self::markov_reward_process_observation_action_pair_index(step);
        visit_count[markov_reward_process_index] += 1;

        let old_observation_value = observation_values[markov_reward_process_index];
        let alpha = 1. / visit_count[markov_reward_process_index] as f64;
        observation_values[markov_reward_process_index] = observation_values
            [markov_reward_process_index]
            + alpha * (step_return - observation_values[markov_reward_process_index]);
        (old_observation_value - observation_values[markov_reward_process_index]).powi(2)
    }
}
