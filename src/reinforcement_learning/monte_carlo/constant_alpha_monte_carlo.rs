use std::marker::PhantomData;

use crate::trajectory::Trajectory;
use crate::{
    action::DiscreteAction, agent::Agent, environment::EpisodicEnvironment,
    observation::DiscreteObservation,
};

use crate::reinforcement_learning::PolicyEstimator;

use super::MonteCarlo;

pub struct ConstantAlphaMonteCarlo<E: EpisodicEnvironment> {
    alpha: f64,
    return_discount: f64,
    episodes: usize,
    phantom_environment: PhantomData<E>,
}

impl<
        AC: DiscreteAction,
        S: DiscreteObservation,
        AG: Agent<Action = AC, Observation = S>,
        E: EpisodicEnvironment<Agent = AG>,
    > ConstantAlphaMonteCarlo<E>
{
    #[must_use]
    pub fn new(alpha: f64, return_discount: f64, episodes: usize) -> Self {
        Self {
            alpha,
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
    > PolicyEstimator for ConstantAlphaMonteCarlo<E>
{
    type Environment = E;

    fn policy_search(self, environment: &mut Self::Environment, agent: &mut E::Agent) {
        self.monte_carlo_policy_search(environment, agent, self.return_discount, self.episodes);
    }
}

impl<
        AC: DiscreteAction,
        S: DiscreteObservation,
        AG: Agent<Action = AC, Observation = S>,
        E: EpisodicEnvironment<Agent = AG>,
    > MonteCarlo<AC, S, AG, E> for ConstantAlphaMonteCarlo<E>
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
        let markov_reward_process_index = Self::tabular_index(step);
        visit_count[markov_reward_process_index] += 1;

        let old_observation_value = observation_values[markov_reward_process_index];
        observation_values[markov_reward_process_index] = observation_values
            [markov_reward_process_index]
            + self.alpha * (step_return - observation_values[markov_reward_process_index]);
        (old_observation_value - observation_values[markov_reward_process_index]).powi(2)
    }
}
