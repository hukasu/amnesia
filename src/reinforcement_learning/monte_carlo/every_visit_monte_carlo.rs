use std::marker::PhantomData;

use crate::{
    action::DiscreteAction,
    agent::Agent,
    environment::EpisodicEnvironment,
    observation::DiscreteObservation,
    reinforcement_learning::{
        monte_carlo::{MonteCarlo, MonteCarloSearchState},
        DiscretePolicyEstimator, PolicyEstimator,
    },
    trajectory::Trajectory,
};

pub struct EveryVisitMonteCarlo<E: EpisodicEnvironment> {
    return_discount: f64,
    episodes: usize,
    phantom_environment: PhantomData<E>,
}

impl<
        AC: DiscreteAction,
        S: DiscreteObservation,
        AG: Agent<Action = AC, Observation = S>,
        E: EpisodicEnvironment<Agent = AG>,
    > EveryVisitMonteCarlo<E>
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
    > PolicyEstimator for EveryVisitMonteCarlo<E>
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
    > MonteCarlo<AC, S, AG, E> for EveryVisitMonteCarlo<E>
{
    fn step_update(
        &self,
        agent: &mut AG,
        step: &Trajectory<S, AC>,
        step_return: &f64,
        monte_carlo_search_state: MonteCarloSearchState,
    ) -> f64 {
        if let Trajectory::Step {
            observation,
            action,
            reward: _,
        } = step
        {
            let markov_reward_process_index = Self::tabular_index(action, observation);
            monte_carlo_search_state.visit_count[markov_reward_process_index] += 1;
            monte_carlo_search_state.total_returns[markov_reward_process_index] += step_return;

            let old_observation_value =
                monte_carlo_search_state.observation_values[markov_reward_process_index];

            // Update state-action value
            monte_carlo_search_state.observation_values[markov_reward_process_index] =
                monte_carlo_search_state.total_returns[markov_reward_process_index]
                    / monte_carlo_search_state.visit_count[markov_reward_process_index] as f64;
            // Propagate change to policy
            agent.policy_improvemnt(
                action,
                observation,
                monte_carlo_search_state.observation_values[markov_reward_process_index],
            );

            (old_observation_value
                - monte_carlo_search_state.observation_values[markov_reward_process_index])
                .powi(2)
        } else {
            panic!("The final step of an Episode should not be included into the Trajectory.");
        }
    }
}
