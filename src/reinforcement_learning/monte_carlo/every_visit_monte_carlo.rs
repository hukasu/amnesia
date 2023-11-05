use std::marker::PhantomData;

use crate::{
    action::DiscreteAction, agent::Agent, environment::EpisodicEnvironment,
    observation::DiscreteObservation,
};

use crate::reinforcement_learning::PolicyEstimator;

use super::MonteCarlo;

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
        let mut visit_count = vec![0usize; S::OBSERVATIONS.len() * AC::ACTIONS.len()];
        let mut returns = vec![0.0f64; S::OBSERVATIONS.len() * AC::ACTIONS.len()];
        let mut observation_values = vec![0.0f64; S::OBSERVATIONS.len() * AC::ACTIONS.len()];
        let mut trajectory = vec![];
        let mut episode_returns = vec![];

        for _episode in 0..self.episodes {
            Self::generate_trajectory(environment, agent, &mut trajectory);
            Self::discounted_return(&trajectory, self.return_discount, &mut episode_returns);

            for (step, g_t) in trajectory.iter().zip(episode_returns.iter()) {
                let markov_reward_process_index =
                    Self::markov_reward_process_observation_action_pair_index(step);
                visit_count[markov_reward_process_index] += 1;
                returns[markov_reward_process_index] += g_t;
                observation_values[markov_reward_process_index] = returns
                    [markov_reward_process_index]
                    / visit_count[markov_reward_process_index] as f64;
            }

            let value_function = Self::make_value_fuction(&observation_values);
            agent.policy_improvemnt(value_function);
        }

        Self::print_observation_action_pairs(
            "Observation Visit Count",
            &visit_count.iter().map(|u| *u as f64).collect::<Vec<_>>(),
        );
        Self::print_observation_action_pairs("Returns", &returns);
        Self::print_observation_action_pairs("Action Value Function", &observation_values);
    }
}

impl<
        AC: DiscreteAction,
        S: DiscreteObservation,
        AG: Agent<Action = AC, Observation = S>,
        E: EpisodicEnvironment<Agent = AG>,
    > MonteCarlo<AC, S, AG, E> for EveryVisitMonteCarlo<E>
{
}
