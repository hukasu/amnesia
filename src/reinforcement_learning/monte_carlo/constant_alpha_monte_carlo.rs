use std::marker::PhantomData;

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
        let mut visit_count = vec![0usize; S::OBSERVATIONS.len() * (AC::ACTIONS.len() + 1)];
        let mut observation_values = vec![0.0f64; S::OBSERVATIONS.len() * (AC::ACTIONS.len() + 1)];

        for _episode in 0..self.episodes {
            let trajectory = Self::generate_trajectory(environment, agent);
            let episode_returns = Self::discounted_return(trajectory.iter(), self.return_discount);

            for (step, g_t) in trajectory.iter().zip(episode_returns.iter()) {
                let markov_reward_process_index =
                    Self::markov_reward_process_observation_action_pair_index(step);
                visit_count[markov_reward_process_index] += 1;
                observation_values[markov_reward_process_index] = observation_values
                    [markov_reward_process_index]
                    + self.alpha * (g_t - observation_values[markov_reward_process_index]);
            }

            let value_function = |observation: &S, action: &AC| {
                let observation_pos = S::OBSERVATIONS
                    .iter()
                    .position(|discrete_observation| discrete_observation.eq(observation));
                let action_pos = AC::ACTIONS
                    .iter()
                    .position(|const_action| const_action.eq(action));
                match (observation_pos, action_pos) {
                    (Some(observation_index), Some(action_index)) => {
                        let markov_reward_process_index =
                            observation_index * (AC::ACTIONS.len() + 1) + action_index;
                        observation_values[markov_reward_process_index]
                    }
                    (None, _) => {
                        panic!("The Trajectory contains a Observation that is not present on the list of possible Observations")
                    }
                    (_, None) => {
                        panic!("The Trajectory contains an Action that is not present on the list of possible Actions")
                    }
                }
            };
            agent.policy_improvemnt(value_function);
        }

        Self::print_observation_action_pairs(
            "Observation Visit Count",
            &visit_count.iter().map(|u| *u as f64).collect::<Vec<_>>(),
        );
        Self::print_observation_action_pairs("Action Value Function", &observation_values);
    }
}

impl<
        AC: DiscreteAction,
        S: DiscreteObservation,
        AG: Agent<Action = AC, Observation = S>,
        E: EpisodicEnvironment<Agent = AG>,
    > MonteCarlo<AC, S, AG, E> for ConstantAlphaMonteCarlo<E>
{
}
