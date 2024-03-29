mod expected_sarsa;
mod q_learning;
mod sarsa;

// Re-exports
pub use self::{expected_sarsa::ExpectedSARSA, q_learning::QLearning, sarsa::SARSA};

use std::collections::VecDeque;

use crate::{
    action::DiscreteAction,
    agent::Agent,
    environment::{Environment, EpisodicEnvironment},
    observation::DiscreteObservation,
    reinforcement_learning::PolicyEstimator,
    trajectory::Trajectory,
};

use super::DiscretePolicyEstimator;

struct TemporalDifferenceConfiguration {
    pub episode_limit: usize,
    pub temporal_difference_step: usize,
    pub learning_rate: f64,
    pub discount_factor: f64,
}

trait TemporalDifference<
    AC: DiscreteAction,
    S: DiscreteObservation,
    AG: Agent<Action = AC, Observation = S>,
    E: EpisodicEnvironment<Agent = AG>,
>: PolicyEstimator<Environment = E>
{
    fn algorithm_specific_evaluation(
        &self,
        agent: &AG,
        action_value: &mut [f64],
        next_step: Option<(&S, &AC)>,
    ) -> f64;

    fn temporal_difference_policy_evaluation(
        &self,
        agent: &mut AG,
        action_value: &mut [f64],
        (s, a, r, next_step): (&S, &AC, f64, Option<(&S, &AC)>),
        temporal_difference_configuration: &TemporalDifferenceConfiguration,
    ) -> f64 {
        let prev_index = Self::tabular_index(a, s);
        let algorithm_specific_evaluation =
            self.algorithm_specific_evaluation(agent, action_value, next_step);

        let old_value = action_value[prev_index];

        // Update state-action value
        action_value[prev_index] = action_value[prev_index]
            + temporal_difference_configuration.learning_rate
                * (r + temporal_difference_configuration.discount_factor
                    * algorithm_specific_evaluation
                    - action_value[prev_index]);
        // Propagate change to policy
        AC::ACTIONS.iter().for_each(|action| {
            let tabular_index = s.index() * AC::ACTIONS.len() + action.index();
            agent.policy_improvemnt(action, s, action_value[tabular_index]);
        });

        (old_value - action_value[prev_index]).powi(2)
    }

    fn temporal_difference_policy_search(
        &self,
        environment: &mut Self::Environment,
        agent: &mut <Self::Environment as Environment>::Agent,
        temporal_difference_configuration: &TemporalDifferenceConfiguration,
    ) {
        let mut action_value = vec![0.; S::OBSERVATIONS.len() * AC::ACTIONS.len()];

        let mut episode = 0usize;
        let mut episode_variation_window = VecDeque::from_iter([f64::MAX; 5]);
        while episode_variation_window
            .iter()
            .any(|ep_v| ep_v > &f64::EPSILON)
            && episode < temporal_difference_configuration.episode_limit
        {
            episode += 1;
            let mut episode_variation = 0.;

            environment.reset_environment();

            let mut temporal_difference =
                VecDeque::with_capacity(temporal_difference_configuration.temporal_difference_step);
            while let Some(observation) = environment.get_observation(agent) {
                let action = agent.act(&observation);
                let reward = environment.receive_action(agent, &action);

                if temporal_difference.len()
                    >= temporal_difference_configuration.temporal_difference_step
                {
                    let past_step = temporal_difference.pop_front().expect("There should be enough Steps on the trajectory to calculate the Temporal Difference.");

                    match past_step {
                        Trajectory::Step {
                            observation: past_obs,
                            action: past_action,
                            reward: past_reward,
                        } => {
                            episode_variation += self.temporal_difference_policy_evaluation(
                                agent,
                                &mut action_value,
                                (
                                    &past_obs,
                                    &past_action,
                                    past_reward,
                                    Some((&observation, &action)),
                                ),
                                temporal_difference_configuration,
                            );
                        }
                        _ => panic!("A final state shouldn't be reached at this point"),
                    }
                }
                temporal_difference.push_back(Trajectory::Step {
                    observation,
                    action,
                    reward,
                });
            }

            for remaining_step in temporal_difference.into_iter() {
                match remaining_step {
                    Trajectory::Step {
                        observation,
                        action,
                        reward,
                    } => {
                        episode_variation += self.temporal_difference_policy_evaluation(
                            agent,
                            &mut action_value,
                            (
                                &observation,
                                &action,
                                reward,
                                None
                            ),
                            temporal_difference_configuration
                        );
                    }
                    _ => panic!("A final state shouldn't have been added to the temporal difference sliding window."),
                }
            }

            episode_variation_window.pop_front();
            episode_variation_window.push_back(episode_variation);
        }

        Self::print_observation_action_pairs("Action Value Function", &action_value);
        println!("Iterated for {} episodes.", episode);
    }

    fn print_observation_action_pairs(header: &str, list: &[f64]) {
        println!("{header}");
        for (acts, s) in list.chunks(AC::ACTIONS.len()).zip(S::OBSERVATIONS) {
            print!("{s:?} ");
            for (action, value) in AC::ACTIONS.iter().zip(acts) {
                print!("[{action:?}; {value}] ");
            }
            println!();
        }
    }
}
