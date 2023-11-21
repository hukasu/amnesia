mod constant_alpha_monte_carlo;
mod every_visit_monte_carlo;
mod first_visit_monte_carlo;
mod incremental_monte_carlo;

pub use self::{
    constant_alpha_monte_carlo::ConstantAlphaMonteCarlo,
    every_visit_monte_carlo::EveryVisitMonteCarlo, first_visit_monte_carlo::FirstVisitMonteCarlo,
    incremental_monte_carlo::IncrementalMonteCarlo,
};

use std::collections::VecDeque;

use crate::{
    action::DiscreteAction, agent::Agent, environment::EpisodicEnvironment,
    observation::DiscreteObservation, reinforcement_learning::PolicyEstimator,
    trajectory::Trajectory,
};

struct MonteCarloSearchState<'a> {
    visited: &'a mut [bool],
    visit_count: &'a mut [usize],
    total_returns: &'a mut [f64],
    observation_values: &'a mut [f64],
}

trait MonteCarlo<
    AC: DiscreteAction,
    S: DiscreteObservation,
    AG: Agent<Action = AC, Observation = S>,
    E: EpisodicEnvironment<Agent = AG>,
>: PolicyEstimator<Environment = E>
{
    /// Updates the value of a Markov Reward Process state
    ///
    /// # Arguments
    /// `step`: The step within a trajectory</br>
    /// `step_return`: Discounted return from step</br>
    /// `visited`: Flags that inform if the state has already been visited on trajectory</br>
    /// `visit_count`: Number of times that state has been visited</br>
    /// `total_returns`: Sum of returns of state across all episodes</br>
    /// `observation_values`: States values
    /// # Return
    /// Change to value squared
    fn step_update(
        &self,
        agent: &mut AG,
        step: &Trajectory<S, AC>,
        step_return: &f64,
        monte_carlo_search_state: MonteCarloSearchState,
    ) -> f64;

    fn monte_carlo_policy_search(
        &self,
        environment: &mut E,
        agent: &mut AG,
        return_discount: f64,
        iteration_limit: usize,
    ) {
        let mut visited: Vec<bool> = vec![false; S::OBSERVATIONS.len() * AC::ACTIONS.len()];
        let mut visit_count = vec![0usize; S::OBSERVATIONS.len() * AC::ACTIONS.len()];
        let mut total_returns = vec![0.0f64; S::OBSERVATIONS.len() * AC::ACTIONS.len()];
        let mut observation_values = vec![0.0f64; S::OBSERVATIONS.len() * AC::ACTIONS.len()];

        let mut trajectory = vec![];
        let mut episode_returns = vec![];

        let mut episode = 0usize;
        let mut episode_variation_window = VecDeque::from_iter([f64::MAX; 5]);
        while episode_variation_window
            .iter()
            .any(|ep_v| ep_v > &f64::EPSILON)
            && episode < iteration_limit
        {
            episode += 1;
            visited.fill(false);

            let mut episode_variation = 0.;

            Self::generate_trajectory(environment, agent, &mut trajectory);
            Self::discounted_return(&trajectory, return_discount, &mut episode_returns);

            for (step, step_return) in trajectory.iter().zip(episode_returns.iter()) {
                episode_variation += self.step_update(
                    agent,
                    step,
                    step_return,
                    MonteCarloSearchState {
                        visited: &mut visited,
                        visit_count: &mut visit_count,
                        total_returns: &mut total_returns,
                        observation_values: &mut observation_values,
                    },
                );
            }
            episode_variation_window.pop_front();
            episode_variation_window.push_back(episode_variation);
        }

        Self::print_observation_action_pairs(
            "Observation Visit Count",
            &visit_count.iter().map(|u| *u as f64).collect::<Vec<_>>(),
        );
        Self::print_observation_action_pairs("Action Value Function", &observation_values);
        println!("Iterated for {} episodes.", episode);
    }

    fn generate_trajectory(
        environment: &mut E,
        agent: &mut E::Agent,
        trajectory: &mut Vec<Trajectory<S, AC>>,
    ) {
        environment.reset_environment();
        trajectory.clear();

        while let Some(observation) = environment.get_observation(agent) {
            let action = agent.act(&observation);
            let reward = environment.receive_action(agent, &action);

            trajectory.push(Trajectory::Step {
                observation,
                action,
                reward,
            });
        }
        trajectory.push(Trajectory::Final {
            observation: environment.final_observation(agent),
        });
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
