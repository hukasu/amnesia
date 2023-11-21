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
    trajectory::Trajectory, ValueFunction,
};

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
        step: &Trajectory<S, AC>,
        step_return: &f64,
        visited: &mut [bool],
        visit_count: &mut [usize],
        total_returns: &mut [f64],
        observation_values: &mut [f64],
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
                    step,
                    step_return,
                    &mut visited,
                    &mut visit_count,
                    &mut total_returns,
                    &mut observation_values,
                );
            }
            episode_variation_window.pop_front();
            episode_variation_window.push_back(episode_variation);

            let value_function = Self::make_value_fuction(&observation_values);
            agent.policy_improvemnt(&value_function);
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

    fn tabular_index(trajectory: &Trajectory<S, AC>) -> usize {
        let (observation_pos, action_pos) = match trajectory {
            Trajectory::Step {
                observation,
                action,
                reward: _,
            } => {
                let observation_pos = S::OBSERVATIONS
                    .iter()
                    .position(|discrete_observation| discrete_observation.eq(observation));
                let action_pos = action.index();
                (observation_pos, action_pos)
            }
            Trajectory::Final { observation: _ } => panic!("Can't turn the final step of a Trajectory into an index for a Markov Reward Process.")
        };
        match (observation_pos, action_pos) {
            (Some(observation_index), action_index) => {
                observation_index * AC::ACTIONS.len() + action_index
            }
            (None, _) => {
                panic!("The Trajectory contains a Observation that is not present on the list of possible Observations")
            }
        }
    }

    fn make_value_fuction(observation_values: &[f64]) -> Box<ValueFunction<S, AC>> {
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
                        observation_index * AC::ACTIONS.len() + action_index;
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
        Box::new(value_function)
    }
}
