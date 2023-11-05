mod constant_alpha_monte_carlo;
mod every_visit_monte_carlo;
mod first_visit_monte_carlo;
mod incremental_monte_carlo;

use std::borrow::BorrowMut;

pub use constant_alpha_monte_carlo::ConstantAlphaMonteCarlo;
pub use every_visit_monte_carlo::EveryVisitMonteCarlo;
pub use first_visit_monte_carlo::FirstVisitMonteCarlo;
pub use incremental_monte_carlo::IncrementalMonteCarlo;

use crate::{
    action::DiscreteAction, agent::Agent, environment::EpisodicEnvironment,
    observation::DiscreteObservation, trajectory::Trajectory,
};

use super::PolicyEstimator;

trait MonteCarlo<
    AC: DiscreteAction,
    S: DiscreteObservation,
    AG: Agent<Action = AC, Observation = S>,
    E: EpisodicEnvironment<Agent = AG>,
>: PolicyEstimator<Environment = E>
{
    fn generate_trajectory(
        environment: &mut E,
        agent: &mut E::Agent,
        trajectory: &mut Vec<Trajectory<S, AC>>,
    ) {
        environment.reset_environment();
        trajectory.clear();

        while let Some(observation) = environment.get_observation(agent.borrow_mut()) {
            let action = (agent.borrow_mut()).act(&observation);
            let reward = environment.receive_action(agent.borrow_mut(), &action);

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

    fn markov_reward_process_observation_action_pair_index(
        trajectory: &Trajectory<S, AC>,
    ) -> usize {
        let (observation_pos, action_pos) = match trajectory {
            Trajectory::Step {
                observation,
                action,
                reward: _,
            } => {
                let observation_pos = S::OBSERVATIONS
                    .iter()
                    .position(|discrete_observation| discrete_observation.eq(observation));
                let action_pos = AC::ACTIONS
                    .iter()
                    .position(|const_action| const_action.eq(action));
                (observation_pos, action_pos)
            }
            Trajectory::Final { observation: _ } => panic!("Can't turn the final step of a Trajectory into an index for a Markov Reward Process.")
        };
        match (observation_pos, action_pos) {
            (Some(observation_index), Some(action_index)) => {
                observation_index * AC::ACTIONS.len() + action_index
            }
            (None, _) => {
                panic!("The Trajectory contains a Observation that is not present on the list of possible Observations")
            }
            (_, None) => {
                panic!("The Trajectory contains an Action that is not present on the list of possible Actions")
            }
        }
    }

    fn make_value_fuction<'a>(observation_values: &'a [f64]) -> Box<dyn Fn(&S, &AC) -> f64 + 'a> {
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
