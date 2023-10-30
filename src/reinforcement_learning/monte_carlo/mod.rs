mod constant_alpha_monte_carlo;
mod every_visit_monte_carlo;
mod first_visit_monte_carlo;

use std::borrow::BorrowMut;

pub use constant_alpha_monte_carlo::ConstantAlphaMonteCarlo;
pub use every_visit_monte_carlo::EveryVisitMonteCarlo;
pub use first_visit_monte_carlo::FirstVisitMonteCarlo;

use crate::{
    action::DiscreteAction, agent::Agent, environment::EpisodicEnvironment, state::DiscreteState,
    trajectory::Trajectory,
};

use super::PolicyEstimator;

trait MonteCarlo<
    AC: DiscreteAction,
    S: DiscreteState,
    AG: Agent<Action = AC, State = S>,
    E: EpisodicEnvironment<Agent = AG>,
>: PolicyEstimator<Environment = E>
{
    fn generate_trajectory(agent: &mut E::Agent) -> Vec<Trajectory<S, AC>> {
        let mut environment = E::start_episode();
        let mut trajectory = vec![];

        while let Some(state) = environment.get_agent_state(agent.borrow_mut()) {
            let action = (agent.borrow_mut()).act(&state);
            let reward = environment.receive_agent_action(agent.borrow_mut(), &action);

            trajectory.push(Trajectory::Step {
                state,
                action,
                reward,
            });
        }
        trajectory.push(Trajectory::Final {
            state: environment.final_state(),
        });

        trajectory
    }

    fn print_state_action_pairs(header: &str, list: &[f64]) {
        println!("{header}");
        for (acts, s) in list.chunks(AC::ACTIONS.len() + 1).zip(S::STATES) {
            print!("{s:?} ");
            for (action, value) in AC::ACTIONS.iter().zip(acts) {
                print!("[{action:?}; {value}] ");
            }
            println!("[Final; {:?}]", acts.last().unwrap_or(&0.));
        }
    }

    fn markov_reward_process_state_action_pair_index(trajectory: &Trajectory<S, AC>) -> usize {
        let (state_pos, action_pos) = match trajectory {
            Trajectory::Step {
                state,
                action,
                reward: _,
            } => {
                let state_pos = S::STATES
                    .iter()
                    .position(|const_state| const_state.eq(state));
                let action_pos = AC::ACTIONS
                    .iter()
                    .position(|const_action| const_action.eq(action));
                (state_pos, action_pos)
            }
            Trajectory::Final { state } => {
                let state_pos = S::STATES
                    .iter()
                    .position(|const_state| const_state.eq(state));
                (state_pos, Some(AC::ACTIONS.len()))
            }
        };
        match (state_pos, action_pos) {
            (Some(state_index), Some(action_index)) => {
                state_index * (AC::ACTIONS.len() + 1) + action_index
            }
            (None, _) => {
                panic!("The Trajectory contains a State that is not present on the list of possible States")
            }
            (_, None) => {
                panic!("The Trajectory contains an Action that is not present on the list of possible Actions")
            }
        }
    }
}
