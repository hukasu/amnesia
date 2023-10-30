use std::marker::PhantomData;

use crate::{
    action::DiscreteAction, agent::Agent, environment::EpisodicEnvironment, state::DiscreteState,
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
        S: DiscreteState,
        AG: Agent<Action = AC, State = S>,
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
        S: DiscreteState,
        AG: Agent<Action = AC, State = S>,
        E: EpisodicEnvironment<Agent = AG>,
    > PolicyEstimator for ConstantAlphaMonteCarlo<E>
{
    type Environment = E;

    fn policy_search(self, agent: &mut E::Agent) {
        let mut visit_count = vec![0usize; S::STATES.len() * (AC::ACTIONS.len() + 1)];
        let mut state_values = vec![0.0f64; S::STATES.len() * (AC::ACTIONS.len() + 1)];

        for _episode in 0..self.episodes {
            let trajectory = Self::generate_trajectory(agent);
            let episode_returns = Self::calculate_return(trajectory.iter(), self.return_discount);

            for (step, g_t) in trajectory.iter().zip(episode_returns.iter()) {
                let composite_index = Self::markov_reward_process_state_action_pair_index(step);
                visit_count[composite_index] += 1;
                state_values[composite_index] = state_values[composite_index]
                    + self.alpha * (g_t - state_values[composite_index]);
            }

            let value_function = |state: &S, action: &AC| {
                let state_pos = S::STATES
                    .iter()
                    .position(|const_state| const_state.eq(state));
                let action_pos = AC::ACTIONS
                    .iter()
                    .position(|const_action| const_action.eq(action));
                match (state_pos, action_pos) {
                    (Some(state_index), Some(action_index)) => {
                        let composite_index = state_index * (AC::ACTIONS.len() + 1) + action_index;
                        state_values[composite_index]
                    }
                    (None, _) => {
                        panic!("The Trajectory contains a State that is not present on the list of possible States")
                    }
                    (_, None) => {
                        panic!("The Trajectory contains an Action that is not present on the list of possible Actions")
                    }
                }
            };
            agent.policy_improvemnt(value_function);
        }

        Self::print_state_action_pairs(
            "State Visit Count",
            &visit_count.iter().map(|u| *u as f64).collect::<Vec<_>>(),
        );
        Self::print_state_action_pairs("Action Value Function", &state_values);
    }
}

impl<
        AC: DiscreteAction,
        S: DiscreteState,
        AG: Agent<Action = AC, State = S>,
        E: EpisodicEnvironment<Agent = AG>,
    > MonteCarlo<AC, S, AG, E> for ConstantAlphaMonteCarlo<E>
{
}
