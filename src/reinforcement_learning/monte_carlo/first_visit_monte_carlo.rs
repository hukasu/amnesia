use std::{borrow::BorrowMut, marker::PhantomData};

use crate::{
    action::DiscreteAction,
    agent::Agent,
    environment::{Environment, EpisodicEnvironment},
    state::DiscreteState,
    trajectory::Trajectory,
};

use crate::reinforcement_learning::PolicyEstimator;

pub struct FirstVisitMonteCarlo<E: EpisodicEnvironment> {
    return_discount: f64,
    episodes: usize,
    phantom_environment: PhantomData<E>,
}

impl<
        AC: DiscreteAction,
        S: DiscreteState,
        AG: Agent<Action = AC, State = S>,
        E: EpisodicEnvironment<Agent = AG>,
    > FirstVisitMonteCarlo<E>
{
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
        S: DiscreteState,
        AG: Agent<Action = AC, State = S>,
        E: EpisodicEnvironment<Agent = AG>,
    > PolicyEstimator for FirstVisitMonteCarlo<E>
{
    type Environment = E;

    fn policy_search(self, agent: &mut <Self::Environment as Environment>::Agent) {
        let mut visit_count = vec![0usize; S::STATES.len() * (AC::ACTIONS.len() + 1)];
        let mut returns = vec![0.0f64; S::STATES.len() * (AC::ACTIONS.len() + 1)];
        let mut state_values = vec![0.0f64; S::STATES.len() * (AC::ACTIONS.len() + 1)];

        for _episode in 0..self.episodes {
            let mut environment = E::start_episode();
            let mut trajectory = vec![];

            while let Some(state) = environment.get_agent_state(agent.borrow_mut()) {
                let action = agent.borrow_mut().act(&state);
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

            let mut visited: Vec<bool> = vec![false; S::STATES.len()];
            let episode_returns = Self::calculate_return(trajectory.iter(), self.return_discount);

            assert_eq!(trajectory.len(), episode_returns.len());

            for (step, g_t) in trajectory.iter().zip(episode_returns.iter()) {
                let (state_pos, action_pos) = match step {
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
                        let composite_index = state_index * (AC::ACTIONS.len() + 1) + action_index;
                        if !visited[state_index] {
                            visited[state_index] = true;
                            visit_count[composite_index] += 1;
                            returns[composite_index] += g_t;
                            state_values[composite_index] =
                                returns[composite_index] / visit_count[composite_index] as f64;
                        }
                    }
                    (None, _) => {
                        panic!("The Trajectory contains a State that is not present on the list of possible States")
                    }
                    (_, None) => {
                        panic!("The Trajectory contains an Action that is not present on the list of possible Actions")
                    }
                }
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
        let print_state_action_pairs = |list: &[f64]| {
            for (acts, s) in list.chunks(AC::ACTIONS.len() + 1).zip(S::STATES) {
                print!("{s:?} ");
                for (action, value) in AC::ACTIONS.iter().zip(acts) {
                    print!("[{action:?}; {value}] ");
                }
                println!("[Final; {:?}]", acts.last().unwrap_or(&0.));
            }
        };
        println!("State Visit Count");
        print_state_action_pairs(&visit_count.iter().map(|u| *u as f64).collect::<Vec<_>>());
        println!("Returns");
        print_state_action_pairs(&returns);
        println!("Action Value Function");
        print_state_action_pairs(&state_values);
    }
}
