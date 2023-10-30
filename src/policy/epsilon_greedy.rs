use std::marker::PhantomData;

use crate::{
    action::DiscreteAction, random_number_generator::RandomNumberGeneratorFacade,
    state::DiscreteState,
};

use super::{Policy, PolicyError};

pub struct EpsilonGreedyPolicy<A: DiscreteAction, S: DiscreteState> {
    epsilon: f64,
    rng_facade: Box<dyn RandomNumberGeneratorFacade>,
    state_action_correspondece: Vec<A>,
    state_phantom: PhantomData<S>,
}

impl<A: DiscreteAction, S: DiscreteState> EpsilonGreedyPolicy<A, S> {
    pub fn new(
        epsilon: f64,
        rng_facade: Box<dyn RandomNumberGeneratorFacade>,
    ) -> Result<Self, PolicyError> {
        if (0.0f64..1.0).contains(&epsilon) {
            let random_start = S::STATES
                .iter()
                .map(|_| A::ACTIONS[(rng_facade.random() * A::ACTIONS.len() as f64) as usize])
                .collect();
            Ok(Self {
                epsilon,
                rng_facade,
                state_action_correspondece: random_start,
                state_phantom: PhantomData,
            })
        } else {
            Err(PolicyError::new(format!(
                "EpsilonGreedyPolicy requires an `epsilon` between `0.0` and `1.0`. Received {}",
                epsilon
            )))
        }
    }
}

impl<A: DiscreteAction, S: DiscreteState> Policy for EpsilonGreedyPolicy<A, S> {
    type State = S;
    type Action = A;

    fn act(&self, state: impl std::borrow::Borrow<Self::State>) -> Self::Action {
        if self.rng_facade.random().lt(&self.epsilon) {
            A::ACTIONS[(self.rng_facade.random() * A::ACTIONS.len() as f64) as usize]
        } else {
            S::STATES
                .iter()
                .zip(self.state_action_correspondece.iter())
                .find_map(|(s, action)| Some(action).cloned().filter(|_| s.eq(state.borrow())))
                .expect("All states must have a corresponding action.")
        }
    }

    fn policy_improvemnt(&mut self, value_function: impl Fn(&Self::State, &Self::Action) -> f64) {
        self.state_action_correspondece
            .iter_mut()
            .zip(Self::State::STATES.iter())
            .for_each(|(policy_action, state)| {
                let max_action = Self::Action::ACTIONS.iter().max_by(|lhs, rhs| {
                    value_function(state, lhs).total_cmp(&value_function(state, rhs))
                });
                if let Some(best_action) = max_action {
                    *policy_action = *best_action;
                } else {
                    panic!("Could not determinate best action for State {state:?}")
                }
            })
    }
}
