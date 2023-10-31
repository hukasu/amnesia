use crate::{action::Action, observation::Observation};

pub enum Trajectory<S: Observation, A: Action> {
    Step {
        observation: S,
        action: A,
        reward: f64,
    },
    Final {
        observation: S,
    },
}
