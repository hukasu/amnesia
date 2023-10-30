use crate::{action::Action, state::State};

pub enum Trajectory<S: State, A: Action> {
    Step { state: S, action: A, reward: f64 },
    Final { state: S },
}
