use std::borrow::Borrow;

use crate::agent::Agent;

/// Environment is an interface between the world and the [Agent].
pub trait Environment {
    type Agent: crate::agent::Agent;

    fn get_agent_state(
        &mut self,
        agent: impl Borrow<Self::Agent>,
    ) -> Option<<Self::Agent as Agent>::State>;

    fn receive_agent_action(
        &mut self,
        agent: impl Borrow<Self::Agent>,
        action: impl Borrow<<Self::Agent as Agent>::Action>,
    ) -> f64;
}

pub trait EpisodicEnvironment: Environment {
    fn start_episode() -> Self;
    fn final_state(&self) -> <Self::Agent as Agent>::State;
}
