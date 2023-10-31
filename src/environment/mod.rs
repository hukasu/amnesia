use std::borrow::Borrow;

use crate::agent::Agent;

// Docs imports
#[allow(unused_imports)]
use crate::observation::Observation;

/// [Environment] is an interface between the world and the [Agent].
pub trait Environment {
    type Agent: crate::agent::Agent;

    /// Generates an [Observation] of the [Environment] in the point of view
    /// of the [Agent].
    fn get_observation(
        &mut self,
        agent: impl Borrow<Self::Agent>,
    ) -> Option<<Self::Agent as Agent>::Observation>;

    fn receive_action(
        &mut self,
        agent: impl Borrow<Self::Agent>,
        action: impl Borrow<<Self::Agent as Agent>::Action>,
    ) -> f64;
}

pub trait EpisodicEnvironment: Environment {
    fn start_episode() -> Self;
    fn final_observation(&self) -> <Self::Agent as Agent>::Observation;
}
