use crate::agent::Agent;

// Docs imports
#[allow(unused_imports)]
use crate::observation::Observation;

/// [Environment] is an interface between the world and the [Agent].  
///
/// The [Environment] has an internal state that is not visible to the [Agent]
/// or too big to be processable, so the [Agent] acts over [Observation]s of the
/// [Environment].
pub trait Environment {
    type Agent: crate::agent::Agent;

    /// Generates an [Observation] of the [Environment] in the point of view
    /// of the [Agent].
    fn get_observation(
        &mut self,
        agent: &Self::Agent,
    ) -> Option<<Self::Agent as Agent>::Observation>;

    fn receive_action(
        &mut self,
        agent: &Self::Agent,
        action: &<Self::Agent as Agent>::Action,
    ) -> f64;
}

pub trait EpisodicEnvironment: Environment {
    /// Reset [`EpisodicEnvironment`] to an initial state.
    fn reset_environment(&mut self);

    /// Get the [Observation] of the terminal state of the [`EpisodicEnvironment`].
    fn final_observation(&self, agent: &Self::Agent) -> <Self::Agent as Agent>::Observation;
}
