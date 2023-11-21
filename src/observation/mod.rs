use std::{fmt::Debug, hash::Hash};

// Docs imports
#[allow(unused_imports)]
use crate::{agent::Agent, environment::Environment};

/// A subset of the [Environment] that is visible to the [Agent].  
///
/// ## Full observability
/// On virtual environments, the [Observation] might be the [Environment] state.
pub trait Observation
where
    Self: Sized,
{
}

/// An [Observation] of the [Environment] that take discrete values.
pub trait DiscreteObservation: Observation
where
    Self: 'static + Sized + Debug + Copy + PartialEq + Eq + Hash,
{
    const OBSERVATIONS: &'static [Self];

    fn index(&self) -> usize {
        Self::OBSERVATIONS
            .iter()
            .position(|observation| observation.eq(self))
            .expect("Observation must exist in Discrete observations.")
    }
}

/// An [Observation] of the [Environment] that take continuous values.
pub trait ContinuousObservation: Observation {}
