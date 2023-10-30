use std::{fmt::Debug, hash::Hash};

/// Trait for types that represent the [State] of an [Environment]
pub trait State
where
    Self: Sized,
{
}

/// Marker Struct for Discrete States
pub trait DiscreteState: State
where
    Self: 'static + Sized + Debug + Copy + PartialEq + Eq + Hash,
{
    const STATES: &'static [Self];
}

/// Marker Struct for Continuous State
pub trait ContinuousState: State {}
