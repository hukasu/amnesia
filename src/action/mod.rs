use std::{fmt::Debug, hash::Hash};

/// Trait that defines the [Action]s that an [Agent] can take.
pub trait Action
where
    Self: Sized,
{
}

/// Trait that defines [Action]s with discrete values.
pub trait DiscreteAction: Action
where
    Self: 'static + Sized + Debug + Copy + PartialEq + Eq + Hash,
{
    const ACTIONS: &'static [Self];
}

/// Trait that defines [Action]s with continuous values.
pub trait ContinuousAction: Action {}
