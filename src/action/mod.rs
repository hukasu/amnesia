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

    fn index(&self) -> usize {
        Self::ACTIONS
            .iter()
            .position(|action| action.eq(self))
            .expect("Action must exist in Discrete actions.")
    }
}

/// Trait that defines [Action]s with continuous values.
pub trait ContinuousAction: Action {}
