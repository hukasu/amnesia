use std::borrow::Borrow;

pub trait Agent
where
    Self: Sized,
{
    type Action: crate::action::Action;
    type State: crate::state::State;

    fn act(&self, state: impl Borrow<Self::State>) -> Self::Action;

    fn policy_improvemnt(&mut self, value_function: impl Fn(&Self::State, &Self::Action) -> f64);
}
