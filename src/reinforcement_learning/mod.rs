use std::borrow::Borrow;

use crate::{agent::Agent, environment::Environment, trajectory::Trajectory};

pub mod monte_carlo;

pub trait PolicyEstimator {
    type Environment: crate::environment::Environment;

    fn policy_search(self, agent: &mut <Self::Environment as Environment>::Agent);

    /// Calculates the returns of a list of rewards using the equation
    /// `G{t,i} = r{t,i} + γ r{t+1,i} + γ^2 r{t+2,i} + ... +  γ^{Ti-1} r{Ti,i}` where
    /// `i` is an episode, `t` is the time step of the episode `i`,
    /// `Ti` is the last step of the episode `i`, and `γ` is the return discount.
    fn calculate_return<
        I: Borrow<
            Trajectory<
                <<Self::Environment as Environment>::Agent as Agent>::Observation,
                <<Self::Environment as Environment>::Agent as Agent>::Action,
            >,
        >,
    >(
        trajectory: impl Iterator<Item = I> + DoubleEndedIterator,
        return_discount: f64,
    ) -> Vec<f64> {
        let rewards = [0.]
            .into_iter()
            .chain(trajectory.filter_map(|step| match step.borrow() {
                Trajectory::Step {
                    observation: _,
                    action: _,
                    reward,
                } => Some(*reward),
                Trajectory::Final { observation: _ } => None,
            }))
            .collect::<Vec<_>>();
        let mut returns = rewards
            .into_iter()
            .rev()
            .enumerate()
            .scan(0., |prev, (i, cur)| {
                *prev += return_discount.powi(i as i32) * cur;
                Some(*prev)
            })
            .collect::<Vec<_>>();
        returns.reverse();
        returns
    }
}
