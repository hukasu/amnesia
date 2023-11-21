use crate::{
    action::DiscreteAction, agent::Agent, environment::Environment,
    observation::DiscreteObservation, trajectory::Trajectory,
};

pub mod monte_carlo;
pub mod temporal_difference;

pub trait PolicyEstimator {
    type Environment: crate::environment::Environment;

    fn policy_search(
        self,
        environment: &mut Self::Environment,
        agent: &mut <Self::Environment as Environment>::Agent,
    );

    /// Calculates the returns of a list of rewards using the equation
    /// `G{t,i} = r{t,i} + γ r{t+1,i} + γ^2 r{t+2,i} + ... +  γ^{Ti-1} r{Ti,i}` where
    /// `i` is an episode, `t` is the time step of the episode `i`,
    /// `Ti` is the last step of the episode `i`, and `γ` is the return discount.
    #[allow(clippy::type_complexity)]
    fn discounted_return(
        trajectory: &Vec<
            Trajectory<
                <<Self::Environment as Environment>::Agent as Agent>::Observation,
                <<Self::Environment as Environment>::Agent as Agent>::Action,
            >,
        >,
        return_discount: f64,
        episode_returns: &mut Vec<f64>,
    ) {
        episode_returns.resize(trajectory.len() - 1, 0.);

        trajectory
            .iter()
            .filter_map(|step| match step {
                Trajectory::Step {
                    observation: _,
                    action: _,
                    reward,
                } => Some(*reward),
                Trajectory::Final { observation: _ } => None,
            })
            .rev()
            .scan(0., |prev_return, reward| {
                let current_return = reward + return_discount * *prev_return;
                *prev_return = current_return;
                Some(current_return)
            })
            .zip(episode_returns.iter_mut().rev())
            .for_each(|(epi_return, out_return)| {
                *out_return = epi_return;
            });
    }
}

trait DiscretePolicyEstimator<
    AC: DiscreteAction,
    S: DiscreteObservation,
    AG: Agent<Action = AC, Observation = S>,
    E: Environment<Agent = AG>,
>: PolicyEstimator<Environment = E>
{
    fn tabular_index(action: &AC, observation: &S) -> usize {
        observation.index() * AC::ACTIONS.len() + action.index()
    }
}

impl<
        AC: DiscreteAction,
        S: DiscreteObservation,
        AG: Agent<Action = AC, Observation = S>,
        E: Environment<Agent = AG>,
        PE: ?Sized + PolicyEstimator<Environment = E>,
    > DiscretePolicyEstimator<AC, S, AG, E> for PE
{
}
