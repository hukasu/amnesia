use std::{collections::VecDeque, marker::PhantomData};

use crate::{
    action::DiscreteAction,
    agent::Agent,
    environment::{Environment, EpisodicEnvironment},
    observation::DiscreteObservation,
    reinforcement_learning::PolicyEstimator,
    trajectory::Trajectory,
};

pub struct QLearning<E: EpisodicEnvironment> {
    episode_limit: usize,
    learning_rate: f64,
    discount_factor: f64,
    phantom_env: PhantomData<E>,
}

impl<
        AC: DiscreteAction,
        S: DiscreteObservation,
        AG: Agent<Action = AC, Observation = S>,
        E: EpisodicEnvironment<Agent = AG>,
    > QLearning<E>
{
    pub fn new(episode_limit: usize, learning_rate: f64, discount_factor: f64) -> Self {
        Self {
            episode_limit,
            learning_rate,
            discount_factor,
            phantom_env: PhantomData,
        }
    }

    fn q_learning_update(
        &self,
        agent: &mut AG,
        action_value: &mut [f64],
        (s, a, r, s_): (&S, &AC, f64, Option<&S>),
    ) -> f64 {
        let prev_index = Self::tabular_index(s, a);
        let cur = match s_ {
            Some(state) => AC::ACTIONS
                .iter()
                .map(|discrete_action| {
                    let index = Self::tabular_index(state, discrete_action);
                    action_value[index]
                })
                .max_by(|lhs, rhs| lhs.total_cmp(rhs))
                .expect("There must be a action with maximum value."),
            None => 0.,
        };

        let old_value = action_value[prev_index];
        action_value[prev_index] = action_value[prev_index]
            + self.learning_rate * (r + self.discount_factor * cur - action_value[prev_index]);

        agent.policy_improvemnt(&|state: &S, action: &AC| -> f64 {
            let index = Self::tabular_index(state, action);
            action_value[index]
        });

        (old_value - action_value[prev_index]).powi(2)
    }

    fn tabular_index(
        state: &<E::Agent as Agent>::Observation,
        action: &<E::Agent as Agent>::Action,
    ) -> usize {
        <S>::OBSERVATIONS
            .iter()
            .position(|discrete_observation| discrete_observation.eq(state))
            .expect("State is not present on the list of Observations.")
            * <AC>::ACTIONS.len()
            + <<E::Agent as Agent>::Action as DiscreteAction>::ACTIONS
                .iter()
                .position(|discrete_action| discrete_action.eq(action))
                .expect("Action is not present on the list of Actions.")
    }

    fn print_observation_action_pairs(header: &str, list: &[f64]) {
        println!("{header}");
        for (acts, s) in list.chunks(AC::ACTIONS.len()).zip(S::OBSERVATIONS) {
            print!("{s:?} ");
            for (action, value) in AC::ACTIONS.iter().zip(acts) {
                print!("[{action:?}; {value}] ");
            }
            println!();
        }
    }
}

impl<
        AC: DiscreteAction,
        S: DiscreteObservation,
        AG: Agent<Action = AC, Observation = S>,
        E: EpisodicEnvironment<Agent = AG>,
    > PolicyEstimator for QLearning<E>
{
    type Environment = E;

    fn policy_search(
        self,
        environment: &mut Self::Environment,
        agent: &mut <Self::Environment as Environment>::Agent,
    ) {
        const TEMPORAL_DIFFERENCE_STEP: usize = 2;

        let mut action_value = vec![0.; S::OBSERVATIONS.len() * AC::ACTIONS.len()];

        let mut episode = 0usize;
        let mut episode_variation_window = VecDeque::from_iter([f64::MAX; 5]);
        while episode_variation_window
            .iter()
            .any(|ep_v| ep_v > &f64::EPSILON)
            && episode < self.episode_limit
        {
            episode += 1;
            let mut episode_variation = 0.;

            environment.reset_environment();

            let mut temporal_difference = VecDeque::with_capacity(TEMPORAL_DIFFERENCE_STEP);
            while let Some(observation) = environment.get_observation(agent) {
                let action = agent.act(&observation);
                let reward = environment.receive_action(agent, &action);

                if temporal_difference.len() < 2 {
                    temporal_difference.push_back(Trajectory::Step {
                        observation,
                        action,
                        reward,
                    });
                } else {
                    temporal_difference.pop_front();
                    temporal_difference.push_back(Trajectory::Step {
                        observation,
                        action,
                        reward,
                    });

                    let first = temporal_difference.get(0).expect("There should be enough Steps on the trajectory to calculate the Temporal Difference.");
                    let last = temporal_difference.get(TEMPORAL_DIFFERENCE_STEP - 1).expect("There should be enough Steps on the trajectory to calculate the Temporal Difference.");

                    match (first, last) {
                        (
                            Trajectory::Step {
                                observation,
                                action,
                                reward,
                            },
                            Trajectory::Step {
                                observation: obs2,
                                action: _,
                                reward: _,
                            },
                        ) => {
                            episode_variation += self.q_learning_update(
                                agent,
                                &mut action_value,
                                (observation, action, *reward, Some(obs2)),
                            );
                        }
                        _ => panic!("A final state shouldn't be reached at this point"),
                    }
                }
            }
            let _final_state = environment.final_observation(agent);

            for remaining_step in temporal_difference.into_iter() {
                match remaining_step {
                    Trajectory::Step {
                        observation,
                        action,
                        reward,
                    } => {
                        episode_variation += self.q_learning_update(
                            agent,
                            &mut action_value,
                            (&observation,
                            &action,
                            reward,
                            None),
                        );
                    }
                    _ => panic!("A final state shouldn't have been added to the temporal difference sliding window."),
                }
            }

            episode_variation_window.pop_front();
            episode_variation_window.push_back(episode_variation);
        }

        Self::print_observation_action_pairs("Action Value Function", &action_value);
        println!("Iterated for {} episodes.", episode);
    }
}
