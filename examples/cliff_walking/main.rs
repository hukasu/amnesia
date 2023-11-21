use amnesia::{
    action::{Action, DiscreteAction},
    agent::{Agent, DiscreteAgent},
    environment::{Environment, EpisodicEnvironment},
    observation::{DiscreteObservation, Observation},
    policy::{epsilon_greedy::EpsilonGreedyPolicy, Policy},
    random_number_generator::RandomNumberGeneratorFacade,
    reinforcement_learning::{
        monte_carlo::{
            ConstantAlphaMonteCarlo, EveryVisitMonteCarlo, FirstVisitMonteCarlo,
            IncrementalMonteCarlo,
        },
        temporal_difference::{ExpectedSARSA, QLearning, SARSA},
        PolicyEstimator,
    },
};

const LEN: usize = 12;

const fn build_cliff_path() -> [CliffPath; LEN * 4] {
    let mut cliff = [CliffPath(0, 0); LEN * 4];
    let mut x = 0;
    let mut y = 0;
    while y < 4 {
        while x < LEN {
            cliff[y * LEN + x] = CliffPath(x, y);
            x += 1;
        }
        x = 0;
        y += 1;
    }
    cliff
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CliffPath(usize, usize);

impl Observation for CliffPath {}

impl DiscreteObservation for CliffPath {
    const OBSERVATIONS: &'static [Self] = &build_cliff_path();
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Walk {
    Up,
    Down,
    Left,
    Right,
}

impl Action for Walk {}

impl DiscreteAction for Walk {
    const ACTIONS: &'static [Self] = &[Self::Up, Self::Down, Self::Left, Self::Right];
}

struct CliffWalker {
    policy: EpsilonGreedyPolicy<Walk, CliffPath, RandFacade>,
}

impl Agent for CliffWalker {
    type Action = Walk;
    type Observation = CliffPath;

    fn act(&self, observation: &Self::Observation) -> Self::Action {
        self.policy.act(observation)
    }

    fn policy_improvemnt(
        &mut self,
        action: &Self::Action,
        observation: &Self::Observation,
        value: f64,
    ) {
        self.policy.policy_improvemnt(action, observation, value);
    }
}

impl DiscreteAgent<Walk, CliffPath> for CliffWalker {
    fn action_probability(&self, action: &Walk, observation: &CliffPath) -> f64 {
        if self.act(observation).eq(action) {
            1.
        } else {
            0.
        }
    }
}

struct Cliff {
    walker_position: CliffPath,
    steps_taken: usize,
}

impl Environment for Cliff {
    type Agent = CliffWalker;

    fn get_observation(
        &mut self,
        _agent: &Self::Agent,
    ) -> Option<<Self::Agent as amnesia::agent::Agent>::Observation> {
        match self.walker_position {
            CliffPath(x, 0) if x != 0 => None,
            _ => Some(self.walker_position).filter(|_| self.steps_taken < 500),
        }
    }

    fn receive_action(
        &mut self,
        _agent: &Self::Agent,
        action: &<Self::Agent as Agent>::Action,
    ) -> f64 {
        self.steps_taken += 1;

        if self.steps_taken < 500 {
            let (delta_x, delta_y) = match action {
                Walk::Up => (0isize, 1isize),
                Walk::Down => (0, -1),
                Walk::Left => (-1, 0),
                Walk::Right => (1, 0),
            };

            self.walker_position = {
                let CliffPath(x, y) = self.walker_position;
                CliffPath(
                    x.saturating_add_signed(delta_x).min(LEN - 1),
                    y.saturating_add_signed(delta_y).min(4 - 1),
                )
            };
            match self.walker_position {
                CliffPath(x, 0) if x == LEN - 1 => 10.,
                CliffPath(x, 0) if x != 0 => -100.,
                _ => -1.,
            }
        } else {
            -100.
        }
    }
}

impl EpisodicEnvironment for Cliff {
    fn reset_environment(&mut self) {
        self.walker_position = CliffPath(0, 0);
        self.steps_taken = 0;
    }

    fn final_observation(&self, _agent: &Self::Agent) -> <Self::Agent as Agent>::Observation {
        self.walker_position
    }
}

struct RandFacade;

impl RandomNumberGeneratorFacade for RandFacade {
    fn random(&self) -> f64 {
        rand::random()
    }
}

fn main() {
    const EPISODES: usize = 1000000;
    const RETURN_DISCOUNT: f64 = 1. - (1. / 16.);
    const ALPHA: f64 = 1. / 64.;
    const EPSILON: f64 = 0.1;

    let mut cliff = Cliff {
        walker_position: CliffPath(0, 0),
        steps_taken: 0,
    };

    println!("First Visit Monte Carlo");
    let mut agent = CliffWalker {
        policy: EpsilonGreedyPolicy::new(EPSILON, RandFacade).unwrap(),
    };
    FirstVisitMonteCarlo::<Cliff>::new(RETURN_DISCOUNT, EPISODES)
        .policy_search(&mut cliff, &mut agent);

    println!("Every Visit Monte Carlo");
    let mut agent = CliffWalker {
        policy: EpsilonGreedyPolicy::new(EPSILON, RandFacade).unwrap(),
    };
    EveryVisitMonteCarlo::<Cliff>::new(RETURN_DISCOUNT, EPISODES)
        .policy_search(&mut cliff, &mut agent);

    println!("Incremental Monte Carlo");
    let mut agent = CliffWalker {
        policy: EpsilonGreedyPolicy::new(EPSILON, RandFacade).unwrap(),
    };
    IncrementalMonteCarlo::<Cliff>::new(RETURN_DISCOUNT, EPISODES)
        .policy_search(&mut cliff, &mut agent);

    println!("Constant Alpha Monte Carlo");
    let mut agent = CliffWalker {
        policy: EpsilonGreedyPolicy::new(EPSILON, RandFacade).unwrap(),
    };
    ConstantAlphaMonteCarlo::<Cliff>::new(ALPHA, RETURN_DISCOUNT, EPISODES)
        .policy_search(&mut cliff, &mut agent);

    println!("Q-Learning");
    let mut agent = CliffWalker {
        policy: EpsilonGreedyPolicy::new(EPSILON, RandFacade).unwrap(),
    };
    QLearning::<Cliff>::new(EPISODES, ALPHA, RETURN_DISCOUNT).policy_search(&mut cliff, &mut agent);

    println!("SARSA");
    let mut agent = CliffWalker {
        policy: EpsilonGreedyPolicy::new(EPSILON, RandFacade).unwrap(),
    };
    SARSA::<Cliff>::new(EPISODES, ALPHA, RETURN_DISCOUNT).policy_search(&mut cliff, &mut agent);

    println!("ExpectedSARSA");
    let mut agent = CliffWalker {
        policy: EpsilonGreedyPolicy::new(EPSILON, RandFacade).unwrap(),
    };
    ExpectedSARSA::<Cliff>::new(EPISODES, ALPHA, RETURN_DISCOUNT)
        .policy_search(&mut cliff, &mut agent);
}
