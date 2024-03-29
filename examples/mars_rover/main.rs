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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoverAction {
    MoveLeft,
    MoveRight,
}

impl Action for RoverAction {}

impl DiscreteAction for RoverAction {
    const ACTIONS: &'static [Self] = &[Self::MoveLeft, Self::MoveRight];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarsSpace {
    S1,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
}

impl Observation for MarsSpace {}

impl DiscreteObservation for MarsSpace {
    const OBSERVATIONS: &'static [Self] = &[
        Self::S1,
        Self::S2,
        Self::S3,
        Self::S4,
        Self::S5,
        Self::S6,
        Self::S7,
    ];
}

struct Rover(EpsilonGreedyPolicy<RoverAction, MarsSpace, RandFacade>);

impl Agent for Rover {
    type Action = RoverAction;
    type Observation = MarsSpace;

    fn act(&self, observation: &Self::Observation) -> Self::Action {
        self.0.act(observation)
    }

    fn policy_improvemnt(
        &mut self,
        action: &Self::Action,
        observation: &Self::Observation,
        value: f64,
    ) {
        self.0.policy_improvemnt(action, observation, value);
    }
}

impl DiscreteAgent<RoverAction, MarsSpace> for Rover {
    fn action_probability(&self, action: &RoverAction, observation: &MarsSpace) -> f64 {
        if self.act(observation).eq(action) {
            1.
        } else {
            0.
        }
    }
}

struct Mars {
    rover_position: MarsSpace,
}

impl Environment for Mars {
    type Agent = Rover;

    fn get_observation(
        &mut self,
        _agent: &Self::Agent,
    ) -> Option<<Self::Agent as Agent>::Observation> {
        match self.rover_position {
            MarsSpace::S1 => None,
            MarsSpace::S7 => None,
            _ => Some(self.rover_position),
        }
    }

    fn receive_action(
        &mut self,
        _agent: &Self::Agent,
        action: &<Self::Agent as Agent>::Action,
    ) -> f64 {
        self.rover_position = match action {
            RoverAction::MoveLeft => match self.rover_position {
                MarsSpace::S1 => MarsSpace::S1,
                MarsSpace::S2 => MarsSpace::S1,
                MarsSpace::S3 => MarsSpace::S2,
                MarsSpace::S4 => MarsSpace::S3,
                MarsSpace::S5 => MarsSpace::S4,
                MarsSpace::S6 => MarsSpace::S5,
                MarsSpace::S7 => MarsSpace::S6,
            },
            RoverAction::MoveRight => match self.rover_position {
                MarsSpace::S1 => MarsSpace::S2,
                MarsSpace::S2 => MarsSpace::S3,
                MarsSpace::S3 => MarsSpace::S4,
                MarsSpace::S4 => MarsSpace::S5,
                MarsSpace::S5 => MarsSpace::S6,
                MarsSpace::S6 => MarsSpace::S7,
                MarsSpace::S7 => MarsSpace::S7,
            },
        };
        match self.rover_position {
            MarsSpace::S1 => 1.,
            MarsSpace::S2 => 0.,
            MarsSpace::S3 => 0.,
            MarsSpace::S4 => 0.,
            MarsSpace::S5 => 0.,
            MarsSpace::S6 => 0.,
            MarsSpace::S7 => 10.,
        }
    }
}

impl EpisodicEnvironment for Mars {
    fn reset_environment(&mut self) {
        self.rover_position = MarsSpace::S4;
    }

    fn final_observation(&self, _agent: &Self::Agent) -> <Self::Agent as Agent>::Observation {
        self.rover_position
    }
}

struct RandFacade;

impl RandomNumberGeneratorFacade for RandFacade {
    fn random(&self) -> f64 {
        rand::random()
    }
}

fn main() {
    const EPISODES: usize = 10000000;
    const RETURN_DISCOUNT: f64 = 1.;
    const EPSILON: f64 = 0.05;
    const ALPHA: f64 = 0.1;

    let mut mars = Mars {
        rover_position: MarsSpace::S4,
    };

    println!("First Visit Monte Carlo");
    let mut agent = Rover(EpsilonGreedyPolicy::new(EPSILON, RandFacade).unwrap());
    FirstVisitMonteCarlo::<Mars>::new(RETURN_DISCOUNT, EPISODES)
        .policy_search(&mut mars, &mut agent);

    println!("Every Visit Monte Carlo");
    let mut agent = Rover(EpsilonGreedyPolicy::new(EPSILON, RandFacade).unwrap());
    EveryVisitMonteCarlo::<Mars>::new(RETURN_DISCOUNT, EPISODES)
        .policy_search(&mut mars, &mut agent);

    println!("Incremental Monte Carlo");
    let mut agent = Rover(EpsilonGreedyPolicy::new(EPSILON, RandFacade).unwrap());
    IncrementalMonteCarlo::<Mars>::new(RETURN_DISCOUNT, EPISODES)
        .policy_search(&mut mars, &mut agent);

    println!("Constant Alpha Monte Carlo");
    let mut agent = Rover(EpsilonGreedyPolicy::new(EPSILON, RandFacade).unwrap());
    ConstantAlphaMonteCarlo::<Mars>::new(ALPHA, RETURN_DISCOUNT, EPISODES)
        .policy_search(&mut mars, &mut agent);

    println!("Q-Learning");
    let mut agent = Rover(EpsilonGreedyPolicy::new(EPSILON, RandFacade).unwrap());
    QLearning::<Mars>::new(EPISODES, ALPHA, RETURN_DISCOUNT).policy_search(&mut mars, &mut agent);

    println!("SARSA");
    let mut agent = Rover(EpsilonGreedyPolicy::new(EPSILON, RandFacade).unwrap());
    SARSA::<Mars>::new(EPISODES, ALPHA, RETURN_DISCOUNT).policy_search(&mut mars, &mut agent);

    println!("ExpectedSARSA");
    let mut agent = Rover(EpsilonGreedyPolicy::new(EPSILON, RandFacade).unwrap());
    ExpectedSARSA::<Mars>::new(EPISODES, ALPHA, RETURN_DISCOUNT)
        .policy_search(&mut mars, &mut agent);
}
