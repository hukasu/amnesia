use std::borrow::Borrow;

use amnesia::{
    action::{Action, DiscreteAction},
    agent::Agent,
    environment::{Environment, EpisodicEnvironment},
    policy::{epsilon_greedy::EpsilonGreedyPolicy, Policy},
    random_number_generator::RandomNumberGeneratorFacade,
    reinforcement_learning::{
        monte_carlo::ConstantAlphaMonteCarlo, monte_carlo::EveryVisitMonteCarlo,
        monte_carlo::FirstVisitMonteCarlo, PolicyEstimator,
    },
    state::{DiscreteState, State},
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

impl State for MarsSpace {}

impl DiscreteState for MarsSpace {
    const STATES: &'static [Self] = &[
        Self::S1,
        Self::S2,
        Self::S3,
        Self::S4,
        Self::S5,
        Self::S6,
        Self::S7,
    ];
}

struct Rover(EpsilonGreedyPolicy<RoverAction, MarsSpace>);

impl Agent for Rover {
    type Action = RoverAction;
    type State = MarsSpace;

    fn act(&self, state: impl std::borrow::Borrow<Self::State>) -> Self::Action {
        self.0.act(state)
    }

    fn policy_improvemnt(&mut self, value_function: impl Fn(&Self::State, &Self::Action) -> f64) {
        self.0.policy_improvemnt(value_function);
    }
}

struct Mars {
    rover_state: MarsSpace,
}

impl Environment for Mars {
    type Agent = Rover;

    fn get_agent_state(
        &mut self,
        _agent: impl Borrow<Self::Agent>,
    ) -> Option<<Self::Agent as Agent>::State> {
        match self.rover_state {
            MarsSpace::S1 => None,
            MarsSpace::S7 => None,
            _ => Some(self.rover_state),
        }
    }

    fn receive_agent_action(
        &mut self,
        _agent: impl Borrow<Self::Agent>,
        action: impl Borrow<<Self::Agent as Agent>::Action>,
    ) -> f64 {
        self.rover_state = match action.borrow() {
            RoverAction::MoveLeft => match self.rover_state {
                MarsSpace::S1 => MarsSpace::S1,
                MarsSpace::S2 => MarsSpace::S1,
                MarsSpace::S3 => MarsSpace::S2,
                MarsSpace::S4 => MarsSpace::S3,
                MarsSpace::S5 => MarsSpace::S4,
                MarsSpace::S6 => MarsSpace::S5,
                MarsSpace::S7 => MarsSpace::S6,
            },
            RoverAction::MoveRight => match self.rover_state {
                MarsSpace::S1 => MarsSpace::S2,
                MarsSpace::S2 => MarsSpace::S3,
                MarsSpace::S3 => MarsSpace::S4,
                MarsSpace::S4 => MarsSpace::S5,
                MarsSpace::S5 => MarsSpace::S6,
                MarsSpace::S6 => MarsSpace::S7,
                MarsSpace::S7 => MarsSpace::S7,
            },
        };
        match self.rover_state {
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
    fn start_episode() -> Self {
        Self {
            rover_state: MarsSpace::S4,
        }
    }

    fn final_state(&self) -> <Self::Agent as Agent>::State {
        self.rover_state
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
    const ALPHA: f64 = 0.1;

    println!("First Visit Monte Carlo");
    let random = RandFacade;
    let mut agent = Rover(EpsilonGreedyPolicy::new(0.05, Box::new(random)).unwrap());
    FirstVisitMonteCarlo::<Mars>::new(RETURN_DISCOUNT, EPISODES).policy_search(&mut agent);

    println!("Every Visit Monte Carlo");
    let random = RandFacade;
    let mut agent = Rover(EpsilonGreedyPolicy::new(0.05, Box::new(random)).unwrap());
    EveryVisitMonteCarlo::<Mars>::new(RETURN_DISCOUNT, EPISODES).policy_search(&mut agent);

    println!("Constant Alpha Visit Monte Carlo");
    let random = RandFacade;
    let mut agent = Rover(EpsilonGreedyPolicy::new(0.05, Box::new(random)).unwrap());
    ConstantAlphaMonteCarlo::<Mars>::new(ALPHA, RETURN_DISCOUNT, EPISODES)
        .policy_search(&mut agent);
}
