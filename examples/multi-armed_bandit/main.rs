use amnesia::{
    action::{Action, DiscreteAction},
    agent::Agent,
    environment::{Environment, EpisodicEnvironment},
    observation::{DiscreteObservation, Observation},
    policy::{epsilon_greedy::EpsilonGreedyPolicy, Policy},
    random_number_generator::RandomNumberGeneratorFacade,
    reinforcement_learning::{
        monte_carlo::ConstantAlphaMonteCarlo, monte_carlo::EveryVisitMonteCarlo,
        monte_carlo::FirstVisitMonteCarlo, PolicyEstimator,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MultiArmedBanditAction {
    Bandit1,
    Bandit2,
    Bandit3,
}

impl Action for MultiArmedBanditAction {}

impl DiscreteAction for MultiArmedBanditAction {
    const ACTIONS: &'static [Self] = &[Self::Bandit1, Self::Bandit2, Self::Bandit3];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MultiArmedBanditObservation {
    Game,
}

impl Observation for MultiArmedBanditObservation {}

impl DiscreteObservation for MultiArmedBanditObservation {
    const OBSERVATIONS: &'static [Self] = &[Self::Game];
}

struct Cassino(bool);

impl Environment for Cassino {
    type Agent = Player;

    fn get_observation(
        &mut self,
        _agent: impl std::borrow::Borrow<Self::Agent>,
    ) -> Option<<Self::Agent as Agent>::Observation> {
        if !self.0 {
            self.0 = true;
            Some(MultiArmedBanditObservation::Game)
        } else {
            None
        }
    }

    fn receive_action(
        &mut self,
        _agent: impl std::borrow::Borrow<Self::Agent>,
        action: impl std::borrow::Borrow<<Self::Agent as Agent>::Action>,
    ) -> f64 {
        let bandit_payout_multiplier = match action.borrow() {
            MultiArmedBanditAction::Bandit1 => 1.,
            MultiArmedBanditAction::Bandit2 => 2.,
            MultiArmedBanditAction::Bandit3 => 3.,
        };
        let rand_value: f64 = rand::random();
        bandit_payout_multiplier * (rand_value * 2.)
    }
}

impl EpisodicEnvironment for Cassino {
    fn start_episode() -> Self {
        Cassino(false)
    }

    fn final_observation(&self) -> <Self::Agent as Agent>::Observation {
        MultiArmedBanditObservation::Game
    }
}

struct Player(EpsilonGreedyPolicy<MultiArmedBanditAction, MultiArmedBanditObservation>);
impl Agent for Player {
    type Action = MultiArmedBanditAction;
    type Observation = MultiArmedBanditObservation;

    fn act(&self, observation: impl std::borrow::Borrow<Self::Observation>) -> Self::Action {
        self.0.act(observation)
    }

    fn policy_improvemnt(
        &mut self,
        value_function: impl Fn(&Self::Observation, &Self::Action) -> f64,
    ) {
        self.0.policy_improvemnt(value_function);
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
    const ALPHA: f64 = 0.05;

    println!("First Visit Monte Carlo");
    let random = RandFacade;
    let mut agent = Player(EpsilonGreedyPolicy::new(0.05, Box::new(random)).unwrap());
    FirstVisitMonteCarlo::<Cassino>::new(RETURN_DISCOUNT, EPISODES).policy_search(&mut agent);

    println!("Every Visit Monte Carlo");
    let random = RandFacade;
    let mut agent = Player(EpsilonGreedyPolicy::new(0.05, Box::new(random)).unwrap());
    EveryVisitMonteCarlo::<Cassino>::new(RETURN_DISCOUNT, EPISODES).policy_search(&mut agent);

    println!("Constant Alpha Visit Monte Carlo");
    let random = RandFacade;
    let mut agent = Player(EpsilonGreedyPolicy::new(0.05, Box::new(random)).unwrap());
    ConstantAlphaMonteCarlo::<Cassino>::new(ALPHA, RETURN_DISCOUNT, EPISODES)
        .policy_search(&mut agent);
}
