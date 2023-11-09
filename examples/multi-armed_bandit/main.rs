use amnesia::{
    action::{Action, DiscreteAction},
    agent::Agent,
    environment::{Environment, EpisodicEnvironment},
    observation::{DiscreteObservation, Observation},
    policy::{epsilon_greedy::EpsilonGreedyPolicy, Policy},
    random_number_generator::RandomNumberGeneratorFacade,
    reinforcement_learning::{
        monte_carlo::ConstantAlphaMonteCarlo,
        monte_carlo::FirstVisitMonteCarlo,
        monte_carlo::{EveryVisitMonteCarlo, IncrementalMonteCarlo},
        q_learning::QLearning,
        sarsa::SARSA,
        PolicyEstimator,
    },
    ValueFunction,
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
        _agent: &Self::Agent,
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
        _agent: &Self::Agent,
        action: &<Self::Agent as Agent>::Action,
    ) -> f64 {
        let bandit_payout_multiplier = match action {
            MultiArmedBanditAction::Bandit1 => 1.,
            MultiArmedBanditAction::Bandit2 => 2.,
            MultiArmedBanditAction::Bandit3 => 3.,
        };
        let rand_value: f64 = rand::random();
        bandit_payout_multiplier * (rand_value * 2.)
    }
}

impl EpisodicEnvironment for Cassino {
    fn reset_environment(&mut self) {
        self.0 = false;
    }

    fn final_observation(&self, _agent: &Self::Agent) -> <Self::Agent as Agent>::Observation {
        MultiArmedBanditObservation::Game
    }
}

struct Player(EpsilonGreedyPolicy<MultiArmedBanditAction, MultiArmedBanditObservation, RandFacade>);
impl Agent for Player {
    type Action = MultiArmedBanditAction;
    type Observation = MultiArmedBanditObservation;

    fn act(&self, observation: &Self::Observation) -> Self::Action {
        self.0.act(observation)
    }

    fn policy_improvemnt(
        &mut self,
        value_function: &ValueFunction<Self::Observation, Self::Action>,
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
    const EPSILON: f64 = 0.05;
    const ALPHA: f64 = 0.05;

    let mut cassino = Cassino(false);

    println!("First Visit Monte Carlo");
    let mut agent = Player(EpsilonGreedyPolicy::new(EPSILON, RandFacade).unwrap());
    FirstVisitMonteCarlo::<Cassino>::new(RETURN_DISCOUNT, EPISODES)
        .policy_search(&mut cassino, &mut agent);

    println!("Every Visit Monte Carlo");
    let mut agent = Player(EpsilonGreedyPolicy::new(EPSILON, RandFacade).unwrap());
    EveryVisitMonteCarlo::<Cassino>::new(RETURN_DISCOUNT, EPISODES)
        .policy_search(&mut cassino, &mut agent);

    println!("Incremental Monte Carlo");
    let mut agent = Player(EpsilonGreedyPolicy::new(EPSILON, RandFacade).unwrap());
    IncrementalMonteCarlo::<Cassino>::new(RETURN_DISCOUNT, EPISODES)
        .policy_search(&mut cassino, &mut agent);

    println!("Constant Alpha Monte Carlo");
    let mut agent = Player(EpsilonGreedyPolicy::new(EPSILON, RandFacade).unwrap());
    ConstantAlphaMonteCarlo::<Cassino>::new(ALPHA, RETURN_DISCOUNT, EPISODES)
        .policy_search(&mut cassino, &mut agent);

    println!("Q-Learning");
    let mut agent = Player(EpsilonGreedyPolicy::new(EPSILON, RandFacade).unwrap());
    QLearning::<Cassino>::new(EPISODES, ALPHA, RETURN_DISCOUNT)
        .policy_search(&mut cassino, &mut agent);

    println!("SARSA");
    let mut agent = Player(EpsilonGreedyPolicy::new(EPSILON, RandFacade).unwrap());
    SARSA::<Cassino>::new(EPISODES, ALPHA, RETURN_DISCOUNT).policy_search(&mut cassino, &mut agent);
}
