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
enum MultiArmedBanditState {
    Game,
}

impl State for MultiArmedBanditState {}

impl DiscreteState for MultiArmedBanditState {
    const STATES: &'static [Self] = &[Self::Game];
}

struct Cassino(bool);

impl Environment for Cassino {
    type Agent = Player;

    fn get_agent_state(
        &mut self,
        _agent: impl std::borrow::Borrow<Self::Agent>,
    ) -> Option<<Self::Agent as Agent>::State> {
        if !self.0 {
            self.0 = true;
            Some(MultiArmedBanditState::Game)
        } else {
            None
        }
    }

    fn receive_agent_action(
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

    fn final_state(&self) -> <Self::Agent as Agent>::State {
        MultiArmedBanditState::Game
    }
}

struct Player(EpsilonGreedyPolicy<MultiArmedBanditAction, MultiArmedBanditState>);
impl Agent for Player {
    type Action = MultiArmedBanditAction;
    type State = MultiArmedBanditState;

    fn act(&self, state: impl std::borrow::Borrow<Self::State>) -> Self::Action {
        self.0.act(state)
    }

    fn policy_improvemnt(&mut self, value_function: impl Fn(&Self::State, &Self::Action) -> f64) {
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
