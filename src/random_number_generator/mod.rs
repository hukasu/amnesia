/// A Facade for an object that can generate random numbers.
pub trait RandomNumberGeneratorFacade {
    /// Returns a number between `0.0f64` and `1.0f64`
    fn random(&self) -> f64;
}
