

export PhyHermite

# Create a structure to hold physicist Hermite polynomials as well as their first and second derivative
struct PhyHermite{m} <: Hermite
    P::ImmutablePolynomial
    Pprime::ImmutablePolynomial
    Ppprime::ImmutablePolynomial
    scale::Bool
end
