export RK, r₁, RKParams, rk4

# Functions that get extended by individual systems
function r₁ end

struct RKParams{N}
  c::Vector{Float64}
  a::Matrix{Float64}
end

const RK4 = RKParams{4}([0.5, 0.5, 1.0, 1.0],
        [1/2   0    0    0
          0   1/2   0    0
          0    0    1    0
         1/6  1/3  1/3  1/6])


const RK31 = RKParams{3}([0.5, 1.0, 1.0],
                      [1/2        0        0
                       √3/3 (3-√3)/3        0
                       (3+√3)/6    -√3/3 (3+√3)/6])

const Euler = RKParams{1}([1.0],ones(1,1))


function rk4(f::Function, x, t::Float64, dt::Float64)
        # Compute intermediary values for k
        k1 = f(t, x);
        k2 = f(t + dt/2, x + dt/2*k1);
        k3 = f(t + dt/2, x + dt/2*k2);
        k4 = f(t + dt, x + dt*k3);
        # Compute updated values for u and t
        xkp1 = x + dt/6*(k1 + 2*k2 + 2*k3 + k4);
        return t+dt, xkp1
end

include("rk.jl")
