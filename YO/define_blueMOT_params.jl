using QuantumSimulations

# DEFINE STATES #
energies = energy.(states) .* (2π / Γ)

# DEFINE FREQUENCIES #
detuning = +0.44 * 4.8
δ1 = -0.264
δ2 = +0.000

Δ1 = 1e6 * (detuning + δ1)
Δ2 = 1e6 * (detuning + δ2)

f1 = energy(states[end]) - energy(states[5]) + Δ1
f2 = energy(states[end]) - energy(states[14]) + Δ2
f_mw = energy(states[5]) - energy(states[1])

freqs = [[f1, f2], [f_mw]] .* (2π / Γ)

# DEFINE SATURATION INTENSITIES #
beam_radius = [5e-3, Inf]
Isat = π*h*c*Γ/(3λ^3)

s1 = 1.93/2
s2 = 1.93/2

s_mw = 0.04

sats = [[s1, s2], [s_mw]]

# DEFINE POLARIZATIONS #
pols = [[σ⁺, σ⁻], [σ⁰]]

ks_relative = [1, f_mw/f1]

offsets = [[0,0,0], [0,0,0]]

# DEFINE FUNCTION TO UPDATE PARAMETERS DURING SIMULATION #
function update_p!(p, r, t)
    B_scalar = min(t / p.sim_params.B_ramp_time, 1.0)
    B_grad = p.sim_params.B_grad_start + (p.sim_params.B_grad_end - p.sim_params.B_grad_start) * B_scalar
    p.sim_params.Bx = +r[1] * B_grad * 1e2 / k / 2
    p.sim_params.By = +r[2] * B_grad * 1e2 / k / 2
    p.sim_params.Bz = -r[3] * B_grad * 1e2 / k
    return nothing
end

function update_p_diffusion!(p, r, t)
    return nothing
end

import MutableNamedTuples: MutableNamedTuple
import Distributions: Geometric, Normal
sim_params = MutableNamedTuple(
    Zeeman_Hx = Zeeman_Hx,
    Zeeman_Hy = Zeeman_Hy,
    Zeeman_Hz = Zeeman_Hz,

    photon_budget = rand(Geometric(1/1000000)),
    
    x_dist = Normal(0, 400e-6),
    y_dist = Normal(0, 400e-6),
    z_dist = Normal(0, 400e-6),
    
    vx_dist = Normal(0, sqrt(kB*50e-6/2m)),
    vy_dist = Normal(0, sqrt(kB*50e-6/2m)),
    vz_dist = Normal(0, sqrt(kB*50e-6/2m)),

    B_ramp_time = 1e-9,
    B_grad_start = 0,
    B_grad_end = 4,
    
    Bx = 0.,
    By = 0.,
    Bz = 0.,

    dt_diffusion = 1e-7 / (1/Γ)
)