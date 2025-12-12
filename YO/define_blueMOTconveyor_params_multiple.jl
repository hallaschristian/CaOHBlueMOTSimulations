### SIMULATION PARAMETERS: 3-FREQUENCY 1+2 BLUE MOT  ###

using QuantumSimulations

# DEFINE STATES #
energies = energy.(states) .* (2π / Γ)

# DEFINE FREQUENCIES #
detuning = +3.0
δ1 = +0.00
δ2 = +0.00
δ3 = -0.60
δ4 = +0.60

Δ1 = 1e6 * (detuning + δ1)
Δ2 = 1e6 * (detuning + δ2)
Δ3 = 1e6 * (detuning + δ3)
Δ4 = 1e6 * (detuning + δ4)

f1 = energy(states[end]) - energy(states[14]) + Δ1
f2 = energy(states[end]) - energy(states[13]) + Δ2
f3 = energy(states[end]) - energy(states[5]) + Δ3
f4 = energy(states[end]) - energy(states[5]) + Δ4

f_mw = energy(states[5]) - energy(states[1])

freqs = [[f1, f2, f3, f4], [f_mw]] .* (2π / Γ)

# DEFINE SATURATION INTENSITIES #
beam_radius = [5e-3, Inf]
Isat = π*h*c*Γ/(3λ^3)
P = @SIval 15 "mW"
Ip = 2P / (π * beam_radius[1]^2)

total_sat = Ip / Isat

s1_ratio = 0.20
s2_ratio = 0.10
s3_ratio = 0.35
s4_ratio = 0.35

s1 = s1_ratio * total_sat
s2 = s2_ratio * total_sat
s3 = s3_ratio * total_sat
s4 = s4_ratio * total_sat

s_mw = 0.04

sats = [[s1, s2, s3, s4], [s_mw]]

# DEFINE POLARIZATIONS #
# pols = [[σ⁺, σ⁻, σ⁺, σ⁻], [σ⁰]]
# pols = [[σ⁺, σ⁺, σ⁺, σ⁻], [σ⁰]]
pols = [[σ⁻, σ⁺, σ⁺, σ⁻], [σ⁰]]

ks_relative = [1, f_mw/f1]

offsets = [[0,0,0], [0,0,0]]

# DEFINE FUNCTION TO UPDATE PARAMETERS DURING SIMULATION #
function update_p!(p, r, t)

    # set ramped scale factor for saturation parameters
    s_scalar = min(t / p.sim_params.s_ramp_time, 1.0)
    s_factor = p.sim_params.total_sat * (p.sim_params.s_factor_start + (p.sim_params.s_factor_end - p.sim_params.s_factor_start) * s_scalar)
    p.sats[1][1] = p.sim_params.s1_ratio * s_factor
    p.sats[1][2] = p.sim_params.s2_ratio * s_factor
    p.sats[1][3] = p.sim_params.s3_ratio * s_factor
    p.sats[1][4] = p.sim_params.s4_ratio * s_factor

    # set ramped scale factor for B field
    B_scalar = min(t / p.sim_params.B_ramp_time, 1.0)
    B_grad = p.sim_params.B_grad_start + (p.sim_params.B_grad_end - p.sim_params.B_grad_start) * B_scalar
    p.sim_params.Bx = +r[1] * B_grad * 1e2 / k / 2
    p.sim_params.By = +r[2] * B_grad * 1e2 / k / 2
    p.sim_params.Bz = -r[3] * B_grad * 1e2 / k
    
    return nothing
end

function update_p_diffusion!(p, r, t)
    s_factor = p.sim_params.total_sat * p.sim_params.s_factor_end
    p.sats[1][1] = p.sim_params.s1_ratio * s_factor
    p.sats[1][2] = p.sim_params.s2_ratio * s_factor
    p.sats[1][3] = p.sim_params.s3_ratio * s_factor
    p.sats[1][4] = p.sim_params.s4_ratio * s_factor
    return nothing
end

import MutableNamedTuples: MutableNamedTuple
import Distributions: Geometric, Normal
sim_params = MutableNamedTuple(
    Zeeman_Hx = Zeeman_Hx,
    Zeeman_Hy = Zeeman_Hy,
    Zeeman_Hz = Zeeman_Hz,
    
    B_ramp_time = 5e-3 / (1/Γ),
    B_grad_start = 0.,
    B_grad_end = 100,

    s_ramp_time = 5e-3 / (1/Γ),
    s_factor_start = 3.0,
    s_factor_end = 0.9,

    photon_budget = rand(Geometric(1/1000000)),
    
    x_dist = Normal(0, 500e-6),
    y_dist = Normal(0, 500e-6),
    z_dist = Normal(0, 500e-6),
    
    vx_dist = Normal(0, sqrt(kB*35e-6/2m)),
    vy_dist = Normal(0, sqrt(kB*35e-6/2m)),
    vz_dist = Normal(0, sqrt(kB*35e-6/2m)),
    
    total_sat = total_sat,
    s1_ratio = s1_ratio,
    s2_ratio = s2_ratio,
    s3_ratio = s3_ratio,
    s4_ratio = s4_ratio,
    
    Bx = 0.,
    By = 0.,
    Bz = 0.,

    dt_diffusion = 1e-7 / (1/Γ)
)