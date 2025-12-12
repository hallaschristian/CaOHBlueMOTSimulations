### SIMULATION PARAMETERS: 3-FREQUENCY 1+2 BLUE MOT  ###

using QuantumSimulations

# DEFINE STATES #
energies = energy.(states) .* (2π / Γ)

# DEFINE FREQUENCIES #
detuning = +7.6
δ1 = +0.00
δ2 = -1.00
δ3 = +0.75

Δ1 = 1e6 * (detuning + δ1)
Δ2 = 1e6 * (detuning + δ2)
Δ3 = 1e6 * (detuning + δ3)

f1 = energy(states[end]) - energy(states[1]) + Δ1
f2 = energy(states[end]) - energy(states[10]) + Δ2
f3 = energy(states[end]) - energy(states[10]) + Δ3

freqs = [f1, f2, f3] .* (2π / Γ)

# DEFINE SATURATION INTENSITIES #
beam_radius = @SIval 5 "mm"
P = @SIval 0.50 * 13.1 "mW" # 0.50 factor to match scattering rate
Isat = π*h*c*Γ/(3λ^3)
I = 2P / (π * beam_radius^2)

total_sat = I / Isat
s1_ratio = 0.37
s2_ratio = 0.28
s3_ratio = 0.35

s1 = s1_ratio * total_sat
s2 = s2_ratio * total_sat
s3 = s3_ratio * total_sat

sats = [s1, s2, s3]

# DEFINE POLARIZATIONS #
pols = [σ⁻, σ⁺, σ⁻]

k_relative = 1

# DEFINE FUNCTION TO UPDATE PARAMETERS DURING SIMULATION #
function update_p_1plus2!(p, r, t)

    # set ramped scale factor for saturation parameters
    s_scalar = min(t / p.sim_params.s_ramp_time, 1.0)
    s_factor = p.sim_params.total_sat * (p.sim_params.s_factor_start + (p.sim_params.s_factor_end - p.sim_params.s_factor_start) * s_scalar)
    p.sats[1] = p.sim_params.s1_ratio * s_factor
    p.sats[2] = p.sim_params.s2_ratio * s_factor
    p.sats[3] = p.sim_params.s3_ratio * s_factor

    # set ramped scale factor for B field
    B_scalar = min(t / p.sim_params.B_ramp_time, 1.0)
    B_grad = p.sim_params.B_grad_start + (p.sim_params.B_grad_end - p.sim_params.B_grad_start) * B_scalar
    p.sim_params.Bx = +r[1] * B_grad * 1e2 / k / 2
    p.sim_params.By = +r[2] * B_grad * 1e2 / k / 2
    p.sim_params.Bz = -r[3] * B_grad * 1e2 / k
    
    return nothing
end

function update_p_1plus2_diffusion!(p, r, t)
    s_factor = p.sim_params.total_sat * p.sim_params.s_factor_end
    p.sats[1] = p.sim_params.s1_ratio * s_factor
    p.sats[2] = p.sim_params.s2_ratio * s_factor
    p.sats[3] = p.sim_params.s3_ratio * s_factor
end

σx_initial = 585e-6
σy_initial = 585e-6
σz_initial = 585e-6
Tx_initial = 35e-6
Ty_initial = 35e-6
Tz_initial = 35e-6

import MutableNamedTuples: MutableNamedTuple
import Distributions: Geometric, Normal
sim_params = MutableNamedTuple(
    Zeeman_Hx = Zeeman_Hx,
    Zeeman_Hy = Zeeman_Hy,
    Zeeman_Hz = Zeeman_Hz,
    
    B_ramp_time = 4e-3 / (1/Γ),
    B_grad_start = 0.,
    B_grad_end = 74.,

    s_ramp_time = 4e-3 / (1/Γ),
    s_factor_start = 0.9,
    s_factor_end = 0.7,

    photon_budget = rand(Geometric(1/13500)),
    
    x_dist = Normal(0, σx_initial),
    y_dist = Normal(0, σy_initial),
    z_dist = Normal(0, σz_initial),
    
    vx_dist = Normal(0, sqrt(kB*Tx_initial/2m)),
    vy_dist = Normal(0, sqrt(kB*Ty_initial/2m)),
    vz_dist = Normal(0, sqrt(kB*Tz_initial/2m)),
    
    total_sat = total_sat,
    s1_ratio = s1_ratio,
    s2_ratio = s2_ratio,
    s3_ratio = s3_ratio,
    
    Bx = 0.,
    By = 0.,
    Bz = 0.,

    dt_diffusion = 1e-7 / (1/Γ)
)