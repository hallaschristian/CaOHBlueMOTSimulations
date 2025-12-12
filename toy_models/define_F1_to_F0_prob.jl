# Define constants for the laser cooling transition
using PhysicalQuantitiesSimple
const λ = @SIval 626 "nm"
const Γ = @SIval 2π * 6.4 "MHz"
const m = @SIval 57 "u"
const k = 2π / λ

# Define ground states
using QuantumStates
QN_bounds = (
    label = "F1",
    N = 1
)
F1_state_basis = order_basis_by_m(enumerate_states(AngularMomentumState, QN_bounds))

F1_state_operator = :(
    T * Identity
)

F1_state_parameters = QuantumStates.@params begin
    T = 0
end

F1_state_ham = Hamiltonian(basis=F1_state_basis, operator=F1_state_operator, parameters=F1_state_parameters)
evaluate!(F1_state_ham)
QuantumStates.solve!(F1_state_ham)

# Define excited states
QN_bounds = (
    label = "F0",
    N = 0
)
F0_state_basis = order_basis_by_m(enumerate_states(AngularMomentumState, QN_bounds))

F0_state_operator = :(
    T * Identity
)

F0_state_parameters = QuantumStates.@params begin
    T = 478e12
end

F0_state_ham = Hamiltonian(basis=F0_state_basis, operator=F0_state_operator, parameters=F0_state_parameters)
evaluate!(F0_state_ham)
QuantumStates.solve!(F0_state_ham)

# Create combined Hamiltonian and define Zeeman Hamiltonians
H = CombinedHamiltonian([F1_state_ham, F0_state_ham])
evaluate!(H)
QuantumStates.solve!(H)
update_basis_tdms!(H)
update_tdms!(H)

ground_state_idxs = 1:3
excited_state_idxs = 4
states_idxs = [ground_state_idxs; excited_state_idxs]

ground_states = H.states[ground_state_idxs]
excited_states = H.states[excited_state_idxs]

d = H.tdms[states_idxs, states_idxs, :]
states = H.states[states_idxs]

Zeeman_x(state, state′) = -(Zeeman(state, state′,-1) - Zeeman(state, state′,1)) / √2
Zeeman_y(state, state′) = -im*(Zeeman(state, state′,-1) + Zeeman(state, state′,1)) / √2
Zeeman_z(state, state′) = Zeeman(state, state′, 0)

# Zeeman_x(state, state′) = (Zeeman(state, state′,-1) + Zeeman(state, state′,1)) / √2
# Zeeman_y(state, state′) = im*(Zeeman(state, state′,-1) - Zeeman(state, state′,1)) / √2
# Zeeman_z(state, state′) = Zeeman(state, state′, 0)

Zeeman_x_mat = real.(operator_to_matrix(Zeeman_x, ground_states) .* (1e-4 * gS * μB * (2π/Γ) / h))
Zeeman_y_mat = imag.(operator_to_matrix(Zeeman_y, ground_states) .* (1e-4 * gS * μB * (2π/Γ) / h))
Zeeman_z_mat = real.(operator_to_matrix(Zeeman_z, ground_states) .* (1e-4 * gS * μB * (2π/Γ) / h))

using StaticArrays
Zeeman_Hx = MMatrix{size(Zeeman_x_mat)...}(Zeeman_x_mat)
Zeeman_Hy = MMatrix{size(Zeeman_y_mat)...}(Zeeman_y_mat)
Zeeman_Hz = MMatrix{size(Zeeman_z_mat)...}(Zeeman_z_mat)

import LoopVectorization: @turbo
function add_terms_dψ!(dψ, ψ, p, r, t)
    @turbo for i ∈ 1:3
        dψ_i_re = zero(eltype(dψ.re))
        dψ_i_im = zero(eltype(dψ.im))
        for j ∈ 1:3
            ψ_i_re = ψ.re[j]
            ψ_i_im = ψ.im[j]
            
            H_re = p.sim_params.Bx * p.sim_params.Zeeman_Hx[i,j] + p.sim_params.Bz * p.sim_params.Zeeman_Hz[i,j]
            H_im = p.sim_params.By * p.sim_params.Zeeman_Hy[i,j]
            
            dψ_i_re += ψ_i_re * H_re - ψ_i_im * H_im
            dψ_i_im += ψ_i_re * H_im + ψ_i_im * H_re
            
        end
        dψ.re[i] += dψ_i_im
        dψ.im[i] -= dψ_i_re
    end
    return nothing
end

# Define simulation parameters
using QuantumSimulations

energies = energy.(states) * (2π / Γ)

detuning = @SIval 16 "MHz"
two_photon_det = @SIval 3 "MHz"

Δ1 = detuning + two_photon_det
Δ2 = detuning

f1 = energy(states[end]) - energy(states[2]) + Δ1
f2 = energy(states[end]) - energy(states[2]) + Δ2

freqs = [f1, f2] * (2π / Γ)

beam_radius = Inf

s1 = 8
s2 = 8

sats = [s1, s2]
pols = [σ⁺, σ⁻]
k_relative = 1
offsets = [0, 0, 0]

function update_p!(p, r, t)   
    B_grad = p.sim_params.B_grad
    p.sim_params.Bx = +r[1] * B_grad * 1e2 / k / 2
    p.sim_params.By = +r[2] * B_grad * 1e2 / k / 2
    p.sim_params.Bz = -r[3] * B_grad * 1e2 / k
end

import MutableNamedTuples: MutableNamedTuple
import Distributions: Geometric, Normal
sim_params = MutableNamedTuple(
    Zeeman_Hx = Zeeman_Hx,
    Zeeman_Hy = Zeeman_Hy, 
    Zeeman_Hz = Zeeman_Hz,
    
    B_grad = 10,
    Bx = 0.,
    By = 0.,
    Bz = 0.
)

function update_phases!(prob)
    for k ∈ 1:3
        for f ∈ axes(prob.p.ϵs,2)
            ϕ1 = exp(im * 2π * rand())
            ϕ2 = exp(im * 2π * rand())
            for q ∈ axes(prob.p.ϵs,3)
                prob.p.ϵs_with_phases[k,f,q] = prob.p.ϵs[k,f,q] * ϕ1
                prob.p.ϵs_with_phases[k+3,f,q] = prob.p.ϵs[k+3,f,q] * ϕ2
            end
        end
    end
    return nothing
end

function prob_func!(prob)
    reset_prob!(prob)
    update_initial_position!(prob, (rand((-8e-3,8e-3)),0,0) ./ (1/k))
    update_initial_velocity!(prob, (0,0,0))
    update_phases!(prob)
    return nothing
end

x(u) = real(u[8+1+1]) * (1/k)
y(u) = real(u[8+1+2]) * (1/k)
z(u) = real(u[8+1+3]) * (1/k)
vx(u) = real(u[8+1+4]) * (Γ/k)
vy(u) = real(u[8+1+5]) * (Γ/k)
vz(u) = real(u[8+1+6]) * (Γ/k)
r(u) = sqrt(x(u)^2 + y(u)^2 + z(u)^2)