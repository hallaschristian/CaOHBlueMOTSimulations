using QuantumStates, PhysicalQuantitiesSimple

include("YO_X.jl")
include("YO_A.jl")

# Define constants for the laser cooling transition
const λ = @SIval 614 "nm"       # wavelength of transition
const Γ = @SIval 2π * 4.8 "MHz" # transition linewidth
const m = @SIval 104.9 "u"      # mass of YO
const k = 2π / λ                # wavenumber of transition

H = CombinedHamiltonian([X_state_ham, A_state_ham])
evaluate!(H)
QuantumStates.solve!(H)
update_basis_tdms!(H)
update_tdms!(H)

ground_state_idxs = 1:16
excited_state_idxs = 21:24
states_idxs = [ground_state_idxs; excited_state_idxs]

ground_states = H.states[ground_state_idxs]
excited_states = H.states[excited_state_idxs]

d = H.tdms[states_idxs, states_idxs, :]
states = H.states[states_idxs]

Zeeman_x(state, state′) = (Zeeman(state, state′,-1) - Zeeman(state, state′,1)) / √2
Zeeman_y(state, state′) = im*(Zeeman(state, state′,-1) + Zeeman(state, state′,1)) / √2
Zeeman_z(state, state′) = Zeeman(state, state′, 0)

Zeeman_x_mat = real.(operator_to_matrix(Zeeman_x, ground_states) .* (1e-4 * gS * μB * (2π/Γ) / h))
Zeeman_y_mat = imag.(operator_to_matrix(Zeeman_y, ground_states) .* (1e-4 * gS * μB * (2π/Γ) / h))
Zeeman_z_mat = real.(operator_to_matrix(Zeeman_z, ground_states) .* (1e-4 * gS * μB * (2π/Γ) / h))

using StaticArrays
Zeeman_Hx = MMatrix{size(Zeeman_x_mat)...}(Zeeman_x_mat)
Zeeman_Hy = MMatrix{size(Zeeman_y_mat)...}(Zeeman_y_mat)
Zeeman_Hz = MMatrix{size(Zeeman_z_mat)...}(Zeeman_z_mat)

import LoopVectorization: @turbo
function add_terms_dψ!(dψ, ψ, p, r, t)
    # @turbo for i ∈ 1:4
    #     dψ_i_re = zero(eltype(dψ.re))
    #     dψ_i_im = zero(eltype(dψ.im))
    #     for j ∈ 1:4
    #         ψ_i_re = ψ.re[j]
    #         ψ_i_im = ψ.im[j]
            
    #         H_re = p.sim_params.Bx * p.sim_params.Zeeman_Hx[i,j] + p.sim_params.Bz * p.sim_params.Zeeman_Hz[i,j]
    #         H_im = p.sim_params.By * p.sim_params.Zeeman_Hy[i,j]
            
    #         dψ_i_re += ψ_i_re * H_re - ψ_i_im * H_im
    #         dψ_i_im += ψ_i_re * H_im + ψ_i_im * H_re
            
    #     end
    #     dψ.re[i] += dψ_i_im
    #     dψ.im[i] -= dψ_i_re
    # end
    @turbo for i ∈ 5:16
        dψ_i_re = zero(eltype(dψ.re))
        dψ_i_im = zero(eltype(dψ.im))
        for j ∈ 5:16
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