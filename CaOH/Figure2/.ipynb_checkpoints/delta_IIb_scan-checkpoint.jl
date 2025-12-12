## TRAJECTORY PARAMETERS ##
n_trajectories = 200

## DIFFUSION PARAMETERS ##
n_iterations = 2
n_trajectories_diffusion1 = 50
n_trajectories_diffusion2 = 20000
n_times = 100
diffusion_t_end = 3e-6
diffusion_τ_total = 3e-6

using Distributed
procs_to_use = 112
if nprocs() <= procs_to_use
    addprocs(procs_to_use-nprocs())
end

@everywhere begin

    include("../../misc/helper_functions.jl")
    include("../../misc/compute_size_temperature.jl")
    include("../define_CaOH_structure.jl")
    include("../define_blueMOT_params.jl")
    include("../define_blueMOT_prob.jl")

    δIIbs = LinRange(-0.75, 2.25, 20)
    function update_δIIb!(prob, δIIb)
        detuning = 7.6
        δ3 = δIIb
        Δ3 = 1e6 * (detuning + δ3) * (2π / Γ)
        prob.p.ωs[3] = prob.p.ω0s[end] - prob.p.ω0s[10] + Δ3
        return nothing
    end

end

(diffusions, diffusion_errors, diffusions_over_time) = compute_diffusion_iteratively(
    prob, prob_func!, prob_diffusion, prob_func_diffusion!, 
    n_iterations, n_trajectories_diffusion1, n_trajectories_diffusion2, n_times, diffusion_t_end, diffusion_τ_total,
    update_δIIb!, δIIbs
)

using DelimitedFiles
writedlm("data/delta_IIb_diffusions_x.txt", stack(diffusions_over_time[:,1,end,:]'))
writedlm("data/delta_IIb_diffusions_z.txt", stack(diffusions_over_time[:,2,end,:]'))

scan_values = zip(δIIbs, [diffusions[i,:,end] for i in axes(diffusions,1)])
sols = distributed_solve(n_trajectories, prob, prob_func!, scan_func_with_diffusion!(update_δIIb!), scan_values)

σs = σ_vs_time.(sols)
σ_means = [mean(σ[200:end]) for σ in σs]
open("data/delta_IIb_sizes.txt", "w") do io
    for σ_mean in σ_means
        println(io, σ_mean)
    end
end

Ts = T_vs_time.(sols)
T_means = [mean(T[200:end]) for T in Ts]
open("data/delta_IIb_temps.txt", "w") do io
    for T_mean in T_means
        println(io, T_mean)
    end
end
