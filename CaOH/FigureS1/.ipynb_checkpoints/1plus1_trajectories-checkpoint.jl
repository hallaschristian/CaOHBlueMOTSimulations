## TRAJECTORY PARAMETERS ##
n_trajectories = 10

## DIFFUSION PARAMETERS ##
n_iterations = 2
n_trajectories_diffusion1 = 10
n_trajectories_diffusion2 = 3000
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

    prob = remake(prob, tspan=(0, 20e-3/(1/Γ)))
    prob.p.sim_params.s1_ratio = 0.5
    prob.p.sim_params.s2_ratio = 0.0
    prob.p.sim_params.s3_ratio = 0.5

end

(diffusions, diffusion_errors, diffusions_over_time) = compute_diffusion_iteratively(
    prob, prob_func!, prob_diffusion, prob_func_diffusion!, 
    n_iterations, n_trajectories_diffusion1, n_trajectories_diffusion2, n_times, diffusion_t_end, diffusion_τ_total
)

using DelimitedFiles
writedlm("data/delta_IIb_diffusions_x.txt", stack(diffusions_over_time[:,1,end,:]'))
writedlm("data/delta_IIb_diffusions_z.txt", stack(diffusions_over_time[:,2,end,:]'))

scan_values = zip(δIIbs, [diffusions[i,:,end] for i in axes(diffusions,1)])
sols = distributed_solve(n_trajectories, prob, prob_func!, scan_func_with_diffusion!(update_δIIb!), scan_values)

σs = compute_σ.(sols)
open("data/delta_IIb_sizes.txt", "w") do io
    for σ in σs
        println(io, σ)
    end
end

densities = compute_density.(sols)
open("data/delta_IIb_densities.txt", "w") do io
    for density in densities
        println(io, density)
    end
end