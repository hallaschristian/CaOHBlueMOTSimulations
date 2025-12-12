## TRAJECTORY PARAMETERS ##
n_trajectories = 300

## DIFFUSION PARAMETERS ##
n_iterations = 2
n_trajectories_diffusion1 = 100
n_trajectories_diffusion2 = 20000
n_times = 100
diffusion_t_end = 2e-6
diffusion_τ_total = 2e-6

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

    prob.p.sim_params.s1_ratio = 0.5625
    prob.p.sim_params.s2_ratio = 0.0
    prob.p.sim_params.s2_ratio = 0.4375

    prob = remake(prob, tspan=(0, 16e-3/(1/Γ)))

end

(diffusions, diffusion_errors, diffusions_over_time) = compute_diffusion_iteratively(
    prob, prob_func!, prob_diffusion, prob_func_diffusion!, 
    n_iterations, n_trajectories_diffusion1, n_trajectories_diffusion2, n_times, diffusion_t_end, diffusion_τ_total
)

using DelimitedFiles
writedlm("data/1plus1_diffusions_x.txt", stack(diffusions_over_time[:,1,end,:]'))
writedlm("data/1plus1_diffusions_z.txt", stack(diffusions_over_time[:,2,end,:]'))

@everywhere begin
    prob.p.diffusion_constant[1] = $diffusions[1,1,end]
    prob.p.diffusion_constant[2] = $diffusions[1,1,end]
    prob.p.diffusion_constant[3] = $diffusions[1,2,end]
end

sols = distributed_solve(n_trajectories, prob, prob_func!)

σs = σ_vs_time(sols[1])
io = open("data/1plus1_size_vs_time.txt", "w") do io
    for σ in σs
        println(io, σ)
    end
end

densities = density_vs_time(sols[1])
io = open("data/1plus1_density_vs_time.txt", "w") do io
    for density in densities
        println(io, density)
    end
end
