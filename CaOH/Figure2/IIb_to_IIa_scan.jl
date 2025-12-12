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

    B_div_C = 0.37/0.28
    A_div_BC = LinRange(0, 22.9, 20)
    A_div_BC = A_div_BC ./ (31.4 .- A_div_BC)
    A_div_C = A_div_BC .* (1 .+ B_div_C)
    C_div_ABC = (A_div_C .+ B_div_C .+ 1).^(-1)
    A_div_ABC = A_div_C .* C_div_ABC
    B_div_ABC = B_div_C .* C_div_ABC

    # Calculate the saturation parameters for IIa and IIb for a range of saturation ratios
    sat_ratios = zip(B_div_ABC, C_div_ABC, A_div_ABC)

    function update_sat_ratios!(prob, sat_ratios)
        I_sat_ratio, IIa_sat_ratio, IIb_sat_ratio = sat_ratios
        prob.p.sim_params.s1_ratio = I_sat_ratio
        prob.p.sim_params.s2_ratio = IIa_sat_ratio
        prob.p.sim_params.s3_ratio = IIb_sat_ratio
        return nothing
    end
    
end

(diffusions, diffusion_errors, diffusions_over_time) = compute_diffusion_iteratively(
    prob, prob_func!, prob_diffusion, prob_func_diffusion!, 
    n_iterations, n_trajectories_diffusion1, n_trajectories_diffusion2, n_times, diffusion_t_end, diffusion_τ_total,
    update_sat_ratios!, sat_ratios
)

using DelimitedFiles
writedlm("data/IIb_to_IIa_diffusions_x.txt", stack(diffusions_over_time[:,1,end,:]'))
writedlm("data/IIb_to_IIa_diffusions_z.txt", stack(diffusions_over_time[:,2,end,:]'))

scan_values = zip(sat_ratios, [diffusions[i,:,end] for i in axes(diffusions,1)])
sols = distributed_solve(n_trajectories, prob, prob_func!, scan_func_with_diffusion!(update_sat_ratios!), scan_values)

σs = σ_vs_time.(sols)
σ_means = [mean(σ[200:end]) for σ in σs]
open("data/IIb_to_IIa_sizes.txt", "w") do io
    for σ_mean in σ_means
        println(io, σ_mean)
    end
end

Ts = T_vs_time.(sols)
T_means = [mean(T[200:end]) for T in Ts]
open("data/IIb_to_IIa_temps.txt", "w") do io
    for T_mean in T_means
        println(io, T_mean)
    end
end
