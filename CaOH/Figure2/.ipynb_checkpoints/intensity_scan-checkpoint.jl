## TRAJECTORY PARAMETERS ##
n_trajectories = 20

## DIFFUSION PARAMETERS ##
n_iterations = 2
n_trajectories_diffusion1 = 10
n_trajectories_diffusion2 = 2000
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

    prob = remake(prob, tspan=(0, 6e-3/(1/Γ)))
    prob.p.sim_params.s_factor_start = 1.0
    prob.p.sim_params.s_factor_end = 1.0

    function d1(β)
        c = cos(β)
        s = sin(β)
        return [
            (1 + c)/2     s/sqrt(2)    (1 - c)/2;
           -s/sqrt(2)     c            s/sqrt(2);
            (1 - c)/2    -s/sqrt(2)    (1 + c)/2
        ]
    end
    
    using LinearAlgebra
    function D1(α, β, γ)
        # Order m = [-1, 0, +1]
        m = [-1, 0, 1]
        L = Diagonal(exp.(-1im .* m .* α))
        R = Diagonal(exp.(-1im .* m .* γ))
        return L * d1(β) * R
    end
    
    prob.p.ϵs[1,1,:] .= D1(0, pi/2, 0) * pols[1]
    prob.p.ϵs[4,1,:] .= D1(0, -pi/2, 0) * pols[1]
    prob.p.ϵs[1,2,:] .= D1(0, pi/2, 0) * pols[2]
    prob.p.ϵs[4,2,:] .= D1(0, -pi/2, 0) * pols[2]
    prob.p.ϵs[1,3,:] .= D1(0, pi/2, 0) * pols[3]
    prob.p.ϵs[4,3,:] .= D1(0, -pi/2, 0) * pols[3]
    
    prob.p.ϵs[2,1,:] .= D1(pi/2, pi/2, 0) * pols[1]
    prob.p.ϵs[5,1,:] .= D1(-pi/2, pi/2, 0) * pols[1]
    prob.p.ϵs[2,2,:] .= D1(pi/2, pi/2, 0) * pols[2]
    prob.p.ϵs[5,2,:] .= D1(-pi/2, pi/2, 0) * pols[2]
    prob.p.ϵs[2,3,:] .= D1(pi/2, pi/2, 0) * pols[3]
    prob.p.ϵs[5,3,:] .= D1(-pi/2, pi/2, 0) * pols[3]
    
    total_sats = LinRange(0.3, 2.1, 10) .* total_sat
    function update_total_s!(prob, total_sat)
        prob.p.sim_params.total_sat = total_sat
        return nothing
    end

end

(diffusions, diffusion_errors, diffusions_over_time) = compute_diffusion_iteratively(
    prob, prob_func!, prob_diffusion, prob_func_diffusion!, 
    n_iterations, n_trajectories_diffusion1, n_trajectories_diffusion2, n_times, diffusion_t_end, diffusion_τ_total,
    update_total_s!, total_sats
)

using DelimitedFiles
writedlm("data/intensity_diffusions_x.txt", stack(diffusions_over_time[:,1,end,:]'))
writedlm("data/intensity_diffusions_z.txt", stack(diffusions_over_time[:,2,end,:]'))

scan_values = zip(total_sats, [diffusions[i,:,end] for i in axes(diffusions,1)])
sols = distributed_solve(n_trajectories, prob, prob_func!, scan_func_with_diffusion!(update_total_s!), scan_values)

σs = σ_vs_time.(sols)
σ_means = [mean(σ[200:end]) for σ in σs]
open("data/intensity_sizes.txt", "w") do io
    for σ_mean in σ_means
        println(io, σ_mean)
    end
end

Ts = T_vs_time.(sols)
T_means = [mean(T[200:end]) for T in Ts]
open("data/intensity_temps.txt", "w") do io
    for T_mean in T_means
        println(io, T_mean)
    end
end
