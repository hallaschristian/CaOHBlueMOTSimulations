## TRAJECTORY PARAMETERS ##
n_trajectories = 400

## DIFFUSION PARAMETERS ##
n_iterations = 2
n_trajectories_diffusion1 = 100
n_trajectories_diffusion2 = 30000
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

    prob = remake(prob, tspan=(0, 15e-3/(1/Γ)))
    prob.p.sim_params.s1_ratio = 0.56
    prob.p.sim_params.s2_ratio = 0.44
    prob.p.sim_params.s3_ratio = 0.0
    prob.p.sim_params.B_grad_end = +82

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
    
    δIIas = LinRange(-2.2, 0.0, 12)[7:12]
    function update_δIIa!(prob, δIIa)
        detuning = 7.6
        δ2 = δIIa
        Δ2 = 1e6 * (detuning + δ2) * (2π / Γ)
        prob.p.ωs[2] = prob.p.ω0s[end] - prob.p.ω0s[10] + Δ2
        return nothing
    end

end

(diffusions, diffusion_errors, diffusions_over_time) = compute_diffusion_iteratively(
    prob, prob_func!, prob_diffusion, prob_func_diffusion!, 
    n_iterations, n_trajectories_diffusion1, n_trajectories_diffusion2, n_times, diffusion_t_end, diffusion_τ_total,
    update_δIIa!, δIIas
)

using DelimitedFiles
writedlm("data/positive_Bfield_diffusions_x2.txt", stack(diffusions_over_time[:,1,end,:]'))
writedlm("data/positive_Bfield_diffusions_z2.txt", stack(diffusions_over_time[:,2,end,:]'))

scan_values = zip(δIIas, [diffusions[i,:,end] for i in axes(diffusions,1)])
sols = distributed_solve(n_trajectories, prob, prob_func!, scan_func_with_diffusion!(update_δIIa!), scan_values)

σs = compute_σ.(sols)
open("data/positive_Bfield_sizes2.txt", "w") do io
    for σ in σs
        println(io, σ)
    end
end

densities = compute_density.(sols)
open("data/positive_Bfield_densities2.txt", "w") do io
    for density in densities
        println(io, density)
    end
end