using Pkg; Pkg.activate(".")
using Persephone
#include("src/Persephone.jl")

# using GLMakie
using LinearAlgebra, TimerOutputs
import Statistics: mean

function main(phi)
    cohesion = 30e6
    iscluster = true
    if iscluster
	path = "/cluster/scratch/ademontserra/EGU_AnisotropicNoHealing_Plasticity"
	path = "/cluster/scratch/ademontserra/EGU_AnisotropicHealingLitho_Plasticity"
	path = "/cluster/scratch/ademontserra/EGU_AnisotropicHealing_Plasticity"
# 	path = "/cluster/scratch/ademontserra/EGU2_IsotropicPlasticity"
    else
        path = "output"
    end
    folder = "Plastic_phi_$(phi)_C_$(cohesion)MPa_rho(P)"
    OUT, iplot = setup_output(path, folder)

    #=========================================================================
        RELOAD CHECKPOINT    
    =========================================================================#
    load = false
    iload = 131
    i0 = 1

    #=========================================================================
        MAKE GRID
    =========================================================================#
    gr, IDs = init_grid(5; split=2, type=:dimensional) # 1st argument = N

    PhaseID = 1
    min_inradius = inrectangle(gr) / 2
    GlobC = [Point2D{Polar}(gr.θ[i], gr.r[i]) for i in 1:(gr.nnod)] # → global coordinates

    #=========================================================================
        Boundary conditions:
            temperature_type = :heated, :insulated
    =========================================================================#
    TBC, UBC = init_BCs(
        gr, IDs; Ttop=300.0, Tbot=4000.0, velocity_type=:free_slip, temperature_type=:heated
    )

    #=========================================================================
        Delete existing statistics file    
    =========================================================================#
    ScratchNu, stats_file = setup_metrics(gr, path, folder; load=load)

    #=========================================================================
        GET DEM STRUCTURE:    
    =========================================================================#
    dem_file = joinpath("DEM", "DEM_1e-3_vol30.h5")
    Δη, ϕ = 1e-3, 0.3
    D = getDEM(dem_file, Δη, ϕ)

    #=========================================================================
        FIX θ OF ELEMENTS CROSSING π and reshape also r (i.e. periodic boundaries)
    =========================================================================#
    θStokes, rStokes = elementcoordinate(GlobC, @views(gr.e2n[1:3, :]))
    θThermal, rThermal = elementcoordinate(GlobC, gr.e2n_p1)
    fixangles!(θStokes)
    fixangles!(θThermal)
    coordinates = ElementCoordinates(θStokes, rStokes)

    #=========================================================================
        IPs polar coordinates (Some stuff needs to be deprecated)
    =========================================================================#
    θ6, r6 = elementcoordinate(GlobC, @views(gr.e2n[1:6, :]))
    θ3 = θStokes
    fixangles6!(θ6)
    fixangles!(θ3)
    ipx, ipz = getips(gr.e2n, θ6, r6)
    ix, iz = polar2cartesian(ipx, ipz)
    transition = gr.Ro - 660e3
    isotropic_idx = findall(ipz .> transition)
    IntC = [@inbounds(Point2D{Polar}(ipx[i], ipz[i])) for i in CartesianIndices(ipx)] # → ip coordinates

    #=========================================================================
        INITIALISE PARTICLES
    =========================================================================#
    particle_info, particle_weights, particle_fields = particles_generator(
        θThermal, rThermal, IntC, gr.e2n_p1; number_of_particles=12
    )

    #=========================================================================
        ALLOCATE/INITIALISE FIELDS    
    =========================================================================#
    # Allocate velocity and pressure fields
    U, P = init_U_P(gr)
    # Allocate velocity gradient, stress and strain tensors:
    F, _, τ, ε, τII, εII, = initstress(gr.nel; nip=7)
    # FSE = [FiniteStrainEllipsoid(1.0, 0.0, 0.0, 1.0, 1.0, 1.0) for _ in 1:gr.nel, _ in 1:6]
    # Allocate viscosity tensor:
    𝓒 = 𝓒init(gr.nel, 7)
    # Allocate nodal velocities
    Ucartesian, Upolar = initvelocity(gr.nnod)
    # Initialise temperature @ nodes
    perturbation = :random
    T = init_temperature(gr, IDs; type=perturbation, units=:dimensional)
    ΔT = similar(T)
    # Initialise temperature @ particles
    # init_particle_temperature!(particle_fields, particle_info, gr.Ro, gr.Ri, type = perturbation)
    particle_fields = T2particle(
        particle_fields, gr.e2n_p1, T, particle_info, particle_weights
    )

    # Annealing rate
    annealing = 0

    # Finite Strain structure
    FSE = FiniteStrain(
        gr.nel;
        nip=7,
        ϵ=1e3, # a1/a2 at which fabric is destroyed
        # annealing_rate = annealing, # annealing rate
        r_iso = 0#6351e3-660e3, # depth above which Ω is isotropic
    )

    #= Viscosity type. Options:
        (*) "IsoviscousIsotropic"
        (*) "TemperatureDependantIsotropic"
        (*) "TemperatureDependantIsotropicPlastic"
        (*) "IsoviscousAnisotropic"
        (*) "TemperatureDependantAnisotropic"
        (*) "TemperatureDependantAnisotropicPlastic"
    =#
    viscosity_type = :TemperatureDependantIsotropicPlastic
    viscosity_type = :Nakagawa
    viscosity_type = :IsoviscousIsotropic

    # Physical parameters for thermal diffusion
    VarT = ThermalParameters(; K=3.3, α=3e-5, Cp=1.2e3, dQdT=0.0, H=0.0)

    # ρ = state_equation(VarT.α, T)
    ρ = Density(gr.r; αi=VarT.α)
    # η = getviscosity(T, gr.r, viscosity_type, gr.nel; η=1.4e21, ϕ=0.0, C=0.0)
    # η = 1 for isotropic
    # η = 1.81 for anisotropic with ϕ = 30%
    # η = 1/0.6899025321942348 for anisotropic with ϕ = 20%
    η = Arrhenius(gr; type=Anisotropic, Ea=200e3, Va=2.6e-6, ϕ=phi, C=cohesion)
    Valη = Val(η)
    g = -9.81
    𝓒 = anisotropic_tensor(FSE.fse, D, Valη)

    #=========================================================================
        SOLVER INVARIANTS (FOR AN IMMUTABLE MESH):
            TODO pre-compute the Jacobians
    =========================================================================#
    # Allocate rotation tensor 
    RotationMatrices = rotation_matrix(gr.θ)
    # Allocate sparsity patterns of Stokes block matrices 
    KKidx, GGidx, MMidx, = sparsitystokes(gr)
    # Allocate spasity pattern of thermal diffusion stiffness matrix
    CMidx, _ = sparsitythermal(gr.e2n, 6)

    # Stokes immutables
    KS, GS, MS, FS, DoF_U, DoF_P, nn, SF_Stokes, ScratchStokes = stokes_immutables(
        gr, gr.nnod, 2, 3, 6, gr.nel, 7
    )

    SF_Stress = stress_shape_functions(; nip=7)

    # Diffusion_immutables
    KT, MT, FT, DoF_T, valA, SF_Diffusion, ScratchDifussion = diffusion_immutables(
        gr, 2, 3, 6, 7
    )

    # Color elements
    _, color_list = color_mesh(gr.e2n)

    #=========================================================================
        LOAD PREVIOUS MODELS
    =========================================================================#
    if load == true
        i0 = iload + 1
        iload = 5
        load_file = string("file_", iload, ".h5")
        toreload = joinpath(pwd(), path, folder, load_file)
        T, F = reloader(toreload)
        mesh(xz.*1e-3, gr.e2n[1:3,:]', color=T, colormap=Reverse(:romaO))
        FSE, F = getFSE(F, FSE)
        𝓒 = anisotropic_tensor(FSE.fse, D, Valη)
    end

    #=========================================================================
        START SOLVER
    =========================================================================#
    to = TimerOutput()
    Time = 0.0
    T0 = deepcopy(T)

    # Picard iterations
    max_it = 20
    tol = 1e-3
    Δt = 0.0
    U0 = deepcopy(U)

    println("Starting simulation")
    for iplot in i0:500
        for _ in 1:50
            reset_timer!(to)

            #= Update material properties =#
            # state_equation!(ρ, VarT.α, T)
            # state_equation!(ρ, VarT.α, T; ρ0 = 3300)
            state_equation!(ρ, T)
            # getviscosity!(η, T, gr.r)
            # getviscosity!(η, T)
            # getviscosity!(η, T, P)
            getviscosity!(η, T, P, εII, τII, gr.e2n, gr.r)

            # for it in 1:max_it
            #= Stokes solver using preconditioned-CG =#
            Ucartesian, Upolar, U, Ucart, P, to = solveStokes(
                U,
                P,
                T,
                gr,
                Ucartesian,
                Upolar,
                g,
                ρ, # this should be T for non-dimensional
                η,
                𝓒,
                εII,
                coordinates,
                RotationMatrices,
                PhaseID,
                UBC,
                KKidx,
                GGidx,
                MMidx,
                to;
                solver=:pardiso,
            )

            #= Stress-Strain post processor =#
	        @timeit to "F" stress!(
                ε, εII, τ, τII, η, 𝓒, Ucart, gr.nel, DoF_U, coordinates, SF_Stress
            )

            println("min:max Uθ ", extrema(@views U[1:2:end]))
            println("mean speed ", mean(@views @. (√(U[1:2:end]^2 + U[2:2:end]^2))))

            #= Adaptive time-step =#
            Δt = calculate_Δt(Ucartesian, gr.nθ, min_inradius)
            println("Δt = $(Δt/yr) years")

	    if Val(η) isa Val{Anisotropic}
	            deformation_gradient!(F, U, gr.nel, DoF_U, coordinates, SF_Stress, Δt)

        	    # isotropic_lithosphere!(F, isotropic_idx)
            	# F = healing(F, FSE)
            	# @timeit to "FSE" FSE = getFSE(F, FSE)
            	# F0, FSE0 = deepcopy(F), deepcopy(FS)
            	# @timeit to "FSE" FSE, F = getFSE_healing(F, FSE, ϵ=1e3)
            	# @timeit to "FSE" getFSE_annealing!(F, FSE, annealing*(Time+Δt))

            	# Finite Strain Ellipsoid calculation
            	if FSE.isotropic_domain.r != 0 # check whether any region of the Earth is always isotropic
                	isotropic_lithosphere!(F, isotropic_idx) # force isotropy         
            	end
            	FSE, F = getFSE(F, FSE)
            	
		#= Compute the viscous tensor =#
           	@timeit to "viscous tensor" 𝓒 = anisotropic_tensor(FSE.fse, D, Valη)

	    end
            #= Diffusion solver =#
            T, T0, ΔT, to = solveDiffusion_threaded(
                color_list,
                CMidx,
                KT,
                MT,
                FT,
                DoF_T,
                coordinates,
                VarT,
                ScratchDifussion,
                SF_Diffusion,
                valA,
                ρ,
                Δt,
                T,
                T0,
                TBC,
                to,
            )

            @timeit to "Particle advenction" begin
                #=
                Particle advection and mappings
                    Fij : ip   |-> particle
                    T   : node |-> particle
                    Ui  : node |-> particle + advection
                =#

                @timeit to "F → particle" particle_fields = F2particle(
                    particle_fields, particle_info, ipx, ipz, F
                )

                @timeit to "T → particle" begin
                    interpolate_temperature!(
                        T0,
                        particle_fields,
                        gr,
                        ρ,
                        T,
                        particle_info,
                        particle_weights,
                        VarT,
                        Δt,
                        ΔT,
                    )
                end

                @timeit to "advection" particle_info, particle_weights, to = advection_RK2(
                    particle_info,
                    gr,
                    particle_weights,
                    Ucartesian,
                    Δt,
                    θThermal,
                    rThermal,
                    coordinates,
                    IntC,
                    to,
                )

                println("Min-max particle temperature = ", extrema(particle_fields.T))
            end

            @timeit to "Locate articles" begin
                #= Particle locations =#
                particle_info, particle_weights, found = tsearch_parallel(
                    particle_info,
                    particle_weights,
                    θThermal,
                    rThermal,
                    coordinates,
                    gr,
                    IntC,
                )

                lost_particles = length(particle_info) - sum(found)
                check_corruption!(found, particle_fields)
                if lost_particles > 0
                    idx = findfirst(i-> i==0, found)
                    println(
                        "Lost particle at ",
                        (particle_info[idx].CCart.x, particle_info[idx].CCart.z),
                        " Corrupted particles: ",
                        length(particle_info) - sum(found) - lost_particles,
                    )
                end

                if lost_particles > 0
                    particle_info, particle_weights, particle_fields = purgeparticles(
                        particle_info, particle_weights, particle_fields, found
                    )
                end
            end

            @timeit to "Particle to node/ip" begin
                #=
                    Map back to original immutable locations
                        Fij : particle -> ip
                        T   : particle -> node
                =#
                @timeit to "F → ip" F = F2ip(
                    F, particle_fields, particle_info, particle_weights, gr.nel
                )

                T = T2node(
                    T,
                    particle_fields,
                    particle_info,
                    particle_weights,
                    gr,
                    IDs;
                    Ttop=300.0,
                    Tbot=4000.0,
                    n=1
                )

                println("\n Min-max nodal temperature = ", extrema(T))
            end

            println("mean T before injection ", mean(particle_fields.T))

            @timeit to "Add/reject particles" begin
                particle_info, particle_weights, particle_fields = addreject(
                    T,
                    F,
                    gr,
                    θThermal,
                    rThermal,
                    IntC,
                    particle_info,
                    particle_weights,
                    particle_fields;
                    min_num_particles=8,
                )
            end

            println("mean T after injection ", mean(particle_fields.T))

            write_stats(U, T, length(particle_info), gr, Time, ScratchNu, stats_file)

            Time += Δt
            show(to; compact=true)
        end

        #= Save output file =#
        println("\n time = ", Time)
        println("\n Saving output...")
        OUT = IOs(path, folder, "file", iplot)
        savedata(
            OUT,
            Upolar,
            Ucartesian,
            T,
            η,
            𝓒,
            ρ.ρ,
            ε,
            F,
            FSE.fse,
            gr.nθ,
            gr.nr,
            particle_fields,
            particle_info,
            Time,
            Val(η),
        )

        println(" ...done!")
    end

end

phi = parse(Float64,ARGS[1])

main(phi)
