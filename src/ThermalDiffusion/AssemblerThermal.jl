function solveDiffusion_threaded(
    color_list,
    CMidx,
    KT,
    MT,
    FT,
    DoF_T,
    coordinates,
    VarT::ThermalParameters,
    ScratchDifussion::ScratchThermalGlobal,
    SF_Diffusion::ShapeFunctionsThermal,
    valA,
    ρ,
    Δt,
    T,
    T0,
    TBC,
    to,
)
    ∂Ωt = TBC.Ω
    tfix = TBC.vfix
    tfree = TBC.ifree
    copyto!(T0, T)

    @timeit to "Thermal diffusion threaded" begin
        # Reset Matrices
        fill!(KT.nzval, 0.0)
        fill!(MT.nzval, 0.0)
        fill!(FT, 0.0)

        # Element assembly
        assembly_threaded!(
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
        )

        # Apply Boundary conditions
        _prepare_matrices!(MT, FT, KT, T, Δt)
        _apply_bcs!(T, MT, FT, ∂Ωt, tfix)

        # Solve temperature
        # T[tfree] .= MT[tfree, tfree] \ FT[tfree]
        T[tfree], = Krylov.gmres(MT[tfree, tfree], FT[tfree])

        # Temperature increment
        ΔT = @tturbo T .- T0
    end
    return T, T0, ΔT, to
end

"""
assembly_threaded!(color_list, K, M, F, dof, coordinates::ElementCoordinates, 
                           VarT::ThermalParameters, A::ScratchThermalGlobal, 
                           B::ShapeFunctionsThermal, valA, ρ, Δt)

Threaded assembly of FEM matrices for thermal diffusion. Based on a colored mesh.
"""
function assembly_threaded!(
    color_list,
    CMidx,
    KT,
    MT,
    FT,
    DoF_T,
    coordinates::ElementCoordinates,
    VarT::ThermalParameters,
    ScratchDifussion::ScratchThermalGlobal,
    SF_Diffusion::ShapeFunctionsThermal,
    valA,
    ρ,
    Δt,
)

    ## Shape function and their derivatives
    ni, nn, nn3 = Val(ScratchDifussion.nip), Val(ScratchDifussion.nnodel), Val(3)
    N, ∇N, dN3ds, w_ip = _get_SF(ni, nn)
    N3, = _get_SF(ni, nn3)

    for color in color_list
        Threads.@threads for element in color
            assemble_element!(
                element,
                CMidx,
                KT,
                MT,
                FT,
                DoF_T,
                coordinates.θ,
                coordinates.r,
                VarT,
                ScratchDifussion,
                N,
                ∇N,
                dN3ds,
                w_ip,
                N3,
                valA,
                ρ,
                Δt,
            )
        end
    end
    return KT, MT, FT
end

function assemble_element!(
    iel,
    CMidx,
    KT,
    MT,
    FT,
    DoF_T,
    θ,
    r,
    VarT::ThermalParameters,
    A::ScratchThermalGlobal,
    N,
    ∇N,
    dN3ds,
    w_ip,
    N3,
    valA,
    ρ,
    Δt,
)

    # Unpack physical parameters
    K, Cp, dQdT = VarT.K, VarT.Cp, VarT.dQdT
    # Polar coordinates of element nodes
    θ_el = @SVector [θ[i, iel] for i in 1:3]
    r_el = @SVector [r[i, iel] for i in 1:3]
    coords = SMatrix{3,2}([θ_el r_el])

    # Jacobian n. 1 (p:=polar, l:=local): reference element --> current element
    J_pl = dN3ds * coords
    # detJ_pl = det(J_pl) # fast |-> unrolled
    # detJ_pl = J_pl[1] * J_pl[4] - J_pl[2] * J_pl[3] # Determinant of Jacobi matrix
    detJ_pl = muladd(J_pl[1], J_pl[4], -J_pl[2] * J_pl[3]) # Determinant of Jacobi matrix

    R_21 = r_el[2] - r_el[1] # = -detJa_PL*deta_dth
    R_31 = r_el[3] - r_el[1] # =  detJa_PL*dxi_dth
    Th_31 = θ_el[3] - θ_el[1]   # = -detJa_PL*dxi_dr
    Th_21 = θ_el[2] - θ_el[1]   # =  detJa_PL*deta_dr

    # the Jacobian ∂ξ∂θ to transform local (ξ, η) into global (θ,r) derivatives
    #     ∂ξ∂θ = [ R_31    -R_21
    #             -Th_31    Th_21] / detJa_PL
    # ∂ξ∂θ = @SMatrix [(r_el[3]-r_el[1]) (r_el[2]-r_el[1])
    #                  (θ_el[3]-θ_el[1])  (θ_el[2]-θ_el[1])]

    Ke = empty_element_matrices(valA)
    Me = empty_element_matrices(valA)
    fe = empty_element_force_vector(valA)

    el_dofs = DoF_T[iel]
    ρ_el = @SVector [ρ[el_dofs[i]] for i in 1:3]
    dQdT_el = @SVector [dQdT for _ in 1:6]

    # INTEGRATION LOOP
    @inbounds for ip in 1:(A.nip)
        # Unpack shape functions 
        N_ip = N[ip]
        N3_ip = N3[ip]
        ∇N_ip = ∇N[ip]

        # Polar coordinates of the integration points
        θ_ip = mydot(θ_el, N3_ip)
        r_ip = mydot(r_el, N3_ip)
        sin_ip, cos_ip = sincos(θ_ip)
        cos_ip_r_ip = cos_ip / r_ip
        sin_ip_r_ip = sin_ip / r_ip

        # ρ at ith integration point
        ρ_ip = mydot(ρ_el, N3_ip)
       
        # Build inverse of the 2nd Jacobian
        # invJ_double = @SMatrix [(∂ξ∂θ[1,1]*cos_ip_r_ip-∂ξ∂θ[2,1]*sin_ip) (-∂ξ∂θ[1,2]*cos_ip_r_ip+∂ξ∂θ[2,2]*sin_ip); 
        #                        -(∂ξ∂θ[1,1]*sin_ip_r_ip+∂ξ∂θ[2,1]*cos_ip) (∂ξ∂θ[1,2]*sin_ip_r_ip+∂ξ∂θ[2,2]*cos_ip) ]
        invJ_double = @SMatrix [
            R_31 * cos_ip_r_ip-Th_31 * sin_ip -R_21 * cos_ip_r_ip+Th_21 * sin_ip
           -R_31 * sin_ip_r_ip-Th_31 * cos_ip R_21 * sin_ip_r_ip+Th_21 * cos_ip
        ]

        # Partial derivatives
        ∂N∂x = invJ_double * ∇N_ip / detJ_pl
        # Integration weight
        ω = r_ip * detJ_pl * w_ip[ip]

        # Update elemental matrices
        NxN = N_ip' * N_ip
        Ke += (∂N∂x' * K * ∂N∂x) * ω
        Me += NxN * ω * ρ_ip * Cp

        # Force vector -- right hand side
        if dQdT != 0
            fe += vec(N_ip) .* dQdT_el * (Δt * ω)
        end
    end

    # Update stiffness matrix 
    return update_stiffness_matrix!(KT, MT, el_dofs, Ke, Me)

    # Update force vector
    if dQdT != 0
        update_force_vector!(FT, fe, el_dofs)
    end
end

empty_element_matrices(::Val{sz}) where {sz} = zeros(SMatrix{sz,sz})

empty_element_matrices(::Val{sI}, ::Val{sJ}) where {sI,sJ} = zeros(SMatrix{sI,sJ})

empty_element_force_vector(::Val{sz}) where {sz} = zeros(SVector{sz})

function update_stiffness_matrix!(K::SparseMatrixCSC{Float64,Int64}, el_dofs, Ke)
    @inbounds for (_j, j) in enumerate(el_dofs), (_i, i) in enumerate(el_dofs)
        K[i, j] += Ke[_i, _j]
    end
end

function update_stiffness_matrix!(
    K::SparseMatrixCSC{Float64,Int64}, M::SparseMatrixCSC{Float64,Int64}, el_dofs, Ke, Me
)
    for (_j, j) in enumerate(el_dofs), (_i, i) in enumerate(el_dofs)
        @inbounds K[i, j] += Ke[_i, _j]
        @inbounds M[i, j] += Me[_i, _j]
    end
end

function update_force_vector!(F::Vector, fe, el_dofs)
    @inbounds for (_i, i) in enumerate(el_dofs)
        F[i] += fe[_i]
    end
end

get_NxN(N) = N' * N

function matrices_diffusion(els, nnodel, nnod)
    _, K = sparsitythermal(els, nnodel) # stiffness matrix
    M = deepcopy(K) # mass matrix
    F = fill(0.0, nnod) # force vector

    return K, M, F
end

function diffusion_immutables(gr, ndim, nvert, nnodel, nip)
    EL2NOD, nnod, nel = gr.e2n, gr.nnod, gr.nel

    ## Constant mesh values
    ScratchDifussion = ScratchThermalGlobal(
        ndim, # := model dimension
        nvert, # := number of vertices per element
        nnodel, # := number of nodes per element
        nel, # := number of elements
        nip,
    ) # := number of vertices per element

    # Shape function and their derivatives
    ni, nn, nn3 = Val(ScratchDifussion.nip), Val(ScratchDifussion.nnodel), Val(3)
    N, ∇N, dN3ds, w_ip = _get_SF(ni, nn)
    N3, = _get_SF(ni, nn3)

    SF_Diffusion = ShapeFunctionsThermal{Float64}(N, ∇N, dN3ds, w_ip, N3)

    # Degrees of freedom per element
    DoF_T = DoFs_Thermal(EL2NOD, ScratchDifussion.nnodel)

    # Diffusion sparse matrices and force vector
    KT, MT, FT = matrices_diffusion(EL2NOD, ScratchDifussion.nnodel, nnod)

    return KT, MT, FT, DoF_T, nn, SF_Diffusion, ScratchDifussion
end
