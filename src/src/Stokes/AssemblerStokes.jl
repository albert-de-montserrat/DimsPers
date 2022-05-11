function solve_stokes_threaded(
    color_list,
    KS,
    GS,
    MS,
    FS,
    TT,
    DoF_U,
    DoF_P,
    coordinates,
    ScratchStokes::ScratchThermalGlobal,
    SF_Stokes::ShapeFunctionsThermal,
    U,
    Upolar,
    Ucartesian,
    P,
    g,
    ρ,
    η,
    𝓒,
    UBC,
    to,
)

    # unpack boundary conditions
    ∂Ωu = UBC.Ω
    ufix = UBC.vfix
    ufree = UBC.ifree
    # unpack fixed element coordinates
    θ, r = coordinates.θ, coordinates.r

    @timeit to "Thermal diffusion threaded" begin
        # Reset Matrices
        fill!(KS.nzval, 0.0)
        fill!(GS.nzval, 0.0)
        fill!(MS.nzval, 0.0)
        fill!(FS, 0.0)

        ## Element assembly
        @timeit to "Assembly" stokes_assembly_threaded!(
            color_list,
            KS,
            MS,
            GS,
            FS,
            DoF_U,
            DoF_P,
            coordinates,
            ScratchStokes,
            SF_Stokes,
            g,
            ρ,
            η,
            𝓒,
        )

        @timeit to "BCs" begin
            KS, GS, FS = _prepare_matrices(KS, GS, FS, TT)
            U, FS = _apply_bcs(U, KS, FS, ∂Ωu, ufix)
        end

        @timeit to "PCG solver" U, P = StokesPcCG(U, P, KS, MS, GS, FS, ufree)

        @timeit to "Remove net rotation" U, Ucart, Upolar, Ucartesian = updatevelocity2(
            U, Ucartesian, Upolar, ρ, TT, coordinates, gr
        )
    end
    return T, to
end

"""
assembly_threaded!(color_list, K, M, F, dof, coordinates::ElementCoordinates, 
VarT::ThermalParameters, A::ScratchThermalGlobal, 
B::ShapeFunctionsThermal, valA, ρ, Δt)

Threaded assembly of FEM matrices for thermal diffusion. Based on a colored mesh.
"""
function stokes_assembly_threaded!(
    color_list,
    KS,
    MS,
    GS,
    FS,
    DoF_U,
    DoF_P,
    coordinates::ElementCoordinates,
    ScratchStokes::ScratchThermalGlobal,
    SF_Stokes::ShapeFunctionsStokes,
    g,
    ρ,
    η,
    𝓒,
)
    for color in color_list
        Threads.@threads for element in color
            stokes_assemble_element!(
                element,
                KS,
                MS,
                GS,
                FS,
                DoF_U,
                DoF_P,
                coordinates.θ,
                coordinates.r,
                ScratchStokes,
                SF_Stokes,
                g,
                ρ,
                η,
                𝓒,
            )
        end
    end

    return KS, GS, MS, FS
end

mydet(A::SMatrix{2,2,T,4}) where {T} = A[1] * A[4] - A[2] * A[3]

function stokes_assemble_element!(
    iel,
    KS,
    MS,
    GS,
    FS,
    DoF_U,
    DoF_P,
    θ,
    r,
    A::ScratchThermalGlobal,
    SF_Stokes::ShapeFunctionsStokes,
    g,
    ρ,
    η,
    𝓒,
)
    N, ∇N, NP, dN3ds, w_ip, N3 = SF_Stokes.N,
    SF_Stokes.∇N, SF_Stokes.NP, SF_Stokes.dN3ds, SF_Stokes.w_ip,
    SF_Stokes.N3

    valU, valP = Val(12), Val(3)

    # Polar coordinates of element nodes
    θ_el = @SVector [θ[i, iel] for i in 1:3]
    r_el = @SVector [r[i, iel] for i in 1:3]
    coords = SMatrix{3,2}([θ_el r_el])

    # Jacobian n. 1 (p:=polar, l:=local): reference element --> current element
    J_pl = dN3ds * coords
    detJ_pl = mydet(J_pl)

    R_21 = r_el[2] - r_el[1]  # = -detJa_PL*deta_dth
    R_31 = r_el[3] - r_el[1]  # =  detJa_PL*dxi_dth
    Th_31 = θ_el[3] - θ_el[1] # = -detJa_PL*dxi_dr
    Th_21 = θ_el[2] - θ_el[1] # =  detJa_PL*deta_dr

    # the Jacobian ∂ξ∂θ to transform local (ξ, η) into global (θ,r) derivatives
    #     ∂ξ∂θ = [ R_31    -R_21
    #             -Th_31    Th_21] / detJa_PL
    # ∂ξ∂θ = @SMatrix [(r_el[3]-r_el[1]) (r_el[2]-r_el[1])
    #                  (θ_el[3]-θ_el[1])  (θ_el[2]-θ_el[1])]

    Ke = empty_element_matrices(valU)
    Ge = empty_element_matrices(valU, valP)
    Me = empty_element_matrices(valP)
    fe = empty_element_force_vector(valU)

    Pdofs = DoF_P[iel]
    Udofs = DoF_U[iel]
    ρ_el = @SVector [ρ[Pdofs[i]] for i in 1:3]

    # INTEGRATION LOOP
    @inbounds for ip in 1:(A.nip)
        # Unpack shape functions 
        N_ip = N[ip]
        NP_ip = NP[ip]
        N3_ip = N3[ip]
        ∇N_ip = ∇N[ip]

        # ρ at ith integration point
        ρ_ip = dot(ρ_el, N3_ip)
        η_ip = _element_viscosity(η, gr.e2n, PhaseID, iel, N3_ip)

        # Polar coordinates of the integration points
        θ_ip = dot(θ_el, N3_ip)
        r_ip = dot(r_el, N3_ip)
        cos_ip = cos(θ_ip)
        sin_ip = sin(θ_ip)
        cos_ip_r_ip = cos_ip / r_ip
        sin_ip_r_ip = sin_ip / r_ip
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
        D = DMatrix(𝓒, iel, ip, Val(η))
        B = @SMatrix [
            ∂N∂x[1, 1] 0 ∂N∂x[1, 2] 0 ∂N∂x[1, 3] 0 ∂N∂x[1, 4] 0 ∂N∂x[1, 5] 0 ∂N∂x[1, 6] 0
            0 ∂N∂x[2, 1] 0 ∂N∂x[2, 2] 0 ∂N∂x[2, 3] 0 ∂N∂x[2, 4] 0 ∂N∂x[2, 5] 0 ∂N∂x[2, 6]
            ∂N∂x[2, 1] ∂N∂x[1, 1] ∂N∂x[2, 2] ∂N∂x[1, 2] ∂N∂x[2, 3] ∂N∂x[1, 3] ∂N∂x[2, 4] ∂N∂x[1, 4] ∂N∂x[2, 5] ∂N∂x[1, 5] ∂N∂x[2, 6] ∂N∂x[1, 6]
        ]

        # Element matrices
        Ke += (B' * D * B) * ω
        Ge += (vec(∂N∂x) .* NP_ip) * ω
        Me += (NP_ip .* NP_ip') * (ω / η_ip)

        # Force vector -- right hand side
        dummy1 = @SVector [
            sin_ip,
            cos_ip,
            sin_ip,
            cos_ip,
            sin_ip,
            cos_ip,
            sin_ip,
            cos_ip,
            sin_ip,
            cos_ip,
            sin_ip,
            cos_ip,
        ]
        dummy2 = @SVector [
            N_ip[1],
            N_ip[1],
            N_ip[2],
            N_ip[2],
            N_ip[3],
            N_ip[3],
            N_ip[4],
            N_ip[4],
            N_ip[5],
            N_ip[5],
            N_ip[6],
            N_ip[6],
        ]
        fe += (g * ω * ρ_ip * cos_ip) * (dummy2 .* dummy1)
    end

    # Update global matrices
    update_KS!(KS, Udofs, Ke)
    update_KS!(MS, Pdofs, Me)
    update_GS!(GS, Udofs, Pdofs, Ge)
    return update_force_vector!(FS, fe, Udofs)
end

function update_KS!(K::SparseMatrixCSC{Float64,Int64}, Udofs, Ke)
    for (_j, j) in enumerate(Udofs), (_i, i) in enumerate(Udofs)
        @inbounds @fastmath K[i, j] += Ke[_i, _j]
    end
end

function update_GS!(G::SparseMatrixCSC{Float64,Int64}, Udofs, Pdofs, Ge)
    for (_j, j) in enumerate(Pdofs), (_i, i) in enumerate(Udofs)
        @inbounds @fastmath G[i, j] += Ge[_i, _j]
    end
end

function update_MS!(M::SparseMatrixCSC{Float64,Int64}, Pdofs, Me)
    for (_j, j) in enumerate(Pdofs), (_i, i) in enumerate(Pdofs)
        @inbounds @fastmath M[i, j] += Me[_i, _j]
    end
end

function stokes_immutables(gr, nnod, ndim, nvert, nnodel, nel, nip)
    # Constant mesh values
    ScratchStokes = ScratchThermalGlobal(
        ndim, # := model dimension
        nvert, # := number of vertices per element
        nnodel, # := number of nodes per element
        nel, # := number of elements
        nip,
    ) # := number of vertices per element

    ni, nn, nnP, nn3 = Val(nip), Val(nnodel), Val(3), Val(3)
    N, ∇N, _, w_ip = _get_SF(ni, nn)
    NP, _, _, _ = _get_SF(ni, nnP)
    N3, _, dN3ds, _ = _get_SF(ni, nn3)

    # Shape function and their derivatives
    SF_Stokes = ShapeFunctionsStokes(N, ∇N, NP, dN3ds, w_ip, N3)

    # Degrees of freedom per element
    DoF_U = DoFs_Velocity(gr.e2n)
    DoF_P = DoFs_Pressure(gr.e2nP)
    # Diffusion sparse matrices and force vector
    _, _, _, _, KS, GS, MS, _ = sparsitystokes(gr)
    FS = Vector{Float64}(undef, 2 * nnod)

    return KS, GS, MS, FS, DoF_U, DoF_P, nn, SF_Stokes, ScratchStokes
end

function stress_shape_functions(; nip=6)
    ni, nn, nnP, nn3 = Val(nip), Val(6), Val(3), Val(3)
    N, ∇N, _, w_ip = _get_SF(ni, nn)
    NP, = _get_SF(ni, nnP)
    N3, _, dN3ds, = _get_SF(ni, nn3)

    # Shape function and their derivatives
    return ShapeFunctionsStress(N, ∇N, NP, dN3ds, w_ip, N3)
end

function DMatrix(𝓒, iel, ip, ::Val{Anisotropic})
    @SMatrix [
        𝓒.η11[iel, ip] 𝓒.η13[iel, ip] 𝓒.η15[iel, ip]
        𝓒.η13[iel, ip] 𝓒.η33[iel, ip] 𝓒.η35[iel, ip]
        𝓒.η15[iel, ip] 𝓒.η35[iel, ip] 𝓒.η55[iel, ip]
    ]
end

DMatrix(𝓒, iel, ip, ::Val{Isotropic}) = @SMatrix [
    4/3 -2/3 0
    -2/3 4/3 0
    0 0 1
]
