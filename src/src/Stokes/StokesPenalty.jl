# ==================================================================================================================
function solveStokes_penalty(
    U,
    P,
    θ,
    r,
    Ucartesian,
    Upolar,
    g,
    ρ,
    η,
    𝓒,
    theta,
    rr,
    TT,
    ∂Ωu,
    ufix,
    ifree,
    PhaseID,
    EL2NOD,
    EL2NODP,
    KKidx,
    GGidx,
    invMMidx,
    to,
)
    @timeit to "Stokes" begin
        @timeit to "Assembly" KK, GG, invMM, Rhs = assembly_stokes_penalty(
            EL2NOD, EL2NODP, theta, rr, g, ρ, η, 𝓒, PhaseID, KKidx, GGidx, invMMidx
        )
        @timeit to "BCs" begin
            KK, GG, Rhs = _prepare_matrices_penalty(KK, GG, Rhs, TT)
            U, Rhs = _apply_bcs(U, KK, Rhs, ∂Ωu, ufix)
        end
        @timeit to "Solve" U, P = PowellHesteness(U, P, KK, invMM, GG, Rhs, ifree)
        @timeit to "Remove net rotation" U, Ucart, Upolar, Ucartesian = updatevelocity(
            U, Ucartesian, Upolar, θ, r, TT
        )
        # @timeit to "Remove nullspace"  U, Ucart, Upolar, Ucartesian = updatevelocity2(U, Ucartesian, Upolar, ρ, EL2NOD, TT, theta, rr, r)

    end
    return Ucartesian, Upolar, U, Ucart, P, to
end
# ==================================================================================================================

@inline function _apply_bcs_penalty(U, KK, Rhs, ∂Ω, vfix)
    U[∂Ω] = vfix # write prescribed temperature values
    Rhs -= KK[:, ∂Ω] * vfix
    return U, Rhs
end

@inline function _prepare_matrices_penalty(KK, GG, Fb, TT)
    dropzeros!(GG)
    KK = SparseMatrixCSC(Symmetric(KK, :L))
    dropzeros!(KK)
    transTT = SparseMatrixCSC(TT')
    KK = transTT * KK * TT
    GG = transTT * GG
    Fb = transTT * Fb
    KK = SparseMatrixCSC(Symmetric(KK, :L))
    return KK, GG, Fb
end

function assembly_stokes_penalty(
    EL2NOD, EL2NODP, theta, r, g, ρ, η, 𝓒, PhaseID, KKidx, GGidx, invMMidx
)
    penalty = 1e6
    # ============================================ MODEL AND BLOCKING PARAMETERS
    ndim = 2
    # nvert = 3
    # EL2NOD0 = deepcopy(EL2NOD)
    nnodel = size(EL2NOD, 1)
    nel = size(EL2NOD, 2)
    nelblk = min(nel, 1200)
    nblk = ceil(Int, nel / nelblk)
    il = one(nelblk)
    iu = nelblk
    # =========== PREPARE INTEGRATION POINTS & DERIVATIVES wrt LOCAL COORDINATES
    nip = 6
    # nUdof = ndim*maximum(EL2NOD)
    # nPdof = maximum(EL2NODP)
    nUdofel = ndim * nnodel
    nPdofel = size(EL2NODP, 1)
    ni, nn, nnP, nn3 = Val(nip), Val(nnodel), Val(3), Val(3)
    N, dNds, _, w_ip = _get_SF(ni, nn)
    NP, _, _, _ = _get_SF(ni, nnP)
    N3, _, dN3ds, _ = _get_SF(ni, nn3)
    # ============================================================= ALLOCATIONS
    detJ_PL = Vector{Float64}(undef, nelblk)
    R_31 = similar(detJ_PL) # =  detJa_PL*dxi_dth
    R_21 = similar(detJ_PL) # = -detJa_PL*deta_dth
    Th_31 = similar(detJ_PL) # = -detJa_PL*dxi_dr
    Th_21 = similar(detJ_PL) # =  detJa_PL*deta_dr
    # NxN = Array{Float64,2}(undef,3,3)
    # ang_dist        = Array{Float64,2}(undef,3,nelblk)
    # VCOORD_th       = similar(ang_dist)
    # VCOORD_r        = similar(ang_dist)
    dNdx = Array{Float64,2}(undef, nelblk, size(dNds[1], 2))
    dNdy = similar(dNdx)
    ω = Vector{Float64}(undef, Int(nelblk))
    invJx_double = Array{Float64,2}(undef, Int(nelblk), ndim)    # storage for x-components of Jacobi matrix
    invJz_double = similar(invJx_double)                     # storage for z-components of Jacobi matrix  
    # ==================================== STORAGE FOR DATA OF ELEMENTS IN BLOCK
    K_blk = fill(0.0, nelblk, Int(nUdofel * (nUdofel + 1) / 2))
    # symmetric stiffness matrix, dim=vel dofs, but only upper triangle
    M_blk = fill(0.0, nelblk, Int(nPdofel * (nPdofel + 1) / 2))
    # symmetric mass matrix, dim=pressure nodes, but only upper triangle
    G_blk = fill(0.0, nelblk, nPdofel * nUdofel)
    # asymmetric gradient matrix, vel dofs x pressure nodes
    Fb_blk = fill(0.0, nelblk, nUdofel)
    # storage for entries in bouyancy force vector    
    invM_blk = fill(0.0, nelblk, nPdofel * nPdofel)
    invMG_blk = fill(0.0, nelblk, nPdofel * nUdofel)

    # ====================================  STORAGE FOR DATA OF ALL ELEMENT MATRICES/VECTORS
    K_all = Array{Float64,2}(undef, Int(nUdofel * (nUdofel + 1) / 2), nel)
    M_all = Array{Float64,2}(undef, Int(nPdofel * (nPdofel + 1) / 2), nel)
    G_all = Array{Float64,2}(undef, nUdofel * nPdofel, nel)
    Fb_all = Array{Float64,2}(undef, nUdofel, nel)
    invM_all = Array{Float64,2}(undef, nPdofel * nPdofel, nel)

    #=========================================================================
    BLOCK LOOP - MATRIX COMPUTATION
    =========================================================================#
    for ib in 1:nblk
        #===========================================================================
        CALCULATE JACOBIAN, ITS DETERMINANT AND INVERSE
        ===========================================================================#
        """
        NOTE: For triangular elements with non-curved edges the Jacobian is
              the same for all integration points (i.e. calculated once
              before the integration loop). Further, linear 3-node shape are
              sufficient to calculate the Jacobian.
        """
        # idx       = @views EL2NOD[1:nvert,il:iu]
        # VCOORD_th = reshape(theta[idx],nvert,nelblk)
        # VCOORD_r  = reshape(r[idx],nvert,nelblk)
        VCOORD_th = view(theta, :, il:iu)
        VCOORD_r = view(r, :, il:iu)

        J_th = (VCOORD_th' * dN3ds')
        J_r = (VCOORD_r' * dN3ds')

        _fill_R_J!(J_th, J_r, VCOORD_th, VCOORD_r, R_31, R_21, Th_31, Th_21, detJ_PL)

        # ------------------------ NUMERICAL INTEGRATION LOOP (GAUSS QUADRATURE)        
        _ip_loop_penalty!(
            K_blk,
            M_blk,
            G_blk,
            Fb_blk,
            g,
            η,
            𝓒,
            ρ,
            EL2NOD,
            PhaseID,
            il:iu,
            dNds,
            w_ip,
            nip,
            nnodel,
            nelblk,
            nPdofel,
            nUdofel,
            dNdx,
            dNdy,
            N,
            N3,
            NP,
            ω,
            invJx_double,
            invJz_double,
            detJ_PL,
            R_21,
            R_31,
            Th_21,
            Th_31,
            VCOORD_th,
            VCOORD_r,
        )

        _invM!(invM_blk, M_blk, detJ_PL)

        _fill_invMxG!(invMG_blk, invM_blk, G_blk, nPdofel, nUdofel)

        _fill_KplusGxinvMxGt(K_blk, invMG_blk, G_blk, nPdofel, nUdofel, penalty)

        # --------------------------------------- WRITE DATA INTO GLOBAL STORAGE
        @views begin
            K_all[:, il:iu] .= K_blk'
            G_all[:, il:iu] .= G_blk'
            M_all[:, il:iu] .= M_blk'
            Fb_all[:, il:iu] .= Fb_blk'
            invM_all[:, il:iu] .= invM_blk'
        end

        # -------------------------------------------------- RESET BLOCK MATRICES
        fill!(K_blk, 0.0)
        fill!(G_blk, 0.0)
        fill!(M_blk, 0.0)
        fill!(Fb_blk, 0.0)
        fill!(invM_blk, 0.0)
        fill!(invMG_blk, 0.0)

        # -------------------------------- READJUST START, END AND SIZE OF BLOCK
        il += nelblk
        if ib == nblk - 1
            # ------------ Account for different number of elements in last block
            nelblk = nel - il + 1  # new block size
            # --------------------------------- Reallocate at blocks at the edge 
            detJ_PL = Vector{Float64}(undef, nelblk)
            R_31 = similar(detJ_PL)
            R_21 = similar(detJ_PL)
            Th_31 = similar(detJ_PL)
            Th_21 = similar(detJ_PL)
            #    ang_dist = Array{Float64,2}(undef,3,nelblk)
            dNdx = Array{Float64,2}(undef, nelblk, size(dNds[1], 2))
            dNdy = similar(dNdx)
            K_blk = fill(0.0, nelblk, Int(nUdofel * (nUdofel + 1) / 2))
            M_blk = fill(0.0, nelblk, Int(nPdofel * (nPdofel + 1) / 2))
            G_blk = fill(0.0, nelblk, nPdofel * nUdofel)
            Fb_blk = fill(0.0, nelblk, nUdofel)
            ω = Vector{Float64}(undef, Int(nelblk))
            invJx_double = Array{Float64,2}(undef, Int(nelblk), ndim)
            invJz_double = similar(invJx_double)
            invM_blk = fill(0.0, nelblk, nPdofel * nPdofel)
            invMG_blk = fill(0.0, nelblk, nPdofel * nUdofel)
        end
        iu += nelblk
    end # end block loop

    # #===========================================================================
    # ASSEMBLY OF GLOBAL SPARSE MATRICES AND RHS-VECTOR
    # ===========================================================================#
    KK, GG, invMM, Fb = _create_stokes_matrices_penalty(
        EL2NOD, nUdofel, ndim, KKidx, GGidx, invMMidx, K_all, G_all, invM_all, Fb_all
    )

    return KK, GG, invMM, Fb
end # END OF ASSEMBLY FUNCTION

function _ip_loop_penalty!(
    K_blk,
    M_blk,
    G_blk,
    Fb_blk,
    g,
    η,
    𝓒,
    ρ,
    EL2NOD,
    PhaseID,
    els,
    dNds,
    w_ip,
    nip,
    nnodel,
    nelblk,
    nPdofel,
    nUdofel,
    dNdx,
    dNdy,
    N,
    N3,
    NP,
    ω,
    invJx_double,
    invJz_double,
    detJ_PL,
    R_21,
    R_31,
    Th_21,
    Th_31,
    VCOORD_th,
    VCOORD_r,
)
    sin_ip = similar(detJ_PL)
    cos_ip = similar(detJ_PL)

    for ip in 1:nip
        N_ip = N3[ip]
        # NP_blk = repeat(NP[ip],nelblk,1)
        NP_blk = NP[ip] .* ones(nelblk)

        #=======================================================================
        PROPERTIES OF ELEMENTS AT ip-TH EVALUATION POINT
        =======================================================================#
        Dens_blk = _element_density(ρ, EL2NOD, PhaseID, els, NP[ip])
        Visc_blk = _element_viscosity(η, EL2NOD, PhaseID, els, NP[ip])
        #  Gravitational force at ip-th integration point
        Fg_blk = g * Dens_blk

        #==========================================================================================================
        CALCULATE 2nd JACOBIAN (FROM CARTESIAN TO POLAR COORDINATES --> curved edges), ITS DETERMINANT AND INVERSE
        ===========================================================================================================
        NOTE: For triangular elements with curved edges the Jacobian needs to be computed at each integration
               point (inside the integration loop). =#
        th_ip = VCOORD_th' * N_ip'
        r_ip = VCOORD_r' * N_ip'

        _derivative_weights!(
            dNds[ip],
            ω,
            dNdx,
            dNdy,
            w_ip[ip],
            th_ip,
            r_ip,
            sin_ip,
            cos_ip,
            R_21,
            R_31,
            Th_21,
            Th_31,
            detJ_PL,
            invJx_double,
            invJz_double,
        )

        _fill_Kblk!(K_blk, 𝓒, Visc_blk, Val(η), ω, dNdx, dNdy, nnodel, els, ip)

        _fill_Gblk!(G_blk, NP_blk, ω, dNdx, dNdy, nPdofel, nUdofel, nnodel)

        _fill_Mblk_penalty!(M_blk, NP_blk, ω, nPdofel)

        _fill_Fbblk!(Fb_blk, Fg_blk, N[ip], ω, sin_ip, cos_ip, nUdofel)
    end # end integration point loop
end # ENF OF IP_LOOP FUNCTION

@inline function _fill_Mblk_penalty!(M_blk, NP_blk, ω, nPdofel)
    # "weight" divided by viscosity at integration point in all elements
    indx = 1
    for i in 1:nPdofel
        @inbounds @simd for j in i:nPdofel
            M_blk[:, indx] .+= ω .* view(NP_blk, :, i) .* view(NP_blk, :, j)
            indx += 1
        end
    end
end

function _invM!(invM_blk, M_blk, detJ)
    TMP = 1.0 ./ detJ
    M_blk = M_blk .* TMP
    # M_blk = M_blk .* TMP[:,ones[1,nPdofel*[nPdofel+1]/2]];

    # How to calculate the determinante of several symmetric 3x3 matrices:
    # det[M] =   M11*M22*M33 + M12*M23*M31 + M13*M21*M32
    #          - M13*M22*M31 - M12*M21*M33 - M11*M23*M32
    #   written in 1-index notation:
    #        =   M1 *M5 *M9  + M4 *M8 *M3  + M7 *M2 *M6
    #          - M7 *M5 *M3  - M4 *M2 *M9  - M1 *M8 *M6
    #   considering symmetry and using lower triangular part only
    #   [M4=M2, M7=M3, M8=M6]
    #        =   M1 *M5 *M9  + M2 *M6 *M3  + M3 *M2 *M6
    #          - M3 *M5 *M3  - M2 *M2 *M9  - M1 *M6 *M6
    #   and knowing where the 6 different values are stored in M_blk
    #   [1-->M1, 2-->M2, 3-->M3, 4-->M5, 5-->M6, 6-->M9]
    #        =   M_blk1*M_blk4*M_blk6 + M_blk2*M_blk5*M_blk3 + M_blk3*M_blk2*M_blk5
    #          - M_blk3*M_blk4*M_blk3 - M_blk2*M_blk2*M_blk6 - M_blk1*M_blk5*M_blk5
    #   re-arranging
    #        =   M_blk1 * [M_blk4*M_blk6 - M_blk5*M_blk5] 
    #          + M_blk2 * [M_blk5*M_blk3 - M_blk2*M_blk6]
    #          + M_blk3 * [M_blk2*M_blk5 - M_blk4*M_blk3]
    detM_blk = @. M_blk[:, 1] * (M_blk[:, 4] * M_blk[:, 6] - M_blk[:, 5] * M_blk[:, 5]) +
        M_blk[:, 2] * (M_blk[:, 5] * M_blk[:, 3] - M_blk[:, 2] * M_blk[:, 6]) +
        M_blk[:, 3] * (M_blk[:, 2] * M_blk[:, 5] - M_blk[:, 4] * M_blk[:, 3])
    detM_blk ./= TMP

    # The determinante is used to calculate the inverse of the symmetric 
    # 3x3 element mass matrices. The same logic as above is used.
    invM_blk[:, 1] = @views @. (M_blk[:, 4] * M_blk[:, 6] - M_blk[:, 5] * M_blk[:, 5]) /
        detM_blk
    invM_blk[:, 2] = @views @. (M_blk[:, 5] * M_blk[:, 3] - M_blk[:, 2] * M_blk[:, 6]) /
        detM_blk
    invM_blk[:, 3] = @views @. (M_blk[:, 2] * M_blk[:, 5] - M_blk[:, 4] * M_blk[:, 3]) /
        detM_blk
    invM_blk[:, 4] = @views invM_blk[:, 2]
    invM_blk[:, 5] = @views @. (M_blk[:, 1] * M_blk[:, 6] - M_blk[:, 3] * M_blk[:, 3]) /
        detM_blk
    invM_blk[:, 6] = @views @. (M_blk[:, 2] * M_blk[:, 3] - M_blk[:, 1] * M_blk[:, 5]) /
        detM_blk
    invM_blk[:, 7] = @views invM_blk[:, 3]
    invM_blk[:, 8] = @views invM_blk[:, 6]
    return invM_blk[:, 9] = @views @. (
        M_blk[:, 1] * M_blk[:, 4] - M_blk[:, 5] * M_blk[:, 5]
    ) / detM_blk
end

function _fill_invMxG!(invMG_blk, invM_blk, G_blk, nPdofel, nUdofel)
    # --------------------------invM*G'----------------------------
    for i in 1:nPdofel
        for j in 1:nUdofel
            for k in 1:nPdofel
                invMG_blk[:, (i - 1) * nUdofel + j] = @views invMG_blk[
                    :, (i - 1) * nUdofel + j
                ] + invM_blk[
                    :, (i - 1) * nPdofel + k
                ] .* G_blk[:, (k - 1) * nUdofel + j]
            end
        end
    end
end

function _fill_KplusGxinvMxGt(K_blk, invMG_blk, G_blk, nPdofel, nUdofel, penalty)
    # -----------------K = K + penalty*G*invM*G'-------------------
    indx = 1
    for i in 1:nUdofel
        for j in i:nUdofel
            for k in 1:nPdofel
                K_blk[:, indx] = @views K_blk[:, indx] +
                    penalty * G_blk[:, (k - 1) * nUdofel + i] .*
                                        invMG_blk[:, (k - 1) * nUdofel + j]
            end
            indx += 1
        end
    end
end

@inline function _create_stokes_matrices_penalty(
    EL2NOD, nUdofel, ndim, KKidx, GGidx, invMMidx, K_all, G_all, invM_all, Fb_all
)
    nel = size(EL2NOD, 2)

    #==========================================================================
    ASSEMBLY OF GLOBAL STIFFNESS MATRIX
    ==========================================================================#
    #  convert triplet data to sparse global stiffness matrix (assembly)
    KK = sparse(KKidx._i, KKidx._j, vec(K_all))

    #==========================================================================
    ASSEMBLY OF GLOBAL GRADIENT MATRIX
    ==========================================================================#
    # convert triplet data to sparse global gradient matrix (assembly)
    GG = sparse(GGidx._i, GGidx._j, vec(G_all))

    #==========================================================================
    ASSEMBLY OF GLOBAL FORCE VECTOR
    ==========================================================================#
    EL2DOF = Array{Int64}(undef, nUdofel, nel)
    @views EL2DOF[1:ndim:nUdofel, :] .= @. ndim * (EL2NOD - 1) + 1
    @views EL2DOF[2:ndim:nUdofel, :] .= @. ndim * (EL2NOD - 1) + 2
    Fb = accumarray(vec(EL2DOF), vec(Fb_all))

    #==========================================================================
    ASSEMBLY OF GLOBAL (1/VISCOSITY)-SCALED MASS MATRIX
    ==========================================================================#
    # convert triplet data to sparse global matrix
    invMM = sparse(invMMidx._i, invMMidx._j, vec(invM_all))

    return KK, GG, invMM, Fb
end

# POWELL-HESTENESS SOLVER ================================================================
@inline function PowellHesteness(U, P, KK, M⁻¹, GG, Rhs, ifree)
    itmax_PH = 50
    itmin_PH = 2
    # itnum = 0
    norm∇U = Vector{Float64}(undef, itmax_PH)

    Rhs .+= GG * P

    LL = factorize(KK[ifree, ifree])
    # LL = cholesky(KK[ifree, ifree])
    GGᵀ = GG'
    # Uifree = U[ifree]
    # Rhsifree = Rhs[ifree]
    penalty = 1e6

    for itPH in 1:itmax_PH
        @inbounds U[ifree] .= LL \ Rhs[ifree]
        ∇U = -GGᵀ * U
        # rm0space!(∇U)
        ΔP = penalty * M⁻¹ * ∇U
        Rhs .+= GG * ΔP
        P .+= ΔP
        # Check convergence -----------------------------------------------
        norm∇U[itPH] = mynorm(∇U)

        if (itPH > itmin_PH) && (norm∇U[itPH] / norm∇U[itPH - 1] > 0.1)
            println("\n", itPH, " Powell-Hesteness iterations\n")
            break
        end
    end

    # Ux = U[1:2:end]
    # Uz = U[2:2:end]
    # M = sum(mean(ρ[EL2NOD],dims=1)'.*area_el)
    # Ax =  sum(mean(ρ[EL2NOD].*Ux[EL2NOD], dims=1)'.*area_el)
    # Az =  sum(mean(ρ[EL2NOD].*Uz[EL2NOD], dims=1)'.*area_el)
    # A = @. (Ax+Az)/M
    # U .-=A 

    return U, P
end # END POWELL-HESTENESS SOLVER
