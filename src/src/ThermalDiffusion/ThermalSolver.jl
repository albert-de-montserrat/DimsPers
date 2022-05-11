# ==================================================================================================================
function solveDiffusion(
    EL2NOD,
    θStokes,
    rStokes,
    Δt,
    κ,
    dQdT,
    ρ,
    Cp,
    shear_heating,
    PhaseID,
    T,
    ∂Ωt,
    tfix,
    tfree,
    CMidx,
    to,
)
    @timeit to "Thermal diffusion" begin
        @timeit to "Assembly" CC, KK, Rhs = assembly_thermal_cylindric(
            EL2NOD, θStokes, rStokes, Δt, κ, dQdT, ρ, Cp, shear_heating, PhaseID, CMidx
        )

        @timeit to "BCs" begin
            # KK, Rhs = _prepare_matrices(KK, Rhs, CC, T, Δt)
            _prepare_matrices!(KK, Rhs, CC, T, Δt)
            _apply_bcs!(T, KK, Rhs, ∂Ωt, tfix)
        end

        # _CG!(T,KK,Rhs,tfree) # faster depending on MKL/BLAS setup and nthreads?
        @timeit to "Solve" T[tfree] .= KK[tfree, tfree] \ Rhs[tfree]
    end
    return T, to
end
# ==================================================================================================================

@inline function _apply_bcs!(T, KK, Rhs, ∂Ω, vfix)
    @views T[∂Ω] .= vfix # write prescribed temperature values
    return Rhs .-= KK[:, ∂Ω] * vfix # move BC to rhs
end

@inline function _prepare_matrices(KK, Rhs, CC, T, Δt)
    α = 1.0
    if α != 1.0
        Rhs .+= (KK - (Δt * (1 - α)) * CC) * T
    else
        Rhs .+= KK * T
    end
    KK .+= @. (Δt * α) * CC
    return KK, Rhs
end

@inline function _prepare_matrices!(KK, Rhs, CC, T, Δt)
    α = 1.0
    if α != 1.0
        Rhs .+= (KK - (Δt * (1 - α)) * CC) * T
    else
        Rhs .+= KK * T
    end
    KK .+= @. (Δt * α) * CC
    return KK, Rhs
end

function assembly_thermal_cylindric(
    EL2NOD, θStokes, rStokes, Δt, κ, dQdt, ρ, Cp, shear_heating, PhaseID, CMidx
)

    # ============================================ MODEL AND BLOCKING PARAMETERS
    ndim = 2
    nip = 6
    nnodel = size(EL2NOD, 1)
    nel = size(EL2NOD, 2)
    nelblk = min(nel, 1200)
    # nelblk  = min(nel, nblk)
    nblk = ceil(Int, nel / nelblk)
    il = one(nelblk)
    iu = nelblk

    # ======================== STORAGE FOR DATA OF ALL ELEMENT MATRICES/VECTORS
    n1 = Int(nnodel * (nnodel + 1) / 2)
    C_all = fill(0.0, n1, nel)
    M_all = fill(0.0, n1, nel)
    Rhs_all = fill(0.0, nnodel, nel)

    # =========== PREPARE INTEGRATION POINTS & DERIVATIVES wrt LOCAL COORDINATES
    ni, nn, nn3 = Val(nip), Val(nnodel), Val(3)
    N, ∇N, dN3ds, w_ip = _get_SF(ni, nn)
    N3, = _get_SF(ni, nn3)
    detJ_PL = Vector{Float64}(undef, nelblk)
    R_31 = similar(detJ_PL) # =  detJa_PL*dxi_dth
    R_21 = similar(detJ_PL) # = -detJa_PL*deta_dth
    Th_31 = similar(detJ_PL) # = -detJa_PL*dxi_dr
    Th_21 = similar(detJ_PL) # =  detJa_PL*deta_dr
    NxN = Array{Float64,2}(undef, nnodel, nnodel)
    dNdx = Array{Float64,2}(undef, nelblk, size(∇N[1], 2))
    dNdy = similar(dNdx)

    # =====================================================================
    # STORAGE FOR DATA OF ELEMENTS IN BLOCK
    # =====================================================================
    C_blk = fill(0.0, nelblk, n1)
    M_blk = fill(0.0, nelblk, n1)
    Rhs_blk = fill(0.0, nelblk, nnodel)
    ω = Vector{Float64}(undef, Int(nelblk))
    invJx_double = Array{Float64,2}(undef, Int(nelblk), ndim) # storage for x-components of Jacobi matrix
    invJz_double = similar(invJx_double) # storage for z-components of Jacobi matrix

    # =========================================================================
    # BLOCK LOOP - MATRIX COMPUTATION
    # =========================================================================
    for ib in 1:nblk

        #===========================================================================
        CALCULATE JACOBIAN, ITS DETERMINANT AND INVERSE
        ===========================================================================#
        #=
        NOTE: For triangular elements with non-curved edges the Jacobian is
              the same for all integration points (i.e. calculated once
              before the integration loop). Further, linear 3-node shape are
              sufficient to calculate the Jacobian.
        =#
        VCOORD_th = view(θStokes, :, il:iu)
        VCOORD_r = view(rStokes, :, il:iu)
        J_th = VCOORD_th' * dN3ds'
        J_r = VCOORD_r' * dN3ds'

        _fill_R_J_thermal!(
            J_th, J_r, VCOORD_th, VCOORD_r, R_31, R_21, Th_31, Th_21, detJ_PL
        )

        # ------------------------ NUMERICAL INTEGRATION LOOP (GAUSS QUADRATURE)
        _ip_loop!(
            C_blk,
            M_blk,
            Rhs_blk,
            Δt,
            κ,
            dQdt,
            ρ,
            Cp,
            shear_heating,
            EL2NOD,
            PhaseID,
            il:iu,
            N3,
            N,
            ∇N,
            w_ip,
            nip,
            nnodel,
            dNdx,
            dNdy,
            NxN,
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

        # --------------------------------------- WRITE DATA INTO GLOBAL STORAGE
        @views begin
            C_all[:, il:iu] = C_blk'
            M_all[:, il:iu] = M_blk'
            Rhs_all[:, il:iu] = Rhs_blk'
        end

        emptyblocks!(C_blk, M_blk, Rhs_blk)

        # -------------------------------- READJUST START, END AND SIZE OF BLOCK
        il += nelblk
        if ib == nblk - 1
            # ------------ Account for different number of elements in last block
            nelblk = nel - iu  # new block size
            # ------------------------- Reallocate at the edge the block sequence
            detJ_PL = Vector{Float64}(undef, nelblk)
            R_31 = similar(detJ_PL) # =  detJa_PL*dxi_dth
            R_21 = similar(detJ_PL) # = -detJa_PL*deta_dth
            Th_31 = similar(detJ_PL) # = -detJa_PL*dxi_dr
            Th_21 = similar(detJ_PL) # =  detJa_PL*deta_dr
            dNdx = Array{Float64,2}(undef, nelblk, size(∇N[1], 2))
            dNdy = similar(dNdx)
            C_blk = fill(0.0, nelblk, n1)
            M_blk = fill(0.0, nelblk, n1)
            Rhs_blk = fill(0.0, nelblk, nnodel)
            ω = Vector{Float64}(undef, Int(nelblk))
            invJx_double = Array{Float64,2}(undef, Int(nelblk), ndim) # storage for x-components of Jacobi matrix
            invJz_double = similar(invJx_double) # storage for z-components of Jacobi matrix
        end
        iu += nelblk
    end # end block loop

    #===========================================================================
    ASSEMBLY OF GLOBAL SPARSE MATRICES AND RHS-VECTOR
    ===========================================================================#
    C = vec(C_all)
    M = vec(M_all)
    R = vec(Rhs_all)
    E = vec(view(EL2NOD, 1:nnodel, :))
    CC, MM, Rhs = _thermal_matrices(C, M, R, E, CMidx)

    return CC, MM, Rhs
end # END OF ASSEMBLY FUNCTION

#===============================================================================
 ------------------------------------------------------------------------------
                        BEGINING OF ASSEMBLY SUBFUNCTIONS
 ------------------------------------------------------------------------------
===============================================================================#
@inline function _thermal_matrices(C, M, R, E, CM_i, CM_j)
    tmp = sparse(CM_i, CM_j, C)
    CC = tmp + tril(tmp, -1)' # other half is given by symmetry
    tmp = sparse(CM_i, CM_j, M)
    MM = tmp + tril(tmp, -1)' # other half is given by symmetry
    Rhs = accumarray(E, R)

    return CC, MM, Rhs
end

@inline function _thermal_matrices(C, M, R, E, CMidx)
    tmp = sparse(CMidx._i, CMidx._j, C)
    CC = tmp + tril(tmp, -1)' # other half is given by symmetry
    tmp = sparse(CMidx._i, CMidx._j, M)
    MM = tmp + tril(tmp, -1)' # other half is given by symmetry
    Rhs = accumarray(E, R)

    return CC, MM, Rhs
end

function _create_triplets(EL2NOD, nnodel)
    indx_j = Array{Int64}(undef, nnodel, nnodel)
    indx_i = Array{Int64}(undef, nnodel, nnodel)
    dummy = collect(1:nnodel)
    @inbounds for i in 1:3
        indx_j[i, :] = dummy
        indx_i[:, i] = dummy'
    end

    # indx_i      = indx_j';
    indx_i = tril(indx_i)
    indxx_i = vec(indx_i)
    filter!(x -> x > 0, indxx_i)
    indx_j = tril(indx_j)
    indxx_j = vec(indx_j)
    filter!(x -> x > 0, indxx_j)

    CM_i = copy(EL2NOD[indxx_i, :])
    CMM_i = vec(CM_i)
    CM_j = copy(EL2NOD[indxx_j, :])
    CMM_j = vec(CM_j)
    indx = CMM_i .< CMM_j
    tmp = copy(CMM_j[indx])
    CMM_j[indx] = CM_i[indx]
    CMM_i[indx] = tmp

    return CMM_i, CMM_j
end

function _ip_loop!(
    C_blk,
    M_blk,
    Rhs_blk,
    Δt,
    κ,
    dQdt,
    ρ,
    Cp,
    shear_heating,
    EL2NOD,
    PhaseID,
    els,
    N3,
    N,
    ∇N,
    w_ip,
    nip,
    nnodel,
    dNdx,
    dNdy,
    NxN,
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
    th_ip = similar(detJ_PL)
    r_ip = similar(detJ_PL)

    @inbounds for ip in 1:nip
        N_ip = N[ip]

        #=======================================================================
        PROPERTIES OF ELEMENTS AT ip-TH EVALUATION POINT
        =======================================================================#
        Cond_blk = _element_conductivity(κ, PhaseID, els)
        dQdt_blk = _element_dQdt(dQdt, PhaseID, els)
        RhoCp_blk = _element_RhoCp(ρ, Cp, EL2NOD, PhaseID, els, N_ip)
        sh_blk = view(shear_heating, els, ip)

        #===========================================================================================================
        CALCULATE 2nd JACOBIAN (FROM CARTESIAN TO POLAR COORDINATES --> curved edges), ITS DETERMINANT AND INVERSE
        ===========================================================================================================
        NOTE: For triangular elements with curved edges the Jacobian needs to be computed at each integration
               point (inside the integration loop).
        TODO: fix type stability error =#
        th_ip = VCOORD_th' * N3[ip]'
        r_ip = VCOORD_r' * N3[ip]'

        _derivative_weights!(
            ∇N[ip],
            ω,
            dNdx,
            dNdy,
            w_ip[ip],
            th_ip,
            r_ip,
            R_21,
            R_31,
            Th_21,
            Th_31,
            detJ_PL,
            invJx_double,
            invJz_double,
        )

        #=======================================================================
        3-node triangle allows lumping to improve stability
        =======================================================================#
        if nnodel == 3 # lumping
            mul!(NxN, N_ip', N_ip)
            @inbounds for ii in 1:nnodel
                NxN[ii, ii] = NxN[ii, 1] + NxN[ii, 2] + NxN[ii, 3]
            end
            @inbounds for ii in 1:nnodel, jj in 1:nnodel
                if ii != jj
                    NxN[ii, jj] = 0.0
                end
            end
        else
            mul!(NxN, N_ip', N_ip)
        end

        _fill_blk_matrices!(
            C_blk,
            M_blk,
            Rhs_blk,
            sh_blk,
            ω,
            dNdx,
            dNdy,
            NxN,
            N_ip,
            nnodel,
            Cond_blk,
            dQdt_blk,
            RhoCp_blk,
            Δt,
        )
    end # end integration point loop
end # ENF OF IP_LOOP FUNCTION

function sumrows(a::Array{Float64,2})
    nrow, ncol = size(a, 1), size(a, 2)
    b = fill(0.0, nrow)
    for i in 1:nrow, j in 1:ncol
        b[i] += a[i, j]
    end
    return b
end

function _fill_R_J_thermal!(
    J_th, J_r, VCOORD_th, VCOORD_r, R_31, R_21, Th_31, Th_21, detJ_PL
)
    VCOORD_r1 = view(VCOORD_r, 1, :)
    VCOORD_r2 = view(VCOORD_r, 2, :)
    VCOORD_r3 = view(VCOORD_r, 3, :)
    VCOORD_th1 = view(VCOORD_th, 1, :)
    VCOORD_th2 = view(VCOORD_th, 2, :)
    VCOORD_th3 = view(VCOORD_th, 3, :)

    @inbounds for i in 1:length(R_31)
        detJ_PL[i] = J_th[i, 1] * J_r[i, 2] - J_th[i, 2] * J_r[i, 1]
        R_31[i] = VCOORD_r3[i] - VCOORD_r1[i]  # =  detJa_PL*dxi_dth
        R_21[i] = VCOORD_r2[i] - VCOORD_r1[i]  # = -detJa_PL*deta_dth
        Th_31[i] = VCOORD_th3[i] - VCOORD_th1[i] # = -detJa_PL*dxi_dr
        Th_21[i] = VCOORD_th2[i] - VCOORD_th1[i] # =  detJa_PL*deta_dr
    end

    return R_31, R_21, Th_31, Th_21, detJ_PL
end

function _derivative_weights!(
    ∇N,
    ω,
    dNdx,
    dNdy,
    w_ip,
    th_ip,
    r_ip,
    R_21,
    R_31,
    Th_21,
    Th_31,
    detJ_PL,
    invJx_double,
    invJz_double,
)

    # --- Start loop (faster and more memory friendly than single - line declarations)
    for i in 1:length(th_ip)
        # -------------------------------- i-th extractions for some extra speed
        detJ = detJ_PL[i]
        invdetJ = 1 / detJ
        sin_ip = sin(th_ip[i])
        cos_ip = cos(th_ip[i])
        R_31_i = R_31[i]
        R_21_i = R_21[i]
        Th_31_i = Th_31[i]
        Th_21_i = Th_21[i]
        r_i = r_ip[i]
        invr_i = 1 / r_i
        # --------------------------------------- Inverse of the double Jacobian
        invJx_double[i, 1] = (R_31_i * cos_ip * invr_i - Th_31_i * sin_ip) * invdetJ
        invJx_double[i, 2] = (-R_21_i * cos_ip * invr_i + Th_21_i * sin_ip) * invdetJ
        invJz_double[i, 1] = (-R_31_i * sin_ip * invr_i - Th_31_i * cos_ip) * invdetJ
        invJz_double[i, 2] = (R_21_i * sin_ip * invr_i + Th_21_i * cos_ip) * invdetJ
        # ---------------------------- Numerical integration of element matrices
        ω[i] = r_i * detJ * w_ip
        # ω[i]  = detJ * w_ip
    end

    # --------------------------------------- Derivatives wrt global coordinates
    mul!(dNdx, invJx_double, ∇N)
    return mul!(dNdy, invJz_double, ∇N)
end

@inline function _fill_blk_matrices!(
    C_blk,
    M_blk,
    Rhs_blk,
    sh_blk,
    ω,
    dNdx,
    dNdy,
    NxN,
    N_ip,
    nnodel,
    Cond_blk::Float64,
    dQdt_blk::Float64,
    RhoCp_blk,
    Δt,
)
    indx = 1
    nn = size(M_blk, 1)
    @inbounds for i in 1:nnodel
        for j in i:nnodel
            # Conductivity matrices of all elements in block
            # C = B'*kappa*B, with B = [ dN1dx dN2dx dN3dx dN4dx dN5dx dN6dx;
            #                            dN1dy dN2dy dN3dy dN4dy dN5dy dN6dy ]
            NxN_ij = NxN[i, j]
            @fastmath @simd for k in 1:nn
                C_blk[k, indx] +=
                    ω[k] * Cond_blk * (dNdx[k, i] * dNdx[k, j] + dNdy[k, i] * dNdy[k, j])

                M_blk[k, indx] += ω[k] * RhoCp_blk[k] * NxN_ij #N_ip[i]*N_ip[j] # NxN_ij
            end
            indx += 1
        end
    end

    # RIGHT HAND SIDE
    if dQdt_blk != 0.0
        Rhs_blk .+= (N_ip' * (Δt * dQdt_blk * ω)')'
    end

    # Rhs_blk .+= (Δt* sh_blk.*ω)*N_ip  # shear heating

end

@inline function _fill_blk_matrices!(
    C_blk,
    M_blk,
    Rhs_blk,
    ω,
    dNdx,
    dNdy,
    NxN,
    N_ip,
    nnodel,
    Cond_blk::Array{Float64,1},
    dQdt_blk::Array{Float64,1},
    RhoCp_blk,
    Δt,
)
    indx = 1
    nn = size(M_blk, 1)
    @inbounds for i in 1:nnodel
        for j in i:nnodel
            # Conductivity matrices of all elements in block
            # C = B'*kappa*B, with B = [ dN1dx dN2dx dN3dx dN4dx dN5dx dN6dx;
            #                            dN1dy dN2dy dN3dy dN4dy dN5dy dN6dy ]
            NxN_ij = NxN[i, j]
            @fastmath @simd for k in 1:nn
                C_blk[k, indx] +=
                    ω[k] * Cond_blk * (dNdx[k, i] * dNdx[k, j] + dNdy[k, i] * dNdy[k, j])

                M_blk[k, indx] += ω[k] * RhoCp_blk[k] * NxN_ij
            end
            indx += 1
        end
    end

    # RIGHT HAND SIDE
    if sum(dQdt_blk) != 0.0
        Rhs_blk .+= @. (N_ip' * (Δt * dQdt_blk * ω)')'
    end
end

# @inline function  _element_conductivity(κ,PhaseID,els)
#     # Use the constant conductivity defined in PHYSICS.K
#     length(PhaseID)==1 ? Cond_blk = κ[PhaseID]' : Cond_blk = vec(κ[PhaseID[els]]')

#     return  Cond_blk
# end

function _element_conductivity(κ, PhaseID, els)
    return length(PhaseID) == 1 ? κ[PhaseID]' : vec(κ[PhaseID[els]]')
end

@inline function _element_RhoCp(ρ, Cp, EL2NOD, PhaseID, els, Nip)
    Dens_blk = vec(ρ[view(EL2NOD, :, els)]' * Nip')

    # Use the constant conductivity defined in PHYSICS.K
    length(PhaseID) == 1 ? Cp_blk = Cp[PhaseID]' : Cp_blk = vec(Cp[PhaseID[els]]')
    # Cp_blk = ifelse(length(PhaseID)==1,
    #                 Cp[PhaseID]',
    #                 vec(Cp[PhaseID[els]]'))
    # if length(PhaseID)==1
    #     # Dens_blk    = ρ[PhaseID]'
    #     Cp_blk      = Cp[PhaseID]'
    # else
    #     # Dens_blk    = vec(ρ[PhaseID[els]]')
    #     Cp_blk      = vec(Cp[PhaseID[els]]')
    # end

    # Make sure a column-vector is returned
    return Dens_blk .* Cp_blk
end

@inline function _element_dQdt(dQdt, PhaseID, els)
    # Use the constant conductivity defined in PHYSICS.K
    length(PhaseID) == 1 ? dQdt_blk = dQdt[PhaseID]' : dQdt_blk = dQdt[PhaseID[els]]'
    return dQdt_blk
end

@inline function emptyblocks!(C_blk, M_blk, Rhs_blk)
    @inbounds for i in eachindex(C_blk)
        C_blk[i] = 0.0
        M_blk[i] = 0.0
    end
    @inbounds for i in eachindex(Rhs_blk)
        Rhs_blk[i] = 0.0
    end
end
