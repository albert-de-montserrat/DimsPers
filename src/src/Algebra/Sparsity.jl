
abstract type Stokes end
abstract type Thermal end

struct Sparsity{T}
    _i::Vector{Int64}
    _j::Vector{Int64}
end

function sparsitystokes(gr)
    EL2NOD, EL2NODP = gr.e2n, gr.e2nP
    ndim = 2
    nnodel = size(EL2NOD, 1)
    nUdofel = ndim * nnodel
    nPdofel = size(EL2NODP, 1)

    KKidx, GGidx, MMidx, invMMidx = _create_stokes_matrices(
        EL2NOD, EL2NODP, nUdofel, nPdofel, ndim
    )

    K = sparse(KKidx._i, KKidx._j, 1.0)
    G = sparse(GGidx._i, GGidx._j, 1.0)
    M = sparse(MMidx._i, MMidx._j, 1.0)
    invM = sparse(invMMidx._i, invMMidx._j, 1.0)

    return KKidx,
    GGidx, MMidx, invMMidx, K + tril(K, -1)', G, M + tril(M, -1)',
    invM + tril(invM, -1)'
end

@inline function _create_stokes_matrices(EL2NOD, EL2NODP, nUdofel, nPdofel, ndim)
    nel = size(EL2NOD, 2)
    dummy = 1:nUdofel
    indx_j = dummy' .* ones(Int64, nUdofel, nUdofel)
    indx_i = dummy .* ones(Int64, nUdofel, nUdofel)

    indx_i = tril(indx_i)
    indxx_i = vec(indx_i)
    filter!(x -> x > 0, indxx_i)
    indx_j = tril(indx_j)
    indxx_j = vec(indx_j)
    filter!(x -> x > 0, indxx_j)

    EL2DOF = Array{Int64}(undef, nUdofel, nel)
    EL2DOF[1:ndim:nUdofel, :] .= @. ndim * (EL2NOD - 1) + 1
    EL2DOF[2:ndim:nUdofel, :] .= @. ndim * (EL2NOD - 1) + 2
    K_i = copy(@views EL2DOF[vec(indxx_i), :])
    K_j = copy(@views EL2DOF[vec(indxx_j), :])
    indx = K_i .< K_j
    tmp = copy(@views K_j[indx])
    K_j[indx] = copy(@views K_i[indx])
    K_i[indx] = tmp
    KKidx = Sparsity{Stokes}(vec(K_i), vec(K_j))
    #  convert triplet data to sparse global stiffness matrix (assembly)
    # KK         = sparse(vec(K_i) , vec(K_j), vec(K_all))

    #==========================================================================
    ASSEMBLY OF GLOBAL GRADIENT MATRIX
    ==========================================================================#
    G_i = repeat(EL2DOF, nPdofel, 1) # global velocity dofs
    indx_j = repeat(collect(1:nPdofel)', nUdofel, 1)
    G_j = @views EL2NODP[indx_j, :] # global pressure nodes
    GGidx = Sparsity{Stokes}(vec(G_i), vec(G_j))
    # convert triplet data to sparse global gradient matrix (assembly)
    # GG     = sparse(vec(G_i) , vec(G_j) ,vec(G_all))

    #==========================================================================
    ASSEMBLY OF GLOBAL (1/VISCOSITY)-SCALED MASS MATRIX
    ==========================================================================#
    dummy = 1:nPdofel
    indx_j = dummy' .* ones(Int64, nPdofel, nPdofel)
    indx_i = dummy .* ones(Int64, nPdofel, nPdofel)

    indx_i = tril(indx_i)
    indxx_i = vec(indx_i)
    filter!(x -> x > 0, indxx_i)
    indx_j = tril(indx_j)
    indxx_j = vec(indx_j)
    filter!(x -> x > 0, indxx_j)
    M_i = copy(@views EL2NODP[indxx_i, :])
    MM_i = vec(M_i)
    M_j = copy(@views EL2NODP[indxx_j, :])
    MM_j = vec(M_j)
    indx = MM_i .< MM_j
    tmp = copy(@views MM_j[indx])
    M_j[indx] = copy(@views MM_i[indx])
    M_i[indx] = tmp
    MMidx = Sparsity{Stokes}(vec(M_i), vec(M_j))
    # convert triplet data to sparse global matrix
    # MM        = sparse(vec(M_i) , vec(M_j) , vec(M_all))

    #==========================================================================
    ASSEMBLY OF GLOBAL INVERSE MASS MATRIX
    ==========================================================================#
    dummy = 1:nPdofel
    indx_j = dummy' .* ones(Int64, nPdofel, nPdofel)
    # indx_I = dummy .* ones(Int64,nPdofel,nPdofel)
    M_i = vec(view(EL2NODP, indx_j, :))
    M_j = vec(view(EL2NODP, indx_j, :))
    invMMidx = Sparsity{Stokes}(vec(M_i), vec(M_j))

    return KKidx, GGidx, MMidx, invMMidx
end

@inline function sparsitythermal(EL2NOD, nnodel)
    nel = size(EL2NOD, 2)
    dummy = 1:nnodel
    indx_j = dummy' .* ones(Int64, nnodel, nnodel)
    indx_i = dummy .* ones(Int64, nnodel, nnodel)
    indx_i = tril(indx_i)
    indxx_i = vec(indx_i)
    filter!(x -> x > 0, indxx_i)
    indx_j = tril(indx_j)
    indxx_j = vec(indx_j)
    filter!(x -> x > 0, indxx_j)
    CM_i = copy(@views EL2NOD[indxx_i, :])
    CMM_i = vec(CM_i)
    CM_j = copy(@views EL2NOD[indxx_j, :])
    CMM_j = vec(CM_j)
    indx = CMM_i .< CMM_j
    tmp = copy(@views CMM_j[indx])
    CMM_j[indx] = CM_i[indx]
    CMM_i[indx] = tmp
    K = sparse(CMM_i, CMM_j, 1.0)
    K .+= tril(K, -1)'

    return Sparsity{Thermal}(CMM_i, CMM_j), K
end
