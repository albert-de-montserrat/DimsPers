
function StokesPcCG_gpu(U, P, KK, MM, GG, Rhs, ifree)
    rtol_Pat = 1e-8
    itmax_Pat = 200
    itmin_Pat = 10
    itnum = 0

    nU = length(Rhs)

    # allocate some stuff
    Ud = CUDA.zeros(nU)
    z = CUDA.zeros(nU)
    Pd = CUDA.zeros(length(P))

    # move FEM matrices to device
    KKd = CuSparseMatrixCSC{Float32}(KK[ifree, ifree])
    GGd = CuSparseMatrixCSC{Float32}(GG)
    GGtransp = CuSparseMatrixCSC{Float32}(sparse(GG'))
    negGGtransp = CuSparseMatrixCSC{Float32}(sparse(-GG'))
    Rhsd = CuArray{Float32}(Rhs)

    # guess of the velocity field 
    sol, = Krylov.cg(KKd, Rhsd[ifree])
    Ud[ifree] .= sol

    # initial pressure residual vector
    r = negGGtransp * Ud
    rm0space!(r)
    Prms = norm(r) # norm of r_i
    tol = Prms * rtol_Pat
    # get preconditioner
    MM, pc = _preconditioner(:jacobi, MM)
    MMd = CuSparseMatrixCSC{Float32}(MM)
    pcd = CuArray{Float32}(pc)

    # Begin of Patera pressure iterations 
    # d = _precondition(pc, MM, r) # precondition residual
    d = _precondition(pcd, r) # precondition residual
    q = deepcopy(d)  # define FIRST search direction q
    rlast = similar(r)
    for itPat in 1:itmax_Pat
        itnum += 1

        rm0space!(q)
        rd = dot(r, d) # numerator for alpha

        #= Perform the S times q multiplication ======================    
            S cannot be calculated explicitly, since Kinv cannot be formed
            Sq = S * q = (Gᵀ * Kinv * G) * q
            Hence, the muliplicatipon is done in 3 steps:
            (1) y   = G*q
            (2) K z = y
            (3) Sq  = Gᵀ*z
        =============================================================#
        y = GGd * q # (1) y = G*q
        sol, = Krylov.cg(KKd, y[ifree])
        z[ifree] .= sol
        # _CholeskyWithFactorization!(z, F, y, ifree) # (2) Solve K z = y
        Sq = GGtransp * z # (3) Sq  = G'*z
        #=============================================================#
        qSq = dot(q, Sq) # denominator to calculate alpha
        copyto!(rlast, r) # needed for Polak-Ribiere version 1
        α = rd / qSq # steps size in direction q

        # Update solution and residual  
        Pd .+= α .* q
        Ud .+= α .* z
        r .-= α .* Sq

        # remove nullspace
        rm0space!(r)

        # Check convergence 
        # if norm(r) < tol && itPat>=itmin_Pat
        #     println("\n", itnum," CG iterations\n")
        #     break
        # end

        d .= _precondition(pcd, r) # precondition residual
        # Make new search direction q S-orthogonal to all previous q's
        β = dot(r - rlast, d) / rd # Polak-Ribiere version 1        
        q .= xpy(q, d, β)
    end

    return Array(Ud), Array(Pd)
end

# Function takes out the mean of vector a
# (thereby removes the constant pressure mode, i.e. the nullspace)
function rm0space!(a::CuArray)
    m = sum(a) / length(a)
    return a .= a .- m
end

# PRECONDITIONING: JACOBI ITERATIONS 
@inline function _precondition(pcd::CuArray, MMd::CuSparseMatrixCSC, r::CuArray)
    weight = (0.2500, 0.4000, 0.7500)
    # d = r .* pcd
    d = r .* pcd .+ weight[1] .* pcd .* (r .- MMd * d)
    d .+= weight[2] .* pcd .* (r .- MMd * d)
    d .+= weight[3] .* pcd .* (r .- MMd * d)
    return d
end

@inline _precondition(pc::CuArray, r::CuArray) = r .* pc

@inline function _updatesolution_gpu!(P, U, r, q, z, Sq, α)
    n1 = length(P)
    n2 = length(U)
    idx1 = 1:n1
    idx2 = (n1 + 1):n2
    @tturbo for i in idx1
        P[i] += α * q[i]
        U[i] += α * z[i]
        r[i] -= α * Sq[i]
    end
    @tturbo for i in idx2
        U[i] += α * z[i]
    end
end
