function _MKLfactorize(KK, Rhs::Vector, ifree::Vector; verbose=false)
    A = KK[ifree, ifree]
    B = Rhs[ifree]
    # Initialize the PARDISO internal data structures.
    ps = MKLPardisoSolver()

    if verbose
        set_msglvl!(ps, Pardiso.MESSAGE_LEVEL_ON)
    end

    # First set the matrix type to handle general real symmetric matrices
    set_matrixtype!(ps, Pardiso.REAL_SYM_POSDEF)
    # Initialize the default settings with the current matrix type
    pardisoinit(ps)
    set_nprocs!(ps, Threads.nthreads())

    fix_iparm!(ps, :T)
    # Get the correct matrix to be sent into the pardiso function.
    # :N for normal matrix, :T for transpose, :C for conjugate
    A_pardiso = get_matrix(ps, A, :T)

    # Analyze the matrix and compute a symbolic factorization.
    set_phase!(ps, Pardiso.ANALYSIS)
    pardiso(ps, A_pardiso, B)

    # Compute the numeric factorization.
    set_phase!(ps, Pardiso.NUM_FACT)
    pardiso(ps, A_pardiso, B)

    return ps, A_pardiso
end

function _MKLsolve!(
    T::Vector, A_pardiso, ps::MKLPardisoSolver, Rhs::Vector{Float64}, ifree::Vector{Int64}
)
    B = Rhs[ifree]
    X = similar(B)

    # Compute the solutions X using the symbolic factorization.
    set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    pardiso(ps, X, A_pardiso, B)

    @turbo for i in 1:length(ifree)
        T[ifree[i]] = X[i]
    end
end

function _MKLrelease!(ps::MKLPardisoSolver)
    # Free the PARDISO data structures.
    set_phase!(ps, Pardiso.RELEASE_ALL)
    return pardiso(ps)
end

function _MKLpardiso!(
    T::Vector{Float64}, KK::SparseMatrixCSC, Rhs::Vector{Float64}, ifree::Vector{Int64}
)
    A = KK[ifree, ifree]
    B = Rhs[ifree]
    X = similar(B)

    # Initialize the PARDISO internal data structures.
    # ps = PardisoSolver()
    ps = MKLPardisoSolver()
    # if verbose
    #     set_msglvl!(ps, Pardiso.MESSAGE_LEVEL_ON)
    # end

    # First set the matrix type to handle general real symmetric matrices
    set_matrixtype!(ps, Pardiso.REAL_SYM_POSDEF)
    # Initialize the default settings with the current matrix type
    pardisoinit(ps)
    set_nprocs!(ps, Threads.nthreads())

    # Remember that we pass in a CSC matrix to Pardiso, so need
    # to set the transpose iparm option.
    fix_iparm!(ps, :T)
    # Get the correct matrix to be sent into the pardiso function.
    # :N for normal matrix, :T for transpose, :C for conjugate
    A_pardiso = get_matrix(ps, A, :T)

    return T[ifree] .= solve(ps, A_pardiso, B)

    # solve!(ps,X, A, B)

    # # Analyze the matrix and compute a symbolic factorization.
    # set_phase!(ps, Pardiso.ANALYSIS)
    # # set_perm!(ps, randperm(n))
    # pardiso(ps, A_pardiso, B)
    # # @printf("The factors have %d nonzero entries.\n", get_iparm(ps, 18))

    # # Compute the numeric factorization.
    # set_phase!(ps, Pardiso.NUM_FACT)
    # pardiso(ps, A_pardiso, B)
    # # @printf("The matrix has %d positive and %d negative eigenvalues.\n",
    # #         get_iparm(ps, 22), get_iparm(ps, 23))

    # # Compute the solutions X using the symbolic factorization.
    # set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    # pardiso(ps, X, A_pardiso, B)
    # # @printf("PARDISO performed %d iterative refinement steps.\n", get_iparm(ps, 7))

    # # solve!(ps,X, A_pardiso, B)
    # # Free the PARDISO data structures.
    # set_phase!(ps, Pardiso.RELEASE_ALL)
    # pardiso(ps)

    # @tturbo for i ∈ 1:length(ifree)
    #     T[ifree[i]] =  X[i]
    # end

end # END PARDISO SOLVER N.1

# PARDISO SOLVER N.2 ================================================================
function _MKLpardiso(A, B::Vector{Float64})

    # Allocate solution vector
    X = similar(B)

    # Initialize the PARDISO internal data structures.
    # ps = PardisoSolver()
    ps = MKLPardisoSolver()
    verbose = false
    if verbose
        set_msglvl!(ps, Pardiso.MESSAGE_LEVEL_ON)
    end

    # set_nprocs!(ps, 4) # Sets the number of threads to use
    # First set the matrix type to handle general real symmetric matrices
    set_matrixtype!(ps, Pardiso.REAL_SYM_POSDEF)
    # Initialize the default settings with the current matrix type
    pardisoinit(ps)
    # Remember that we pass in a CSC matrix to Pardiso, so need
    # to set the transpose iparm option.
    fix_iparm!(ps, :T)
    # Get the correct matrix to be sent into the pardiso function.
    # :N for normal matrix, :T for transpose, :C for conjugate
    A_pardiso = get_matrix(ps, A, :T)
    solve!(ps, X, A_pardiso, B)
    # Free the PARDISO data structures.
    set_phase!(ps, Pardiso.RELEASE_ALL)
    pardiso(ps)

    return X
end # END PARDISO SOLVER N.2

# PARDISO 6.2 N.1 ========================================================================
function _pardiso!(
    T::Vector{Float64}, KK::SparseMatrixCSC, Rhs::Vector{Float64}, ifree::Vector{Int64}
)
    A = KK[ifree, ifree]
    B = Rhs[ifree]
    X = similar(B)

    # Initialize the PARDISO internal data structures.
    ps = PardisoSolver()
    verbose = false
    if verbose
        set_msglvl!(ps, Pardiso.MESSAGE_LEVEL_ON)
    end

    # First set the matrix type to handle general real symmetric matrices
    set_matrixtype!(ps, Pardiso.REAL_SYM_POSDEF)

    # Initialize the default settings with the current matrix type
    pardisoinit(ps)

    # Get the correct matrix to be sent into the pardiso function.
    # :N for normal matrix, :T for transpose, :C for conjugate
    A_pardiso = get_matrix(ps, A, :T)

    # Analyze the matrix and compute a symbolic factorization.
    set_phase!(ps, Pardiso.ANALYSIS)
    pardiso(ps, A_pardiso, B)

    # Compute the numeric factorization.
    set_phase!(ps, Pardiso.NUM_FACT)
    pardiso(ps, A_pardiso, B)

    # Compute the solutions X using the symbolic factorization.
    set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    pardiso(ps, X, A_pardiso, B)

    # Free the PARDISO data structures.
    set_phase!(ps, Pardiso.RELEASE_ALL)
    pardiso(ps)

    # Refill global solution array
    for i in 1:length(ifree)
        T[ifree[i]] = X[i]
    end
end # END PARDISO 6.2 N.1

# PARDISO 6.2 N.2 ========================================================================
function _pardiso(A::SparseMatrixCSC, B::Vector{Float64})

    # Initialize the PARDISO internal data structures.
    ps = PardisoSolver()
    verbose = false
    if verbose
        set_msglvl!(ps, Pardiso.MESSAGE_LEVEL_ON)
    end

    # First set the matrix type to handle general real symmetric matrices
    set_matrixtype!(ps, Pardiso.REAL_SYM_POSDEF)

    # Initialize the default settings with the current matrix type
    pardisoinit(ps)

    # Get the correct matrix to be sent into the pardiso function.
    # :N for normal matrix, :T for transpose, :C for conjugate
    A_pardiso = get_matrix(ps, A, :T)

    # Analyze the matrix and compute a symbolic factorization.
    set_phase!(ps, Pardiso.ANALYSIS)
    pardiso(ps, A_pardiso, B)

    # Compute the numeric factorization.
    set_phase!(ps, Pardiso.NUM_FACT)
    pardiso(ps, A_pardiso, B)

    # Compute the solutions X using the symbolic factorization.
    set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    X = similar(B)
    pardiso(ps, X, A_pardiso, B)

    # Free the PARDISO data structures.
    set_phase!(ps, Pardiso.RELEASE_ALL)
    pardiso(ps)

    return X
end

function _Cholesky!(
    T::Vector{Float64}, KK::SparseMatrixCSC, Rhs::Vector{Float64}, ifree::Vector{Int64}
)
    Rhs_free = view(Rhs, ifree)
    A = KK[ifree, ifree]
    A = factorize(A)
    return T[ifree] = A.L' \ (A.L \ Rhs_free[A.p])
end # END CHOLESKY SOLVER 

"""
Solve Symmetric Positive Definite system of eqs using via CHOLESKY factorization and allocate solution
in the un-constrained elements of the solution array containing allocated boundary conditions. Return 
solution T and factorization F
"""
@inline function _CholeskyFactorizationSolve(
    T::Vector{Float64}, KK, Rhs::Vector{Float64}, ifree::Vector{Int64}
)
    K = KK[ifree, ifree]
    # F = cholesky((K')')
    # F = factorize((K')')
    F = lu(K)
    _CholeskyWithFactorization!(T, F, Rhs, ifree)
    return T, F
end
"""
Solve Symmetric Positive Definite system of eqs using via CHOLESKY factorization. Return 
solution T and factorization F
"""
@inline function _CholeskyFactorizationSolve(KK, Rhs::Vector{Float64})
    A = KK[ifree, ifree]
    # F = cholesky((A')')
    F = factorize((A')')
    T = F \ Rhs[ifree]
    return T, F
end

"""
Solve Symmetric Positive Definite system of eqs using A GIVEN factorization and allocate solution
in the un-constrained elements of the solution array containing allocated boundary conditions. In 
place substitution of input solution array T
"""
@inline function _CholeskyWithFactorization!(
    T::Vector{Float64}, A, Rhs, ifree::Vector{Int64}
)
    return T[ifree] .= A \ Rhs[ifree]
end
