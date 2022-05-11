# Out-of-place C = A*B
@inline function gemm(A::AbstractMatrix, B::AbstractMatrix)
    m, n, p = size(A, 1), size(B, 2), size(A, 2)

    C = Matrix{eltype(A)}(undef, m, n)

    @turbo for i in 1:m, k in 1:n
        aux = zero(eltype(A[1]))
        for j in 1:p
            aux += A[i, j] * B[j, k]
        end
        C[i, k] = aux
    end
    return C
end

# threaded version of vectorized gemm
@inline function gemmt(A::AbstractMatrix, B::AbstractMatrix)
    m, n, p = size(A, 1), size(B, 2), size(A, 2)

    C = Matrix{eltype(A)}(undef, m, n)

    @tturbo for i in 1:m, k in 1:n
        aux = zero(eltype(A[1]))
        for j in 1:p
            aux += A[i, j] * B[j, k]
        end
        C[i, k] = aux
    end
    return C
end

# In-place C = A*B
@inline function gemm!(
    C::AbstractArray{T}, A::AbstractArray{T}, B::AbstractArray{T}
) where {T}
    m, n, p = size(A, 1), size(B, 2), size(A, 2)
    @turbo for i in 1:m, k in 1:n
        aux = zero(eltype(A[1]))
        for j in 1:p
            aux += A[i, j] * B[j, k]
        end
        C[i, k] = aux
    end
end

@inline function gemmt!(
    C::AbstractArray{T}, A::AbstractArray{T}, B::AbstractArray{T}
) where {T}
    m, n, p = size(A, 1), size(B, 2), size(A, 2)
    @tturbo for i in 1:m, k in 1:n
        aux = zero(eltype(A[1]))
        for j in 1:p
            aux += A[i, j] * B[j, k]
        end
        C[i, k] = aux
    end
end

# x + p*y -> y (vec+scalar*vec)
function xpy!(y::Vector{T}, x::Vector{T}, p::T) where {T}
    @tturbo for i in eachindex(y)
        y[i] = x[i] + p * y[i]
    end
end

# x + p*y -> y (vec+scalar*vec)
# xpy(y::CuArray, x::CuArray, p::Number) = x .+ p .* y

# out-of-place Sparse CSC matrix times dense vector
function spmv(A::AbstractSparseMatrix, x::DenseVector)
    out = zeros(eltype(x), A.m)
    Aj = rowvals(A)
    nzA = nonzeros(A)
    @inbounds for col in 1:(A.n)
        xj = x[col]
        @fastmath for j in nzrange(A, col)
            out[Aj[j]] += nzA[j] * xj
        end
    end
    return out
end

function _spmvt!(out, A, Aj, xj, nzA, col)
    @fastmath for j in nzrange(A, col)
        out[Aj[j]] += nzA[j] * xj
    end
end

# in-of-place Sparse CSC matrix times dense vector
function spmv!(out::DenseVector, A::AbstractSparseMatrix, x::DenseVector)
    fill!(out, zero(eltype(out)))
    Aj = rowvals(A)
    nzA = nonzeros(A)
    @inbounds for col in 1:(A.n)
        xj = x[col]
        @fastmath for j in nzrange(A, col)
            out[Aj[j]] += nzA[j] * xj
        end
    end
    return out
end

mynorm(r::Vector) = sqrt(mydot(r, r))

# Dot product
function mydot(a::AbstractArray{T}, b::AbstractArray{T}) where {T}
    n = zero(T)
    @turbo for i in eachindex(a)
        n += a[i] * b[i]
    end
    return n
end

@generated function mydot2(a::SVector{N, T}, b::AbstractArray{T}) where {N, T}
    quote
        Base.@_inline_meta
        val = zero(T)
        Base.Cartesian.@nexprs $N i-> val += a[i] * b[i]
        return val
    end
end

# Threaded dot product
function mydott(a::AbstractArray{T}, b::AbstractArray{T}) where {T}
    n = zero(T)
    @tturbo for i in eachindex(a)
        n += a[i] * b[i]
    end
    return n
end
