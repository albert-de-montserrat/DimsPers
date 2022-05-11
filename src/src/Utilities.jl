@inline function accumarray(a, b)
    # TODO: consider blocking for larger sizes (if exceeding 1-2 ms)
    v = [fill(0.0, maximum(a)) for i in 1:Threads.nthreads()]
    Threads.@threads for i in axes(b, 1)
        @inbounds v[Threads.threadid()][a[i]] += b[i]
    end
    return Threads.nthreads() > 1 ? sum(v) : v[1]
end

@inline function _angular_distances!(VCOORD_th, ang_dist)
    @inbounds begin
        ang_dist[1, :] .= @. abs(@views (VCOORD_th[1, :] - VCOORD_th[2, :]))
        ang_dist[2, :] .= @. abs(@views (VCOORD_th[2, :] - VCOORD_th[3, :]))
        ang_dist[3, :] .= @. abs(@views (VCOORD_th[3, :] - VCOORD_th[1, :]))
    end

    idx = sum(ang_dist .> pi; dims=1) .> 1
    c = sum(idx)
    if c > 0
        # idx = findall(x->x==true,idx)
        @inbounds for i in 1:length(idx)
            if idx[i] == true
                j = view(VCOORD_th, :, i) .< pi
                @views VCOORD_th[j, i] .+= 2 * pi
            end
        end
    end
end

function elementcoordinate(GlobC, e2n)
    m, n = size(e2n)
    x = Matrix{Float64}(undef, m, n)
    z = Matrix{Float64}(undef, m, n)
    @inbounds for j in 1:n
        idx = view(e2n, :, j)
        for i in 1:m
            ii = idx[i]
            x[i, j] = GlobC[ii].x
            z[i, j] = GlobC[ii].z
        end
    end
    return x, z
end

function fixangles!(x)
    @inbounds for j in axes(x, 2)
        # v1,v2,v3 = x[1,j],x[2,j],x[3,j]
        v = ntuple(i -> x[i, j], 3)
        a1 = abs(v[1] - v[2])
        a2 = abs(v[2] - v[3])
        a3 = abs(v[3] - v[1])

        if (a1 + a2 + a3) > π
            for k in 1:3
                if v[k] < π
                    x[k, j] += 2π
                end
            end
        end
    end
end

function fixangles6!(x)
    @inbounds for j in axes(x, 2)
        v = ntuple(i -> x[i, j], 6)
        a1 = abs(v[1] - v[2])
        a2 = abs(v[2] - v[3])
        a3 = abs(v[3] - v[1])

        if (a1 + a2 + a3) > π
            for k in 1:6
                if v[k] < π
                    x[k, j] += 2π
                end
            end
        end
    end
end

getspeed(A::Point2D{Cartesian}) = √(A.x^2 + A.z^2)

getspeed(x::Real, z::Real) = √(x^2 + z^2)

# function calculate_Δt(Ucartesian, nθ, min_inradius)
#     # C = 0.1 # Courant number
#     # r = 1.22 # inner radius
#     # velocity = maximum(getspeed.(Ucartesian))
#     # dx_limit = 2π*r/nθ/4
#     # dt_adv = dx_limit / velocity * C
#     # min(dt_adv, 1.0)

#     min_inradius/maximum(getspeed.(Ucartesian))
# end

calculate_Δt(Ucartesian, nθ, min_inradius) = min_inradius / maximum(getspeed.(Ucartesian))

function getpoints(P)
    n = length(P)
    x = Vector{Float64}(undef, n)
    z = similar(x)

    Threads.@threads for i in eachindex(P)
        @inbounds x[i] = P[i].x
        @inbounds z[i] = P[i].z
    end
    return x, z
end

getpointsX(P) = [@inbounds P[i].x for i in eachindex(P)]
getpointsZ(P) = [@inbounds P[i].z for i in eachindex(P)]

function add_bubble_node(EL2NOD, thr)
    nel = size(EL2NOD, 2)
    nnod = maximum(EL2NOD)
    θ, r = deepcopy(thr[1, :]), deepcopy(thr[2, :])
    θel = @views θ[EL2NOD[1:3, :]]
    rel = @views r[EL2NOD[1:3, :]]
    fixangles!(θel)
    θ_new = mean(θel; dims=1)
    r_new = mean(rel; dims=1)
    θ = vcat(θ, vec(θ_new))
    r = vcat(r, vec(r_new))

    return vcat(EL2NOD, transpose(nnod .+ (1:nel))), hcat(thr, [θ_new; r_new]), θ, r
end

function getips(EL2NOD, θ, r)
    ndim = 2
    nvert = 3
    nnodel = 6
    nel = size(EL2NOD, 2)
    nip = 6
    ni, nn = Val(nip), Val(nnodel)
    N, dNds, _, w_ip = _get_SF(ni, nn)

    θip = Array{Float64,2}(undef, nel, nip)
    rip = similar(θip)

    @inbounds for iel in 1:nel
        VCOORD_th = SVector{6}(view(θ, :, iel))
        VCOORD_r = SVector{6}(view(r, :, iel))

        for ip in 1:nip
            θip[iel, ip] = (N[ip] * VCOORD_th)[1]
            rip[iel, ip] = (N[ip] * VCOORD_r)[1]
        end
    end

    return θip, rip
end
