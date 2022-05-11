mutable struct PINFO
    CPolar::Point2D{Polar}
    CCart::Point2D{Cartesian}
    UCart::Point2D{Cartesian}
    t::Int64
    t_parent::Int64
    color::Vector{Float32}
end

# mutable struct PINFO{A,B,C,D}
#     CPolar::A
#     CCart::B
#     UCart::B
#     t::C
#     t_parent::C
#     color::D
# end

struct PWEIGHTS # consider making it immutable
    barycentric::MArray{Tuple{3},Float64,1,3}
    ip_weights::MArray{Tuple{6},Float64,1,6}
    node_weights::MArray{Tuple{3},Float64,1,3}
end

# struct PWEIGHTS{T,M} # consider making it immutable
#     barycentric::T
#     ip_weights::M
#     node_weights::T
# end

mutable struct PVAR{M<:AbstractArray}
    T::M
    ΔT_subgrid::M
    Fxx::M
    Fzz::M
    Fxz::M
    Fzx::M
end

function init_pinfo(np)
    return [
        PINFO(
            Point2D{Polar}(0.0, 0.0),
            Point2D{Cartesian}(0.0, 0.0),
            Point2D{Cartesian}(0.0, 0.0),
            Int32(0),
            Int32(0),
            zeros(3),
        ) for _ in 1:np
    ]
end

function init_pweights(np)
    return [
        PWEIGHTS((@SVector zeros(3)), (@SVector zeros(6)), (@SVector zeros(3))) for
        _ in 1:np
    ]
end

function init_pvars(np)
    return PVAR(
        Vector{Float64}(undef, np),
        fill(1.0, np),
        fill(1.0, np),
        fill(1.0, np),
        fill(0.0, np),
        fill(0.0, np),
    )
end

@inline function getvertices!(v, x, i0)
    @inbounds for ii in 1:3
        id = @views i0[ii]
        v[ii] = x[id]
    end
end

function getvertices!(v::Array{Point2D{Polar},1}, θ, r, idx)
    @inbounds for ii in 1:3
        v[ii].x = θ[ii, idx]
        v[ii].z = r[ii, idx]
    end
end

function getvertices!(v::Array{Point2D{Cartesian},1}, x, z, idx)
    @inbounds for ii in 1:3
        v[ii].x = x[ii, idx]
        v[ii].z = z[ii, idx]
    end
end

function getvertices!(v::Point2D{Polar}, x, z, idx)
    @inbounds for ii in 1:3
        v[ii].x = x[ii, idx]
        v[ii].z = z[ii, idx]
    end
end

@inline getcartesian(A::PINFO) = A.CCart

@inline getpolar(A::PINFO) = A.CPolar

@inline function polar2cartesian(p::Point2D{Polar})
    return Point2D{Cartesian}(p.z * sin(p.x), p.z * cos(p.x))
end

@inline polar2x(p::Point2D{Polar}) = p.z * sin(p.x)
@inline polar2z(p::Point2D{Polar}) = p.z * cos(p.x)

@inline cartesian2θ(A::Point2D{Cartesian}) = atan(A.x, A.z)
@inline cartesian2r(A::Point2D{Cartesian}) = sqrt(A.x^2 + A.z^2)

@inline function cartesian2polar(p::Point2D{Cartesian})
    return Point2D{Polar}(atan(p.x, p.z), sqrt(p.x^2 + p.z^2))
end

@inline function cartesian2polar(x::AbstractArray, z::AbstractArray)
    return (atan.(x, z), @. sqrt(x^2 + z^2))
end

@inline polar2cartesian(θ::AbstractArray, r::AbstractArray) = (@. r * sin(θ), @. r * cos(θ))

function cartesian2polar!(particle_info::Array{PINFO,1})
    Threads.@threads for i in axes(particle_info, 1)
        particle_info[i].CPolar.x = atan(particle_info[i].CCart.x, particle_info[i].CCart.z)
        particle_info[i].CPolar.z = sqrt(
            particle_info[i].CCart.x^2 + particle_info[i].CCart.z^2
        )

        # Fix θ
        if particle_info[i].CPolar.x < 0
            particle_info[i].CPolar.x += 2π
        elseif particle_info[i].CPolar.x > 2π
            particle_info[i].CPolar.x -= 2π
        end

        # Fix radius
        if particle_info[i].CPolar.z < 6.351e6 - 2890e3
            particle_info[i].CPolar.z = 6.351e6 - 2890e3 + 2e3
            particle_info[i].CCart.x =
                particle_info[i].CPolar.z * sin(particle_info[i].CPolar.x)
            particle_info[i].CCart.z =
                particle_info[i].CPolar.z * cos(particle_info[i].CPolar.x)

        elseif particle_info[i].CPolar.z > 6.351e6
            particle_info[i].CPolar.z = 6.351e6 - 2e3
            particle_info[i].CCart.x =
                particle_info[i].CPolar.z * sin(particle_info[i].CPolar.x)
            particle_info[i].CCart.z =
                particle_info[i].CPolar.z * cos(particle_info[i].CPolar.x)
        end
    end
end

function polar2cartesian!(particle_info::Array{PINFO,1})
    Threads.@threads for i in axes(particle_info, 1)
        zp, xp = particle_info[i].CPolar.z, particle_info[i].CPolar.x
        particle_info[i].CCart.x = zp * sin(xp)
        particle_info[i].CCart.z = zp * cos(xp)
    end
end

@inline function distance(p1::Point2D{Polar}, p2::Array{Point2D{Polar},1})
    ld = length(p2)
    d = Vector{Float64}(undef, ld)
    @inbounds for i in 1:ld
        # d[i] = √(p1.z^2 + p2[i].z^2 - 2p1.z * p2[i].z * cos(p1.x - p2[i].x))
        d[i] = distance(polar2cartesian(p1), polar2cartesian(p2[i]))
    end
    return d
end

@inline function distance(p1::Point2D{Cartesian}, p2::Point2D{Cartesian})
    return √((p1.x - p2.x)^2 + (p1.z - p2.z)^2)
end

@inline function distance(p1::Point2D{Polar}, p2::Point2D{Polar})
    return distance(polar2cartesian(p1), polar2cartesian(p2))
end

@inline function distance(p1::Point2D{Polar}, p2::NamedTuple)
    return distance(polar2cartesian(p1), polar2cartesian(p2))
end

@inline function distance(p1::Point2D{Cartesian}, p2::NamedTuple)
    return √((p1.x - p2.x)^2 + (p1.z - p2.z)^2)
end

@inline distance(x1::T, z1::T, x2::T, z2::T) where {T<:Real} = √((x1 - x2)^2 + (z1 - z2)^2)

@inline function distance(p1::NTuple{2,T}, p2::NTuple{2,T}) where {T<:Real}
    return √((p1[1] - p2[1])^2 + (p1[2] - p2[2])^2)
end

@inline function getcentroid(p1::Point2D{T}, p2::Point2D{T}, p3::Point2D{T}) where {T}
    return Point2D{T}((p1.x + p2.x + p3.x) / 3, (p1.z + p2.z + p3.z) / 3)
end

@inbounds function barycentric_coordinates(bary_dummy, in, vc, pc)
    x1, x2, x3 = vc[1].x, vc[2].x, vc[3].x
    z1, z2, z3 = vc[1].z, vc[2].z, vc[3].z
    x, z = pc.x, pc.z
    inside = false

    bary_dummy[1] = @muladd ((z2 - z3) * (x - x3) + (x3 - x2) * (z - z3)) /
        ((z2 - z3) * (x1 - x3) + (x3 - x2) * (z1 - z3))

    if 0 <= bary_dummy[1] <= 1
        bary_dummy[2] = @muladd ((z3 - z1) * (x - x3) + (x1 - x3) * (z - z3)) /
            ((z2 - z3) * (x1 - x3) + (x3 - x2) * (z1 - z3))

        if 0 <= bary_dummy[2] <= 1
            bary_dummy[3] = 1 - bary_dummy[1] - bary_dummy[2]

            0 <= bary_dummy[3] <= 1 ? in = true : in = false
        end
    end

    return bary_dummy, inside
end

@inbounds function barycentric_coordinates2(intri, vc, pc)
    x1, x2, x3 = vc[1].x, vc[2].x, vc[3].x
    z1, z2, z3 = vc[1].z, vc[2].z, vc[3].z
    x, z = pc.x, pc.z
    intri = false

    bc1, bc2, bc3 = 0.0, 0.0, 0.0

    bc1 = @muladd ((z2 - z3) * (x - x3) + (x3 - x2) * (z - z3)) /
        ((z2 - z3) * (x1 - x3) + (x3 - x2) * (z1 - z3))

    if 0 <= bc1 <= 1
        bc2 = @muladd ((z3 - z1) * (x - x3) + (x1 - x3) * (z - z3)) /
            ((z2 - z3) * (x1 - x3) + (x3 - x2) * (z1 - z3))
        if 0 <= bc2 <= 1
            bc3 = 1 - bc1 - bc2
            intri = 0 <= bc3 <= 1 ? true : false
        end
    end

    bary_dummy = @MVector [bc1, bc2, bc3]

    return bary_dummy, intri
end

@inline function barycentric_coordinates3(vc, pc)
    # Coordinates of triangle's vertices
    x1, x2, x3 = vc[1].x, vc[2].x, vc[3].x
    z1, z2, z3 = vc[1].z, vc[2].z, vc[3].z
    # Coordinates of query point
    x, z = pc.x, pc.z

    bc1 = @muladd ((z2 - z3) * (x - x3) + (x3 - x2) * (z - z3)) /
        ((z2 - z3) * (x1 - x3) + (x3 - x2) * (z1 - z3))

    if 0.0 <= bc1 <= 1.0
        bc2 = @muladd ((z3 - z1) * (x - x3) + (x1 - x3) * (z - z3)) /
            ((z2 - z3) * (x1 - x3) + (x3 - x2) * (z1 - z3))
        if 0.0 <= bc2 <= 1.0
            bc3 = 1.0 - bc1 - bc2
            0.0 <= bc3 <= 1.0 ? intri = true : intri = false
        end
    end

    return intri
end

function check_corruption!(found, particle_fields)
    np = length(particle_fields.T)
    Threads.@threads for i in 1:np
        @inbounds if isinf(particle_fields.T[i]) ||
            isinf(particle_fields.Fxx[i]) ||
            isinf(particle_fields.Fxz[i]) ||
            isinf(particle_fields.Fzx[i]) ||
            isinf(particle_fields.Fzz[i]) ||
            isnan(particle_fields.T[i]) ||
            isnan(particle_fields.Fxx[i]) ||
            isnan(particle_fields.Fxz[i]) ||
            isnan(particle_fields.Fzx[i]) ||
            isnan(particle_fields.Fzz[i])
            found[i] = false
        end
    end
end

function purgeparticles(particle_info, particle_weights, particle_fields, found)
    ikeep = findall(found)
    particle_info = [@inbounds particle_info[i] for i in ikeep]
    particle_weights = [@inbounds particle_weights[i] for i in ikeep]
    particle_fields.T = [@inbounds particle_fields.T[i] for i in ikeep]
    particle_fields.Fxx = [@inbounds particle_fields.Fxx[i] for i in ikeep]
    particle_fields.Fzz = [@inbounds particle_fields.Fzz[i] for i in ikeep]
    particle_fields.Fxz = [@inbounds particle_fields.Fxz[i] for i in ikeep]
    particle_fields.Fzx = [@inbounds particle_fields.Fzx[i] for i in ikeep]

    # @inbounds for i in eachindex(found)
    #     if ikeep[i] == false
    #         popat!(particle_fields.T, i)
    #         popat!(particle_fields.Fxx, i)
    #         popat!(particle_fields.Fzz, i)
    #         popat!(particle_fields.Fxz, i)
    #         popat!(particle_fields.Fzx, i)
    #     end
    # end 

    return particle_info, particle_weights, particle_fields
end

function check_crossingπ!(vertices, particle)
    if (particle.x < π) &&
        (maximum(@SVector [vertices[1].x, vertices[3].x, vertices[3].x]) > 2π)
        for i in 1:3
            @inbounds if vertices[i].x > π
                vertices[i].x -= 2π
            end
        end
    end
end

function check_crossingπ(vertices, xp)
    xp[1] = vertices[1].x
    xp[2] = vertices[2].x
    xp[3] = vertices[3].x
    maxθ = maximum(xp)
    minθ = minimum(xp)

    check = false
    if (maxθ - minθ) > π
        for i in 1:3
            @inbounds if vertices[i].x > π
                vertices[i].x -= 2π
                check = true
            end
        end
    end
    return vertices, check
end

@inline function fillweights!(particle_weights, bary_dummy, particle, vertices, IC, ipart)
    @inbounds @simd for ii in 1:6
        if ii < 4
            # barycentric weights
            particle_weights[ipart].barycentric[ii] = bary_dummy[ii]
            # Nodal weights (for particle 2 node)
            particle_weights[ipart].node_weights[ii] = distance(particle, vertices[ii])
        end
        # Barycentric weights (for particle 2 ip)
        particle_weights[ipart].ip_weights[ii] = distance(particle, IC[ii])
    end
end

@inline function getreps(t0, nel, maxreps)
    reps = fill(Int32(0), nel)
    toremove = Int32[]
    @inbounds for (i, t) in enumerate(t0)
        reps[t] += Int32(1)
        if reps[t] > maxreps
            push!(toremove, i)
        end
    end
    return reps, toremove
end

@inline removeparticles(PC, ikeep) = [PC[i] for i in ikeep]

@inline function removeparticles_fields(PC, ikeep)
    nkeep = length(ikeep)
    Tp = Vector{Float64}(undef, nkeep)
    Fxxp = similar(Tp)
    Fzzp = similar(Tp)
    Fxzp = similar(Tp)
    Fzxp = similar(Tp)

    for (j, i) in enumerate(ikeep)
        @inbounds Tp[j] = PC.T[i]
        @inbounds Fxxp[j] = PC.Fxx[i]
        @inbounds Fzzp[j] = PC.Fzz[i]
        @inbounds Fxzp[j] = PC.Fxz[i]
        @inbounds Fzxp[j] = PC.Fzx[i]
    end
    PC.T = Tp
    PC.Fxx = Fxxp
    PC.Fzz = Fzzp
    PC.Fxz = Fxzp
    PC.Fzx = Fzxp
    return PC
end

function addreject(
    T,
    F,
    gr,
    θThermal,
    rThermal,
    IC,
    particle_info,
    particle_weights,
    particle_fields;
    min_num_particles=6,
)
    e2n, e2n_p1 = gr.e2n, gr.e2n_p1

    nel = size(e2n_p1, 2)
    vertices = [Point2D{Polar}(rand(), rand()) for _ in 1:3] # element vertices 
    t0 = [particle_info[i].t for i in axes(particle_info, 1)]

    max_ppel, min_ppel = 15, min_num_particles

    # --- Find number of particles per element    
    reps, irem = getreps(t0, nel, max_ppel)
    # np = length(particle_info)
    # ikeep = setdiff(1:np, irem)
    # if !isempty(irem)
    #     particle_info = removeparticles(particle_info, ikeep)
    #     particle_weights = removeparticles(particle_weights, ikeep)
    #     particle_fields = removeparticles_fields(particle_fields, ikeep)
    # end

    # -- Create particles to be added
    iadd = findall(x -> x < min_ppel, reps) # elements to be injected
    nxel = @. max(min_ppel - reps, 0) # number of particles to inject x element
    nadd = sum(nxel)  # total number of new particles 

    println(nadd, " particles added \n")

    # -- Allocate stuff
    vertices = [Point2D{Polar}(0.0, 0.0) for _ in 1:3] # element vertices 

    if !isempty(iadd)
        @inbounds for iel in iadd
            # -- Parent element
            iparent = div(iel, 4, RoundUp)
            # Vertices of previous parent element
            getvertices!(vertices, θThermal, rThermal, iel)

            # # -- Find neighbouring particles
            # nb = neighbours0[iel]
            # iₙ = ismember(t0, nb)
            # particlesₙ = [particle_info[i].CPolar for i in iₙ]
            # Tₙ = particle_fields.T[iₙ]
            # Fxxₙ = particle_fields.Fxx[iₙ]
            # Fzzₙ = particle_fields.Fzz[iₙ]
            # Fxzₙ = particle_fields.Fxz[iₙ]
            # Fzxₙ = particle_fields.Fzx[iₙ]

            # -- Add particle stuff
            nrand = nxel[iel]
            for _ in 1:nrand
                # -- Particle coordinates
                particle = randomintriangle(vertices)

                # -- Barycentric coordinates and check if point is inside triangle
                bary_coords, = barycentric_coordinates2(true, vertices, particle)

                # --- Weights (for particle → node)
                node_weights = distance(particle, vertices)
                nw = @MVector [node_weights[1], node_weights[2], node_weights[3]]

                # --- Weights (for particle → integration point)
                ip_weights = distance(particle, IC[iparent, :])
                iw = @MVector [
                    ip_weights[1],
                    ip_weights[2],
                    ip_weights[3],
                    ip_weights[4],
                    ip_weights[5],
                    ip_weights[6],
                ]

                # #  -- Interpolate fields to new particles
                # ω = distance(particle, particlesₙ).^-1
                # sumω = sum(ω)
                # # Deformation gradient
                # Fxxp = mydot(ω,Fxxₙ)/sumω
                # Fzzp = mydot(ω,Fzzₙ)/sumω
                # Fzxp = mydot(ω,Fxzₙ)/sumω
                # Fxzp = mydot(ω,Fzxₙ)/sumω
                # # Temperature
                # Tp = mydot(ω,Tₙ)/sumω

                # Deformation gradient
                idx = view(e2n, 1:6, iparent)
                wF = @. 1 / iw^2
                Fxxp = ip2particle([F[i][1, 1] for i in idx], wF)
                Fzzp = ip2particle([F[i][2, 2] for i in idx], wF)
                Fzxp = ip2particle([F[i][2, 1] for i in idx], wF)
                Fxzp = ip2particle([F[i][1, 2] for i in idx], wF)

                # Temperature
                idx = view(e2n_p1, :, iel)
                Tp = weightednode2particle(view(T, idx), bary_coords)

                # -- Add to Particle Structures
                push!(
                    particle_info,
                    PINFO(
                        particle,
                        polar2cartesian(particle),
                        Point2D{Cartesian}(0.0, 0.0),
                        iel,
                        iparent,
                        zeros(3),
                    ),
                )

                push!(particle_weights, PWEIGHTS(bary_coords, iw, nw))

                push!(particle_fields.T, Tp)
                push!(particle_fields.ΔT_subgrid, 0.0)
                push!(particle_fields.Fxx, Fxxp)
                push!(particle_fields.Fzz, Fzzp)
                push!(particle_fields.Fxz, Fxzp)
                push!(particle_fields.Fzx, Fzxp)
            end
        end
    end

    return particle_info, particle_weights, particle_fields
end

@inbounds function randomintriangle(vp)
    p, q = rand(), rand()
    if (p + q) > 1
        p = 1 - p
        q = 1 - q
    end
    x = @muladd vp[1].x + (vp[2].x - vp[1].x) * p + (vp[3].x - vp[1].x) * q
    z = @muladd vp[1].z + (vp[2].z - vp[1].z) * p + (vp[3].z - vp[1].z) * q

    return Point2D{Polar}(x, z)
end

function particles_generator(θ3, r3, IntC, e2n_p1; number_of_particles=12)
    color = zeros(3)
    vertices = [Point2D{Polar}(rand(), rand()) for _ in 1:3] # element vertices 
    particle_info = PINFO[]
    particle_weights = PWEIGHTS[]
    nrand = number_of_particles
    node_weights = zeros(3)
    ip_weights = similar(node_weights)
    xp = zeros(3)

    for iel in axes(e2n_p1, 2)
        getvertices!(vertices, θ3, r3, iel)
        # check if element is crossing π and correct if needed
        vertices, check = check_crossingπ(vertices, xp)

        for _ in 1:nrand

            # generate radom particle inside triangle
            particle_polar = randomintriangle(vertices)
            # check if particle was rotated
            if check == true
                if particle_polar.x > 2pi
                    particle_polar.x -= 2pi
                end
            end
            t_parent = ceil(Int, iel / 4)

            # add to Particle Structures
            push!(
                particle_info,
                PINFO(
                    particle_polar,
                    polar2cartesian(particle_polar),
                    Point2D{Cartesian}(0.0, 0.0),
                    iel,
                    ceil(Int, iel / 4),
                    color,
                ),
            )

            # Barycentric coordinates and check if point is inside triangle
            bary_coords, = barycentric_coordinates2(true, vertices, particle_polar)
            # Weights (for particle → node)
            node_weights .= distance(particle_polar, vertices)
            nw = @MVector [node_weights[1], node_weights[2], node_weights[3]]
            #  Weights (for particle → integration point)
            ip_weights = distance(particle_polar, IntC[t_parent, :])
            iw = @MVector [
                ip_weights[1],
                ip_weights[2],
                ip_weights[3],
                ip_weights[4],
                ip_weights[5],
                ip_weights[6],
            ]

            push!(particle_weights, PWEIGHTS(bary_coords, iw, nw))
        end
    end

    return particle_info, particle_weights, init_pvars(length(particle_info))
end

@inline function bilinear_weigth(vc, pc, N)
    if N == 1 # index shuffle switcher
        x1, x2, x3 = vc[3].x, vc[1].x, vc[2].x
        z1, z2, z3 = vc[3].z, vc[1].z, vc[2].z
    elseif N == 2
        x1, x2, x3 = vc[2].x, vc[3].x, vc[1].x
        z1, z2, z3 = vc[2].z, vc[3].z, vc[1].z
    end
    x, z = pc.x, pc.z
    return N =
        ((z2 - z3) * (x - x3) + (x3 - x2) * (z - z3)) /
        ((z2 - z3) * (x1 - x3) + (x3 - x2) * (z1 - z3))
end

function anycustom(a, b)
    out = false
    for i in axes(b, 1)
        if @inbounds b[i] == a
            out = true
            break
        end
    end
    return out
end

function findlostparticles(t_lost, particle_info, θThermal, rThermal)
    idx = unique(t_lost)
    idx = idx[2:end]
    vertices = [Point2D{Polar}(rand(), rand()) for _ in 1:3] # element vertices 

    for i in idx
        intri = false
        while intri == false
            # -- i-th particle
            particle = particle_info[i].CPolar
            # -- Check if it's in the same element as in previous tstep
            previous_el = particle_info[i].t
            # -- Vertices of previous parent element
            getvertices!(vertices, θThermal, rThermal, previous_el)
            # -- Calculate barycentric coordinates and check if point is inside triangle
            _, intri = barycentric_coordinates2(true, vertices, particle)
        end
    end
end

function check_parent(vertices, θThermal, rThermal, xp, particle, iel)
    # -- Vertices of previous parent element
    getvertices!(vertices, θThermal, rThermal, iel)
    # -- check if element is crossing π and correct if needed
    check_crossingπ!(vertices, particle)
    # -- Calculate barycentric coordinates and check if point is inside triangle
    bary_coords, intri = barycentric_coordinates2(true, vertices, particle)
    return bary_coords, intri
end

function ismember(a, b)
    v = fill(0, length(b))
    n = 0
    for i in axes(b, 1)
        for j in axes(a, 1)
            @inbounds if a[j] == b[i]
                n += 1
                v[n] = i
                break
            end
        end
    end
    return view(v, 1:n)
end

function intriangle(vc::NamedTuple, pc)
    # Coordinates of triangle's vertices
    x1, x2, x3 = vc.x[1], vc.x[2], vc.x[3]
    z1, z2, z3 = vc.z[1], vc.z[2], vc.z[3]
    # Coordinates of query point
    x, z = pc.x, pc.z

    intri = false
    bc1 = @muladd ((z2 - z3) * (x - x3) + (x3 - x2) * (z - z3)) /
        ((z2 - z3) * (x1 - x3) + (x3 - x2) * (z1 - z3))
    if 0 <= bc1 <= 1
        bc2 = @muladd ((z3 - z1) * (x - x3) + (x1 - x3) * (z - z3)) /
            ((z2 - z3) * (x1 - x3) + (x3 - x2) * (z1 - z3))
        if 0 <= bc2 <= 1
            bc3 = 1 - bc1 - bc2
            if 0 <= bc3 <= 1
                intri = true
            end
        end
    end
    return intri
end

function intriangle(vc::Vector{Point2D{T}}, pc) where {T}
    # Coordinates of triangle's vertices
    x1, x2, x3 = vc[1].x, vc[2].x, vc[3].x
    z1, z2, z3 = vc[1].z, vc[2].z, vc[3].z
    # Coordinates of query point
    x, z = pc.x, pc.z

    intri = false
    bc1 = @muladd ((z2 - z3) * (x - x3) + (x3 - x2) * (z - z3)) /
        ((z2 - z3) * (x1 - x3) + (x3 - x2) * (z1 - z3))
    if 0 <= bc1 <= 1
        bc2 = @muladd ((z3 - z1) * (x - x3) + (x1 - x3) * (z - z3)) /
            ((z2 - z3) * (x1 - x3) + (x3 - x2) * (z1 - z3))
        if 0 <= bc2 <= 1
            bc3 = 1 - bc1 - bc2
            if 0 <= bc3 <= 1
                intri = true
            end
        end
    end
    return intri
end

function barycentric(vc::Vector{Point2D{T}}, pc) where {T}
    # Coordinates of triangle's vertices
    x1, x2, x3 = vc[1].x, vc[2].x, vc[3].x
    z1, z2, z3 = vc[1].z, vc[2].z, vc[3].z

    bc1 = @muladd ((z2 - z3) * (pc.x - x3) + (x3 - x2) * (pc.z - z3)) /
        ((z2 - z3) * (x1 - x3) + (x3 - x2) * (z1 - z3))
    bc2 = @muladd ((z3 - z1) * (pc.x - x3) + (x1 - x3) * (pc.z - z3)) /
        ((z2 - z3) * (x1 - x3) + (x3 - x2) * (z1 - z3))
    bc3 = 1 - bc1 - bc2

    return bc1, bc2, bc3
end

function barycentric(vc::NamedTuple, pc)
    # Coordinates of triangle's vertices
    x1, x2, x3 = vc.x[1], vc.x[2], vc.x[3]
    z1, z2, z3 = vc.z[1], vc.z[2], vc.z[3]

    bc1 = @muladd ((z2 - z3) * (pc.x - x3) + (x3 - x2) * (pc.z - z3)) /
        ((z2 - z3) * (x1 - x3) + (x3 - x2) * (z1 - z3))
    bc2 = @muladd ((z3 - z1) * (pc.x - x3) + (x1 - x3) * (pc.z - z3)) /
        ((z2 - z3) * (x1 - x3) + (x3 - x2) * (z1 - z3))
    bc3 = 1 - bc1 - bc2

    return bc1, bc2, bc3
end

function tsearch_parallel(
    particle_info, particle_weights, θThermal, rThermal, coordinates, gr, IntC
)
    np = length(particle_info) # num of particles 
    found = fill(false, np) # true if particle found in same element as in previous t-step
    vertices = [[Point2D{Polar}(0.0, 0.0) for _ in 1:3] for _ in 1:Threads.nthreads()] # element vertices 
    macro_neighbour = gr.neighbours
    θ, r = coordinates.θ, coordinates.r

    # This greedy algorithm is not enterely balanced, it is slighlty better to break
    # it into the three following separate balanced threaded loops
    @batch for ipart in 1:np
        # check if particle is in the same triangle as in previous time step i.e. found = true
        isinparent!(particle_info, θThermal, rThermal, vertices, found, ipart)
    end
    # this is the guy responsible of breaking the balanced load
    @batch for ipart in 1:np
        # if found[ipart] == false
        @inbounds found[ipart] && continue
        # check neighbour that contains new position of the particle
        isinneighbours!(
            particle_info, θThermal, rThermal, θ, r, vertices, ipart, found, macro_neighbour
        )
        # end
    end

    @batch for ipart in 1:np
        # fill weights information
        fillparticle!(
            particle_weights, particle_info, vertices, IntC, θThermal, rThermal, ipart
        )
    end

    return particle_info, particle_weights, found
end

getvertices(θ, r, t) = (x=@views(θ[:, t]), z=@views(r[:, t]))

function getvertices_e2n(gr::Grid, t::NTuple)
    return (x=(gr.θ[t[1]], gr.θ[t[2]], gr.θ[t[3]]), z=(gr.r[t[1]], gr.r[t[2]], gr.r[t[3]]))
end

@inbounds function isinparent!(particle_info, θThermal, rThermal, vertices, found, ipart)
    nt = Threads.threadid()
    # -- Vertices of previous parent element
    getvertices!(vertices[nt], θThermal, rThermal, particle_info[ipart].t)
    # -- Calculate barycentric coordinates and check if point is inside triangle
    return found[ipart] = intriangle(vertices[nt], particle_info[ipart].CPolar)
end

@inbounds function isinneighbours!(
    particle_info, θThermal, rThermal, θ, r, vertices, ipart, found, macro_neighbour
)
    nt = Threads.threadid()

    # check if it's still within same parent element (not included in neighbour list)
    iparent = particle_info[ipart].t_parent
    getvertices!(vertices[nt], θ, r, iparent)
    if intriangle(vertices[nt], particle_info[ipart].CPolar)
        els_p1 = (4 * (iparent - 1) + 1):((4 * (iparent - 1) + 1) + 3) # indices of child triangles

        for p in els_p1# indices of child triangles
            # Vertices of previous parent element
            getvertices!(vertices[nt], θThermal, rThermal, p)
            # Calculate barycentric coordinates and check if point is inside triangle
            intri = intriangle(vertices[nt], particle_info[ipart].CPolar)
            if intri == true
                particle_info[ipart].t = p
                found[ipart] = true
                break
            end
        end

    else
        for mn in macro_neighbour[iparent]
            getvertices!(vertices[nt], θ, r, mn)
            intri = intriangle(vertices[nt], particle_info[ipart].CPolar)

            if intri == true
                els_p1 = (4 * (mn - 1) + 1):((4 * (mn - 1) + 1) + 3) # indices of child triangles

                for p in els_p1# indices of child triangles
                    # Vertices of previous parent element
                    getvertices!(vertices[nt], θThermal, rThermal, p)
                    # Calculate barycentric coordinates and check if point is inside triangle
                    intri = intriangle(vertices[nt], particle_info[ipart].CPolar)
                    if intri == true
                        particle_info[ipart].t = p
                        found[ipart] = true
                        break
                    end
                end
            end
        end
    end
end

@inbounds function fillparticle!(
    particle_weights, particle_info, vertices, IntC, θThermal, rThermal, ipart
)
    nt = Threads.threadid()
    particle_info[ipart].t_parent = ceil(Int, particle_info[ipart].t / 4)
    pc = particle_info[ipart].CPolar
    # -- Check if element is crossing π and correct if needed
    getvertices!(vertices[nt], θThermal, rThermal, particle_info[ipart].t)
    # -- barycentric weights
    particle_weights[ipart].barycentric .= barycentric(vertices[nt], pc)
    # -- Nodal weights (for particle 2 node)
    particle_weights[ipart].node_weights .= distance(pc, vertices[nt])
    # particle_weights[ipart].ip_weights .= distance(pc,  view(IntC,particle_info[ipart].t_parent,:))
    @inbounds for ii in 1:6
        # -- Barycentric weights (for particle 2 ip)
        particle_weights[ipart].ip_weights[ii] = distance(
            pc, IntC[particle_info[ipart].t_parent, ii]
        )
    end
end