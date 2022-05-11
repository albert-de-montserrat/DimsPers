#=
Usage: V_i = interp2d_cubic(GCOORD,EL2NOD,els,lc,V,OPTS)

Purpose: Interpolates variables in a triangular finite element mesh using
         quasi-cubic interpolation functions. Based on code developed by
         Chao Shi & Jason Phipps Morgan, 2010.

Input
  GCOORD : [matrix]    : coordinates of all nodes in mesh
  EL2NOD : [matrix]    : finite element connectivity matrix (nnodel x nel)
  els    : [rowvector] : element in which each point is located
  lc     : [matrix]    : local coordinates in each element
  V      : [matrix]    : variables to be interpolated (nnod x nvar, i.e. 
                         one variable field in each column)
  OPTS   : [structure] : options for this functions

Output
  V_i    : [matrix]    : interpolated values (npt x nvar, i.e. one 
                         interpolated field in each column)
=#

import LinearAlgebra: det

struct QCubic
    x_el::SubArray{Float64,2,Vector{Float64},Tuple{Matrix{Int64}},false}
    z_el::SubArray{Float64,2,Vector{Float64},Tuple{Matrix{Int64}},false}
end

@inline function local_coordinates(vc, pc::Point2D{T}) where {T}
    # Coordinates of triangle's vertices
    x1, x2, x3 = vc[1].x, vc[2].x, vc[3].x
    z1, z2, z3 = vc[1].z, vc[2].z, vc[3].z
    # Coordinates of query point
    x, z = pc.x, pc.z

    denom = -x1 * z3 + x1 * z2 - x2 * z1 + x2 * z3 + x3 * z1 - x3 * z2

    λ2 = -(-x1 * z + x1 * z3 - x3 * z1 + x * z1 + x3 * z - x * z3) / denom

    λ3 = (x1 * z2 - x1 * z - x2 * z1 + x2 * z + x * z1 - x * z2) / denom

    λ1 = 1 - (λ3 + λ2)

    return @SVector [λ1, λ2, λ3]
end

@inline function quasicubic_SF(λ)
    @SVector [
        λ[1],
        λ[2],
        λ[3],
        λ[1] * λ[2],
        √2 * λ[3] * λ[2],
        λ[3] * λ[1],
        λ[1] * λ[2] * (1 - λ[3] - 2 * λ[2]),
        √2 * λ[3] * λ[2] * (λ[3] - λ[2]),
        λ[3] * λ[1] * (1 - λ[2] - 2 * λ[3]),
    ]
end

@inline function det(v::Vector{Point2D{T}}) where {T}
    return v[2].x * v[3].z + v[3].x * v[1].z + v[1].x * v[2].z - v[2].x * v[1].z -
           v[3].x * v[2].z - v[1].x * v[3].z
end

@inline function element_center(v::Vector{Point2D{T}}) where {T}
    return ((v[1].x + v[2].x + v[3].x) / 3, (v[1].z + v[2].z + v[3].z) / 3)
end

@inline function local_weights(vc)
    center = element_center(vc)
    @SVector [√((vc[i].x - center[1])^2 + (vc[i].z - center[2])^2) for i in 1:3]
end

function local_derivatives(gr::Grid, vc, field, iel)
    # unpacks
    e2n = gr.e2n_p1
    θ, r = gr.x, gr.z
    # cache indices 
    idx = @views e2n[:, iel]
    x_el, z_el = @views (θ[idx]), @views (r[idx])

    vertices!(vc, x_el, z_el)

    ω = local_weights(vc) # weights 

    invdetM = 1 / det(vc)

    invMx = @SVector [
        (vc[2].z - vc[3].z) * invdetM,
        (vc[3].z - vc[1].z) * invdetM,
        (vc[1].z - vc[2].z) * invdetM,
    ]
    invMz = @SVector [
        (vc[3].x - vc[2].x) * invdetM,
        (vc[1].x - vc[3].x) * invdetM,
        (vc[2].x - vc[1].x) * invdetM,
    ]

    B = @views field[idx]
    dBdx = mydot(B, invMx)
    dBdz = mydot(B, invMz)

    return dBdx, dBdz, ω
end

function element_derivatives(gr, vc, fields)
    e2n_p1 = gr.e2n_p1
    nnod = maximum(e2n_p1)
    nels = size(e2n_p1, 2)
    dBdx_nod = zeros(nnod)
    dBdz_nod = zeros(nnod)
    ω_nod = zeros(nnod)
    dBdx_el = Vector{Float64}(undef, nels)
    dBdz_el = similar(dBdx_el)

    @inbounds for iel in 1:nels
        dBdx_ipart, dBdz_ipart, ω = local_derivatives(gr, vc, fields, iel)

        dBdx_el[iel] = dBdx_ipart
        dBdz_el[iel] = dBdz_ipart

        # reduction over nodes
        @inbounds for i in 1:3
            el_nod = e2n_p1[i, iel]
            dBdx_nod[el_nod] += dBdx_ipart * ω[i]
            dBdz_nod[el_nod] += dBdz_ipart * ω[i]
            ω_nod[el_nod] += ω[i]
        end
    end

    dBdx, dBdz = similar(dBdx_nod), similar(dBdx_nod)
    @turbo for i in eachindex(dBdx)
        dBdx[i] = dBdx_nod[i] / ω_nod[i]
        dBdz[i] = dBdz_nod[i] / ω_nod[i]
    end

    dBdx_el2nod = view(dBdx, e2n_p1) .- dBdx_el' # remove background slope x
    dBdz_el2nod = view(dBdz, e2n_p1) .- dBdz_el' # remove background slope z

    return dBdx_el2nod, dBdz_el2nod
end

function slope(gr, dBdx_el2nod, dBdz_el2nod, nels)
    slope_info = Array{Float64,2}(undef, 6, nels)
    sqrthalf = √0.5
    @inbounds for i in 1:nels
        idx = @views gr.e2n_p1[:, i]
        x_el, z_el = @views (gr.x[idx]), @views (gr.z[idx])
        c1x = -x_el[1] + x_el[2]
        c1z = -x_el[1] + x_el[3]
        c2x = -z_el[1] + z_el[2]
        c2z = -z_el[1] + z_el[3]

        slope_xi1 = c1x * dBdx_el2nod[1, i] + c2x .* dBdz_el2nod[1, i]
        slope_xi2 = c1x * dBdx_el2nod[2, i] + c2x .* dBdz_el2nod[2, i]
        slope_xi3 = c1x * dBdx_el2nod[3, i] + c2x .* dBdz_el2nod[3, i]
        slope_zi1 = c1z * dBdx_el2nod[1, i] + c2z .* dBdz_el2nod[1, i]
        slope_zi2 = c1z * dBdx_el2nod[2, i] + c2z .* dBdz_el2nod[2, i]
        slope_zi3 = c1z * dBdx_el2nod[3, i] + c2z .* dBdz_el2nod[3, i]

        dBdSC = sqrthalf * slope_xi3 - sqrthalf * slope_zi3 # dT/dS at point C (slope along side BC at C(0,1)):  dT/dS=dT/dxi*dxi/dS + dT/deta*deta/dS
        dBdSB = sqrthalf * slope_xi2 - sqrthalf * slope_zi2 # dT/dS at point B (slope along side BC at B(1,0))   the 'S+' is from C(0,1) to B(1,0)

        slope_info[1, i] = (slope_xi1 - slope_xi2) * 0.5 # this is to feed to N4
        slope_info[2, i] = (dBdSC - dBdSB) * 0.5         # N5
        slope_info[3, i] = (slope_zi1 - slope_zi3) * 0.5 # N6
        slope_info[4, i] = (slope_xi1 + slope_xi2) * 0.5 # N7
        slope_info[5, i] = (dBdSC + dBdSB) * 0.5         # N8
        slope_info[6, i] = (slope_zi1 + slope_zi3) * 0.5 # N9
    end
    return slope_info
end

function cubic_particle_interpolation!(
    fields_particles, particle_info, slope_info, gr, fields, vc
)
    for (i, info) in enumerate(particle_info)
        iel = info.t
        pc = info.CCart
        x_el, z_el = @views (gr.x[gr.e2n_p1[:, iel]]), @views (gr.z[gr.e2n_p1[:, iel]])
        els = view(gr.e2n_p1, :, iel)

        # compute shape functions 
        vertices!(vc, x_el, z_el)
        λ = local_coordinates(vc, pc)
        N = quasicubic_SF(λ)

        # # interpolate field to particle 
        Fel = view(fields, els)
        slope_info_part = view(slope_info, :, iel)

        fields_particles[i] =
            Fel[1] * N[1] +
            Fel[2] * N[2] +
            Fel[3] * N[3] +
            slope_info_part[1] * N[4] +
            slope_info_part[2] * N[5] +
            slope_info_part[3] * N[6] +
            slope_info_part[4] * N[7] +
            slope_info_part[5] * N[8] +
            slope_info_part[6] * N[9]
    end
end

function vertices!(v::Array{Point2D{T},1}, x, z) where {T}
    @inbounds for i in 1:3
        v[i].x = x[i]
        v[i].z = z[i]
    end
end

function quasicubic_interpolation(gr, particle_info, fields)
    np = length(particle_info)
    nels = size(gr.e2n_p1, 2)

    vc = [Point2D{Cartesian}(0.0, 0.0) for _ in 1:3] # element vertices

    dBdx_el2nod, dBdz_el2nod = element_derivatives(gr, vc, fields)

    # also need to map the slope information to the xi, yi (ξ, η) axes
    slope_info = slope(gr, dBdx_el2nod, dBdz_el2nod, nels)

    fields_particles = Vector{Float64}(undef, np)
    cubic_particle_interpolation!(
        fields_particles, particle_info, slope_info, gr, fields, vc
    )
    return fields_particles
end

function velocity_error(Upolar, particle_info, gr, output_path)
    Uth, Ur = analytic_flow(gr.θ, gr.r)
    # Allocate a couple of buffer arrays
    Uth_node = [u.x for u in Upolar]
    Ur_node = [u.z for u in Upolar]
    θp, rp = particle_coordinates(particle_info; coordinates_system="polar") # initial particle position

    Uth_an, Ur_an = analytic_flow(θp, rp)
    speed_an = @. √(Uth_an^2 + Ur_an^2)

    Uth_particle = quasicubic_interpolation(gr, particle_info, Uth_node)
    Ur_particle = quasicubic_interpolation(gr, particle_info, Ur_node)
    speed_particle = @. √(Uth_particle^2 + Ur_particle^2)

    error_th = norm(@. Uth_an - Uth_particle)
    error_r = norm(@. Ur_an - Ur_particle)
    error_speed = norm(@. speed_an - speed_particle)

    fname = joinpath(output_path, "velocity_error.txt")
    open(fname, "a") do io
        writedlm(io, [error_th error_r error_speed])
    end
end
