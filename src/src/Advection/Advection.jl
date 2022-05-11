function step_RK(
    particle_info,
    gr,
    particle_weights,
    Ucartesian,
    xp0,
    zp0,
    Δt,
    θThermal,
    rThermal,
    coordinates,
    IntC,
    multiplier,
)
    e2n_p1 = gr.e2n_p1

    # (a) velocity @ particles
    particle_info = velocities2particle(particle_info, e2n_p1, particle_weights, Ucartesian)

    # velocities2particle_cubic!(gr, particle_info, Ucartesian)

    # (b) advect particles  
    particle_info = particlesadvection(xp0, zp0, particle_info, Δt; multiplier=multiplier)

    particle_info, particle_weights, found = tsearch_parallel(
        particle_info, particle_weights, θThermal, rThermal, coordinates, gr, IntC
    )

    # println(length(particle_info) - sum(found), " lost particles")

    # (c) store particles velocity
    Uxp, Uzp = particle_velocity(particle_info)

    return Uxp, Uzp, particle_info, particle_weights, found
end

# function advection_RK2!(Particles, xp0, zp0, Uxp1, Uzp1, Uxp2, Uzp2, Δt)
#     cte = Δt/2
#     Threads.@threads for i in eachindex(Particles)
#         @inbounds Particles[i].CCart.x = @muladd xp0[i] + (Uxp1[i] + Uxp2[i])*cte
#         @inbounds Particles[i].CCart.z = @muladd zp0[i] + (Uzp1[i] + Uzp2[i])*cte

#         @inbounds Particles[i].CPolar.x = atan(Particles[i].CCart.x, Particles[i].CCart.z)
#         @inbounds Particles[i].CPolar.z = sqrt(Particles[i].CCart.x^2 + Particles[i].CCart.z^2) 

#         # Fix θ
#         @inbounds if Particles[i].CPolar.x < 0
#             Particles[i].CPolar.x += 2π
#         elseif Particles[i].CPolar.x > 2π
#             Particles[i].CPolar.x -= 2π
#         end

#         # Fix radius
#        # Fix radius
#        @inbounds if Particles[i].CPolar.z > 6.351e6
#             Particles[i].CPolar.z = 6.351e6 - 2e3
#             Particles[i].CCart.x = polar2x(Particles[i].CPolar)
#             Particles[i].CCart.z = polar2z(Particles[i].CPolar)

#         elseif Particles[i].CPolar.z < 3.461e6
#             Particles[i].CPolar.z = 3.461e6 + 2e3
#             Particles[i].CCart.x = polar2x(Particles[i].CPolar)
#             Particles[i].CCart.z = polar2z(Particles[i].CPolar)
#         end

#     end
# end

function final_advection_RK!(Particles, xp0, zp0, Uxp1, Uzp1, Uxp2, Uzp2, Δt)
    cte = Δt / 2
    Threads.@threads for i in eachindex(Particles)
        @inbounds Particles[i].CCart.x = @muladd xp0[i] + (Uxp1[i] + Uxp2[i]) * cte
        @inbounds Particles[i].CCart.z = @muladd zp0[i] + (Uzp1[i] + Uzp2[i]) * cte

        @inbounds Particles[i].CPolar.x = atan(Particles[i].CCart.x, Particles[i].CCart.z)
        @inbounds Particles[i].CPolar.z = sqrt(
            Particles[i].CCart.x^2 + Particles[i].CCart.z^2
        )

        # Fix θ
        @inbounds if Particles[i].CPolar.x < 0
            Particles[i].CPolar.x += 2π
        elseif Particles[i].CPolar.x > 2π
            Particles[i].CPolar.x -= 2π
        end

        # Fix radius
        @inbounds if Particles[i].CPolar.z > 6.351e6
            Particles[i].CPolar.z = 6.351e6 - 2e3
            Particles[i].CCart.x = polar2x(Particles[i].CPolar)
            Particles[i].CCart.z = polar2z(Particles[i].CPolar)

        elseif Particles[i].CPolar.z < 3.461e6
            Particles[i].CPolar.z = 3.461e6 + 2e3
            Particles[i].CCart.x = polar2x(Particles[i].CPolar)
            Particles[i].CCart.z = polar2z(Particles[i].CPolar)
        end
    end
end

function advection_RK2(
    particle_info::Vector{PINFO},
    gr,
    particle_weights,
    Ucartesian,
    Δt,
    θThermal,
    rThermal,
    coordinates,
    IntC,
    to,
)

    # Allocate a couple of buffer arrays
    xp0, zp0 = particle_coordinates(particle_info) # initial particle position

    # STEP 1 
    Uxp1, Uzp1, particle_info, particle_weights, found = step_RK(
        particle_info,
        gr,
        particle_weights,
        Ucartesian,
        xp0,
        zp0,
        Δt,
        θThermal,
        rThermal,
        coordinates,
        IntC,
        1.0,
    )

    # STEP 2 
    # _, _, _, _, particle_info, particle_weights, found = step_RK(
    Uxp2, Uzp2, particle_info, particle_weights, found = step_RK(
        particle_info,
        gr,
        particle_weights,
        Ucartesian,
        xp0,
        zp0,
        Δt,
        θThermal,
        rThermal,
        coordinates,
        IntC,
        1.0,
    )

    # (c) advect particles
    final_advection_RK!(particle_info, xp0, zp0, Uxp1, Uzp1, Uxp2, Uzp2, Δt)

    return particle_info, particle_weights, to
end

function advection_RK4(
    particle_info::Vector{PINFO},
    gr,
    particle_weights,
    Ucartesian,
    Δt,
    θThermal,
    rThermal,
    IntC,
    to,
)

    # Allocate a couple of buffer arrays
    xp0, zp0 = particle_coordinates(particle_info) # initial particle position

    @timeit to "Advection" begin
        # STEP 1 
        xp1, zp1, Uxp1, Uzp1, particle_info, particle_weights = step_RK(
            particle_info,
            gr,
            particle_weights,
            Ucartesian,
            xp0,
            zp0,
            Δt,
            θThermal,
            rThermal,
            IntC,
            0.5,
        )

        # STEP 2 
        xp2, zp2, Uxp2, Uzp2, particle_info, particle_weights = step_RK(
            particle_info,
            gr,
            particle_weights,
            Ucartesian,
            xp0,
            zp0,
            Δt,
            θThermal,
            rThermal,
            IntC,
            0.5,
        )

        # STEP 3 
        xp3, zp3, Uxp3, Uzp3, particle_info, particle_weights = step_RK(
            particle_info,
            gr,
            particle_weights,
            Ucartesian,
            xp0,
            zp0,
            Δt,
            θThermal,
            rThermal,
            IntC,
            1.0,
        )

        # STEP 4
        _, _, Uxp4, Uzp4, particle_info, particle_weights = step_RK(
            particle_info,
            gr,
            particle_weights,
            Ucartesian,
            xp0,
            zp0,
            Δt,
            θThermal,
            rThermal,
            IntC,
            1.0,
        )

        # (c) advect particles
        advection_RK_step4!(
            particle_info, xp0, zp0, Uxp1, Uzp1, Uxp2, Uzp2, Uxp3, Uzp3, Uxp4, Uzp4, Δt
        )
    end

    return particle_info, to
end

function advection_RK_step4!(
    Particles, xp0, zp0, Uxp1, Uzp1, Uxp2, Uzp2, Uxp3, Uzp3, Uxp4, Uzp4, Δt
)
    cte = Δt / 6
    Threads.@threads for i in eachindex(Particles)
        @inbounds Particles[i].CCart.x =
            xp0[i] + (Uxp1[i] + 2 * (Uxp2[i] + Uxp3[i]) + Uxp4[i]) * cte
        @inbounds Particles[i].CCart.z =
            zp0[i] + (Uzp1[i] + 2 * (Uzp2[i] + Uzp3[i]) + Uzp4[i]) * cte

        @inbounds Particles[i].CPolar.x = atan(Particles[i].CCart.x, Particles[i].CCart.z)
        @inbounds Particles[i].CPolar.z = sqrt(
            Particles[i].CCart.x^2 + Particles[i].CCart.z^2
        )

        # Fix θ
        @inbounds if Particles[i].CPolar.x < 0
            Particles[i].CPolar.x += 2π

        elseif Particles[i].CPolar.x > 2π
            Particles[i].CPolar.x -= 2π
        end

        # Fix radius
        @inbounds if Particles[i].CPolar.z > 6.351e6
            Particles[i].CPolar.z = 6.351e6 - 2e3
            Particles[i].CCart.x = polar2x(Particles[i].CPolar)
            Particles[i].CCart.z = polar2z(Particles[i].CPolar)

        elseif Particles[i].CPolar.z < 3.461e6
            Particles[i].CPolar.z = 3.461e6 + 2e3
            Particles[i].CCart.x = polar2x(Particles[i].CPolar)
            Particles[i].CCart.z = polar2z(Particles[i].CPolar)
        end
    end
end

@inline function velocities2particle(particle_info, e2n_p1, particle_weights, Ucartesian)
    Threads.@threads for i in axes(particle_info, 1)
        # -- Reshape velocity arrays 
        # Ux_p1, Uz_p1 = getvelocity(Ucartesian, view(e2n_p1,:,particle_info[i].t) )
        # -- Velocity interpolation (directly using linear SF)
        @inbounds particle_info[i].UCart.x = node2particle(
            getvelocityX(Ucartesian, view(e2n_p1, :, particle_info[i].t)),
            particle_weights[i].barycentric,
        )
        @inbounds particle_info[i].UCart.z = node2particle(
            getvelocityZ(Ucartesian, view(e2n_p1, :, particle_info[i].t)),
            particle_weights[i].barycentric,
        )
    end
    return particle_info
end

@inline function velocities2particle!(
    particle_info, e2n_p1, particle_weights, Ucartesian, i
)
    # -- Reshape velocity arrays 
    Ux_p1, Uz_p1 = getvelocity(Ucartesian, view(e2n_p1, :, particle_info[i].t))
    # -- Velocity interpolation (directly using linear SF)
    particle_info[i].UCart.x = node2particle(Ux_p1, particle_weights[i].barycentric)
    return particle_info[i].UCart.z = node2particle(Uz_p1, particle_weights[i].barycentric)
end

function particlesadvection(xp0, zp0, Particles, Δt; multiplier=1)
    # np = length(Particles)
    # xp = Vector{Float64}(undef, np)
    # zp = similar(xp)
    Δtmultiplier = Δt * multiplier
    @fastmath Threads.@threads for i in eachindex(Particles)
        @inbounds Particles[i].CCart.x = @muladd xp0[i] +
            Particles[i].UCart.x * Δtmultiplier
        @inbounds Particles[i].CCart.z = @muladd zp0[i] +
            Particles[i].UCart.z * Δtmultiplier
        @inbounds Particles[i].CPolar.x = atan(Particles[i].CCart.x, Particles[i].CCart.z)
        @inbounds Particles[i].CPolar.z = @muladd √(
            Particles[i].CCart.x * Particles[i].CCart.x +
            Particles[i].CCart.z * Particles[i].CCart.z,
        )

        # Fix θ
        @inbounds if Particles[i].CPolar.x < 0
            Particles[i].CPolar.x += 2π
        elseif Particles[i].CPolar.x > 2π
            Particles[i].CPolar.x -= 2π
        end

        # Fix radius
        @inbounds if Particles[i].CPolar.z > 6.351e6
            Particles[i].CPolar.z = 6.351e6 - 2e3
            Particles[i].CCart.x = polar2x(Particles[i].CPolar)
            Particles[i].CCart.z = polar2z(Particles[i].CPolar)

        elseif Particles[i].CPolar.z < 3.461e6
            Particles[i].CPolar.z = 3.461e6 + 2e3
            Particles[i].CCart.x = polar2x(Particles[i].CPolar)
            Particles[i].CCart.z = polar2z(Particles[i].CPolar)
        end
    end
    return Particles
end

function particlesadvection!(xp, zp, xp0, zp0, particle_info, Δt; multiplier=1)
    Threads.@threads for i in eachindex(particle_info)
        @inbounds particle_info[i].CCart.x =
            xp[i] = xp0[i] + particle_info[i].UCart.x * Δt * multiplier
        @inbounds particle_info[i].CCart.z =
            zp[i] = zp0[i] + particle_info[i].UCart.z * Δt * multiplier
        @inbounds particle_info[i].CPolar.x = atan(
            particle_info[i].CCart.x, particle_info[i].CCart.z
        )
        @inbounds particle_info[i].CPolar.z = sqrt(
            particle_info[i].CCart.x^2 + particle_info[i].CCart.z^2
        )

        # Fix θ
        @inbounds if particle_info[i].CPolar.x < 0
            particle_info[i].CPolar.x += 2π

        elseif particle_info[i].CPolar.x > 2π
            particle_info[i].CPolar.x -= 2π
        end

        # Fix radius
        @inbounds if Particles[i].CPolar.z > 6.351e6
            Particles[i].CPolar.z = 6.351e6 - 2e3
            particle_info[i].CCart.x = xp[i] = polar2x(particle_info[i].CPolar)
            particle_info[i].CCart.z = zp[i] = polar2z(particle_info[i].CPolar)

        elseif Particles[i].CPolar.z < 3.461e6
            Particles[i].CPolar.z = 3.461e6 + 2e3
            particle_info[i].CCart.x = xp[i] = polar2x(particle_info[i].CPolar)
            particle_info[i].CCart.z = zp[i] = polar2z(particle_info[i].CPolar)
        end
    end
end

function advection!(particle_info, Δt)
    particlesadvection!(particle_info, Δt)  # advect cartesian coordinates
    cartesian2polar!(particle_info)
    return particle_info
end

function particle_velocity(P::Vector{PINFO})
    np = length(P)
    uxp = Vector{Float64}(undef, np)
    uzp = similar(uxp)
    Threads.@threads for i in 1:np
        @inbounds uxp[i] = P[i].UCart.x
        @inbounds uzp[i] = P[i].UCart.z
    end

    return uxp, uzp
end

function particle_velocity(P::PINFO)
    uxp = P.UCart.x
    uzp = P.UCart.z
    return uxp, uzp
end

function particle_velocity!(uxp, uzp, P::Vector{PINFO})
    Threads.@threads for i in eachindex(P)
        @inbounds uxp[i] = P[i].UCart.x
        @inbounds uzp[i] = P[i].UCart.z
    end
end
