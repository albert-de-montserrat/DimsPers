struct ThermalParameters{T}
    K::T
    κ::T
    α::T
    Cp::T
    dQdT::T
    H::T

    function ThermalParameters(; ρ=3300, K=3e0, α=5e-5, Cp=1.2e3, dQdT=0.0, H=0.0)
        return new{eltype(α)}(K, K / ρ / Cp, α, Cp, dQdT, H)
    end
end

function thermal_parameters(; κ=1.0, α=1e-6, Cp=1.0, dQdT=0.0, H=0.0)
    return ThermalParameters(κ, α, Cp, dQdT, H)
end

function init_U_P(gr; ρ=3300, g=9.81)
    nU = maximum(gr.e2n)
    U = fill(0.0, nU .* 2)
    P = Vector{Float64}(undef, maximum(gr.e2nP))
    R = maximum(gr.r)
    meanR = R - mean(gr.r)
    @tturbo for i in eachindex(P)
        P[i] = ((R - gr.r[i]) - meanR) * ρ * g
        # P[i] = (R - gr.r[i]) * ρ * g
    end

    return U, P
end

function initP!(P, r; ρ=3300, g=9.81)
    @tturbo for i in eachindex(P)
        P[i] = r[i] * ρ * g
    end
end

function init_temperature(gr, IDs; type=:harmonic, units=:adimensional)
    (; θ, r) = gr
    maxr = maximum(r)

    Ttop, Tbot = units == :adimensional ? (0.0, 1.0) : (300.0, 4000.0)

    # adiabat
    potential_T = 1900
    Tp = units == :adimensional ? potential_T / 2500.0 : potential_T # non-dimensional potential temperature
    # T = @. (Tbot-Tp)*(r)/(gr.Ro-gr.Ri) + Tp # adiabat
    # T = @. (2500-Tp)*(maxr - r)/(gr.Ro-gr.Ri) + Tp # adiabat
    T = @. 0.3e-3 * (maxr - r) + Tp # adiabat

    # top boundary layer
    t = 1000e6 * 3600 * 365.25 * 24
    κ = 7.5e-7
    maxT = maximum(T)
    Ths = @. Ttop + (maxT - Ttop) * erf((maxr - r) * 0.5 / sqrt(κ * t)) # half-space cooling
    T = min.(T, Ths)

    # bottom boundary layer
    idx = r .< 6.351e6 - 2790e3
    a = -(minimum(@views T[idx]) - Tbot) / (minimum(r) - maximum(r[idx]))
    b = Tbot - a * minimum(r)
    bottom_boundary = @. a * (r[idx]) + b
    T[idx] .= bottom_boundary

    t = 50e6 * 3600 * 365.25 * 24
    idx = r .< (6.351e6 - 2690e3)
    T0 = minimum(T[idx])
    T[idx] = @. T0 + (Tbot - T0) * erfc((r[idx] - (6.351e6 - 2890e+3)) / 2 / sqrt(1e-6 * t))

    # Ths = @. Ttop + (2700-Ttop)*erf((maxr-r)*0.5/sqrt(κ*t)) # half-space cooling
    # T .= min.(Ths, T)

    if type == :harmonic
        # Harmonic hermal perturbation
        # s = @. 300(r)/(gr.Ro-gr.Ri)*Tbot + 300
        a = maximum(r) .- (r ./ maximum(r) .+ minimum(r))
        # s = @. (a)/(2.22-1.22)
        s = @. (Tbot - Ttop) * (gr.Ro - r) / (gr.Ro - gr.Ri) + Ttop
        y44 = @. 1.0 / 8.0 * √(35.0 / π) * cos(4.0 * θ)
        δT = @. 30 * y44 * sin(π * (1 - a))
        @. T = s + δT

    elseif type == :random
        # Linear temperature with random perturbation
        # T .+= rand(1.0:50.0, length(T))
        @tturbo for i in eachindex(T)
            T[i] += rand()*10 
        end

    elseif type == :realistic
        transition = gr.Ro - 0.2276
        T = @. (gr.Ro - r) / (gr.Ro - transition)
        idx = r .< transition
        T[idx] .= 1.0
    end

    fixT!(T, Ttop, Tbot, IDs)

    return T
end

function init_particle_temperature!(
    particle_fields, particle_info, Ro, Ri; type=:harmonic, units=:adimensional
)
    r = [particle_info[i].CPolar.z for i in eachindex(particle_info)]
    # s = @. (Ro-r)/(Ro-Ri)

    Ttop, Tbot = units == :adimensional ? (0.0, 1.0) : (300.0, 4000.0)

    # adiabat
    potential_T = 1900
    Tp = units == :adimensional ? potential_T / 2500.0 : potential_T # non-dimensional potential temperature
    # T = @. (Tbot-Tp)*(r)/(gr.Ro-gr.Ri) + Tp # adiabat
    T = @. (2500 - Tp) * (r) / (gr.Ro - gr.Ri) + Tp # adiabat

    if type == :harmonic
        θ = [particle_info[i].CPolar.x for i in eachindex(particle_info)]
        s = @. (4.2e3 - 3e2) * (r) / (gr.Ro - gr.Ri) + 3e2
        # Harmonic hermal perturbation
        y44 = @. 1.0 / 8.0 * √(35.0 / π) * cos(4.0 * θ)
        δT = @. 0.2 * y44 * sin(π * (1 - s))
        particle_fields.T = @. Ri * s / (Ro - s) + δT

    elseif type == :random
        # Linear temperature with random perturbation
        particle_fields.T = T .+ rand(length(T)) * 10
    end
end

function fixT!(T, Ttop, Tbot, IDs)
    @inbounds for i in axes(T, 1)
        if IDs[i] == :inner
            T[i] = Tbot
        elseif IDs[i] == :outter
            T[i] = Ttop
        end
    end
end
