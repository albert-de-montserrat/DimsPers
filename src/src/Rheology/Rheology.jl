export Density, Arrhenius

abstract type AbstractViscosity end
abstract type Isotropic end
abstract type Anisotropic end

struct Isoviscous{T} <: AbstractViscosity
    val::Float64
end

struct TemperatureDependant{T} <: AbstractViscosity
    node::Vector{Float64}
end

struct IsoviscousPlastic{T} <: AbstractViscosity
    node::Float64
    ip::Matrix{Float64}
    ϕ::Float64
    C::Float64

    function IsoviscousPlastic(T, val, ϕ, C, nel, nip)
        return new{T}(val, zeros(nel, nip), ϕ, C)
    end
end

struct TemperatureDependantPlastic{T} <: AbstractViscosity
    node::Vector{Float64}
    ip::Matrix{Float64}
    ϕ::Float64
    C::Float64

    function TemperatureDependantPlastic(T, val, ϕ, C, nel, nip)
        return new{T}(val, zeros(nel, nip), C, ϕ)
    end
end

struct Nakagawa{T} <: AbstractViscosity
    node::Vector{Float64}
    ip::Matrix{Float64}
    η0::Float64
    ϕ::Float64
    C::Float64
    Af::Function

    function Nakagawa(T, η0, val, ϕ, C, nel, nip, f::Function)
        return new{T}(val, zeros(nel, nip), η0, C, ϕ, f)
    end
end

struct Mallard2016{T} <: AbstractViscosity
    node::Vector{Float64}
    ip::Matrix{Float64}
    ϕ::Float64

    function TemperatureDependantPlastic(T, val, ϕ, nel, nip)
        return new{T}(val, zeros(nel, nip), ϕ)
    end
end

struct Arrhenius{T} <: AbstractViscosity
    node::Vector{Float64} # viscosity at nodes
    ip::Matrix{Float64} # viscosity at integration points
    η0::Float64 # reference viscosity
    R::Float64  # Gas constant (8.314 Jmol^{-1}K^{-1})
    Ea::Float64 # Activation energy
    Va::Float64 # Activation volume
    T0::Float64 # Reference temperature
    C::Float64 # Cohesion
    ϕ::Float64 # angle of friction

    function Arrhenius(
        gr::Grid; C=30e6, ϕ=1, η0=5e20, Ea=100e3, Va=1e-6, T0=1600, R=8.314, type=Isotropic
    )
        C, ϕ, η0, R, Ea, Va, T0 = Float64.((C, ϕ, η0, R, Ea, Va, T0))
        return new{type}(
            similar(gr.x), Matrix{Float64}(undef, gr.nel, 7), η0, R, Ea, Va, T0, C, ϕ
        )
    end
end

for visc in (:Isoviscous, :Nakagawa, :TemperatureDependantPlastic, :Arrhenius)
    @eval begin
        Base.Val(x::$(visc){Isotropic}) = Val{Isotropic}()
        Base.Val(x::$(visc){Anisotropic}) = Val{Anisotropic}()
    end
end

function getviscosity(T, r, type, nel; η=1.4e21, ϕ=0.06, C=2e6, nip=7)
    H(x) = x < 0 ? 0 : 1

    if type == :IsoviscousIsotropic
        return Isoviscous{Isotropic}(float(η))

    elseif type == :IsoviscousAnisotropic
        return Isoviscous{Anisotropic}(float(η))

    elseif type == :TemperatureDependantIsotropic
        return TemperatureDependant{Isotropic}(@. exp(13.8156 * (0.5 - T)))

    elseif type == :Nakagawa
        Af(r) = r < 660e3 ? 1 : 30
        P = @. r * 3300 * 9.81
        E = 290e3
        V = 2.4e-6
        A = η / exp(E / 8.314 / 1600)
        visc = @. A * exp((E + P * V) / 8.3143 / T) * Af(r)
        @. visc = min(max(visc, 1e18), 1e25)

        return Nakagawa(Isotropic, η, visc, ϕ, C, nel, nip, Af)

    elseif type == :TemperatureDependantIsotropicPlastic
        # return TemperatureDependantPlastic(Isotropic, @.(exp(13.8156*(0.5-T))), ϕ, nel, nip)
        # return TemperatureDependantPlastic(Isotropic, @.(exp(23.03/(1+T) -23.03/2)), ϕ, nel, nip)
        return TemperatureDependantPlastic(
            Isotropic,
            @.((1 + (10 - 1) * H(1.22 + 0.7716 - r)) * exp(23.03 / (1 + T) - 23.03 / 2)),
            ϕ,
            C,
            nel,
            nip,
        )
        # return TemperatureDependantPlastic(Isotropic, @.((1+(10-1)*H(1.22+0.7716-r))*exp(13.8156*(0.5-T))), ϕ, nel, nip)

    elseif type == :VanHeckIsotropic
        return TemperatureDependant{Isotropic}(
            @.((1 + (10 - 1) * H(1.22 + 0.7716 - r)) * exp(23.03 / (1 + T) - 23.03 / 2))
        )

    elseif type == :TemperatureDependantAnisotropic
        return TemperatureDependant{Anisotropic}(@. exp(13.8156 * (0.5 - T)))

    elseif type == :TemperatureDependantAnisotropicPlastic
        # return TemperatureDependant(Anisotropic, @.(exp(13.8156*(0.5-T))), ϕ, nel, nip)
        return TemperatureDependant(
            Anisotropic,
            @.((1 + (10 - 1) * H(1.22 + 0.7716 - r)) * exp(23.03 / (1 + T) - 23.03 / 2)),
            ϕ,
            C,
            nel,
            nip,
        )

    elseif type == :VanHeckAnisotropic
        return TemperatureDependant{Anisotropic}(@. exp(23.03 / (1 + T) - 23.03 / 2))
    end
end

getviscosity!(η::Isoviscous, T) = nothing

function getviscosity!(η::Arrhenius{M}, T::AbstractArray, P::AbstractArray) where {M}
    # η(T, P) = η0⋅exp((Ea + P⋅va)/R⋅T - Ea/R⋅T0)
    # v = DiffusionCreep()
    @tturbo for i in eachindex(η.node)
        @inbounds η.node[i] =
            η.η0 * exp((η.Ea + P[i] * η.Va) / (η.R * T[i]) - η.Ea / (η.R * η.T0))
            # computeViscosity_EpsII(1, v, (P=P[i], T=T[i], f=1e0, d=2e-5) ; cutoff=(1e16, 1e25)) 
    end
end

function getviscosity(η::Arrhenius{M}, Ti::Real, Pi::Real) where {M}
    return η.η0 * exp((η.Ea + Pi * η.Va) / η.R / Ti - η.Ea / η.R / η.T0)
end

function getviscosity!(η::TemperatureDependant, T, r)
    H(x) = x < 0 ? 0 : 1

    Threads.@threads for i in eachindex(η.node)
        # update viscosity
        # @inbounds η.node[i] = exp(13.8156*(0.5-T[i]))
        @inbounds η.node[i] = max(
            min(
                (1 + (10 - 1) * H(1.22 + 0.7716 - r[i])) *
                exp(23.03 / (1 + T[i]) - 23.03 / 2),
                1e25,
            ),
            1e18,
        )
    end
end

function getviscosity!(η::Nakagawa, T, r)
    Af = η.Af
    E = 290e3
    V = 2.4e-6
    A = η.η0 / exp(E / 8.314 / 1600)

    # update viscosity [(-r[i]*3300*9.81) := lithostatic pressure]
    @inbounds for i in eachindex(η.node)
        η.node[i] = max(
            min(A * exp((E + (r[i] * 3300 * 9.81) * V) / 8.3143 / T[i]) * Af(r[i]), 1e25),
            1e18,
        )
    end
end

function getviscosity!(η::TemperatureDependantPlastic, T, r)
    H(x) = x < 0 ? 0 : 1

    @inbounds for i in eachindex(η.node)
        depth = r[i] - 1.22
        # viscosity correction from Richards et al. [2001] and Tackley [2000b] 
        m = T[i] < 0.6 + 2 * (1 - depth) ? 1 : 0.1
        # update viscosity
        # η.node[i] = exp(23.03/(1+T[i]) -23.03/2) * m
        η.node[i] =
            (1 + (10 - 1) * H(1.22 + 0.7716 - r[i])) * exp(23.03 / (1 + T[i]) - 23.03 / 2)
        # η.node[i] = (1+(10-1)*H(1.22+0.7716-r[i]))*exp(13.8156*(0.5-T[i]))

    end
end

function getviscosity!(η::Mallard2016, T, r)
    # Parameters from Mallard et al 2016 - Nature
    a, B, d0, dstep = 1e6, 30, 0.276, 0.02

    # η(z,T) from Mallard et al 2016 - Nature
    @inbounds for i in eachindex(η)
        depth = 2.22 - r[i]
        # viscosity correction 
        m = T[i] < 0.6 + 7.5 * depth ? 1 : 0.1
        # depth dependent component
        ηz = a * exp(log(B) * (1 - 0.5 * (1 - tanh((d0 - depth) / dstep))))
        # update viscosity
        η[i] = ηz * exp(0.064 - 30 / (T[i] + 1)) * m
    end
end

struct Density{T}
    ρ::T # real density
    ρ0::T # reference density
    α::T # thermal expansivity
    K::T # bulk modulus
    Phydro::T # hydrostatic pressure

    function Density(r; αi=3e-5)
        R = 6.351e6
        ρ = similar(r)
        ρ0 = similar(r)
        α = fill(αi, length(r))
        K = similar(r)
        Phydro = similar(r)

        for i in eachindex(r)
            depth = (R - r[i])
            if depth < 410e3
                Ki = 163
                dKdP = 3.9
                Δρ = 0

            elseif 660e3 > depth ≥ 410e3
                Ki = 85
                dKdP = 3.9
                Δρ = 180

            elseif 2740e3 > depth ≥ 660e3
                Ki = 210
                dKdP = 4
                Δρ = 435 + 180

            elseif depth ≥ 2740e3
                Ki = 210
                dKdP = 4
                Δρ = 61.6 + 435 + 180
            end

            Phydro[i] = hydrostaticPressure(depth)
            ρ0[i] = 3300 + Δρ
            K[i] = (Ki + dKdP * Phydro[i] / 1e9) * 1e9
        end

        return new{typeof(r)}(ρ, ρ0, α, K, Phydro)
    end
end

function hydrostaticPressure(depth)
    dz = 5e3

    if depth > dz
        r = 0:dz:depth
        Phydro = 0

        for i in 2:length(r)
            if r[i] < 410e3
                ρ = 3300

            elseif 660e3 > r[i] ≥ 410e3
                ρ = 3300 + 180

            elseif 2740e3 > r[i] ≥ 660e3
                ρ = 3300 + 435 + 180

            elseif r[i] ≥ 2740e3
                ρ = 3300 + 61.6 + 435 + 180
            end
            Phydro += dz * ρ * 9.81
        end

    else
        Phydro = 0
    end

    return Phydro
end

## EQUATIONS OF STATE 

Base.getindex(ρ::Density, I::Integer) = ρ.ρ[I]
Base.setindex!(ρ::Density, x::Real, I::Integer) = setindex!(ρ.ρ, x, I)

state_equation(α, T; ρ0=1) = @. ρ0 * (1 - α * T);

state_equation!(ρ, α, T; ρ0=1) = Threads.@threads for i in eachindex(T)
    @inbounds ρ[i] = ρ0 * (1 - α * T[i])
end

function state_equation!(ρ::Density, T)
    (; ρ0, Phydro, K, α) = ρ
    Threads.@threads for i in eachindex(T)
         @inbounds ρ[i] = ρ0[i] * (1 - α[i] * T[i] + Phydro[i] / K[i])
      #  @inbounds ρ[i] = ρ0[i] * (1 - α[i] * T[i])
        # @inbounds ρ[i] = 3300 * (1 - α[i] * T[i])
    end
end

effective_viscosity(η1, η2) = 1 / (1 / η1 + 1 / η2)

function druckerPrager(P, C, ϕ)
    sinϕ, cosϕ = sincosd(ϕ)
    # A = 6 * C * cosϕ / (√3 * (3 + sinϕ))
    # B = 2 * sinϕ / (√3 * (3 + sinϕ))
    # return A + B * P
    return cosϕ*C + sinϕ * P
end

## VISCOSITY

function getviscosity!(η::Arrhenius{M}, T, P, εII, τII, e2n, r) where {M}
    # Plastic Parameters
    (; C, ϕ) = η
    # Element parameters
    nip = 7
    nnodel = size(e2n, 1)
    nnodel = 6
    nel = size(e2n, 2)

    # Shape functions
    ni, nn, nn3 = Val(nip), Val(nnodel), Val(3)
    N, = _get_SF(ni, nn)
    N3, = _get_SF(ni, nn3)

    minP = abs(minimum(P))
    λ = 1
    maxr = maximum(r)
    for iel in 1:nel

        # Element arrays
        T_el = @SVector [T[e2n[i, iel]] for i in 1:6]
        r_el = @SVector [r[e2n[i, iel]] for i in 1:6]
        P_el = @SVector [P[e2n[i, iel]] for i in 1:3]
        
        @inbounds for ip in 1:nip
            
            εII_ip = εII[iel, ip]
            T_ip = mydot(T_el, N[ip])
            z_ip = mydot(r_el, N[ip])
            P_ip = mydot(P_el, N3[ip]) + minP
            if (maxr-2780e3) ≤ z_ip ≤ (maxr-660e3)
                λ = 1e1
            elseif z_ip < (maxr-2780e3)
                λ = 1e-3
            elseif z_ip > (maxr-660e3)
                λ = 1
            end

            ηTP = min(max(λ*getviscosity(η, T_ip, P_ip ), 1e18), 1e25)
            # args = (P=P_ip+minP, T=T_ip, f=1e0, d=2e-5)
            # ηTP = min(max(computeViscosity_EpsII(εII_ip, v, args) , 1e18), 1e25)
            
            # Plastic corrections
            if (C > 0) && (ϕ > 0) && (εII_ip != 0.0)  # should be false only in the 1st time step or if C=0 and ϕ=0
                # yield stress
                τy = druckerPrager(P_ip, C*(1+rand()*0.05), ϕ)
                if τy < τII[iel, ip] 
                    # 'plastic viscosity'
                    ηy = 0.5 * τy / εII_ip
                    # η.ip[iel, ip] = max(1e18, min(1e25, effective_viscosity(ηTP, ηy)))
                    η.ip[iel, ip] = max(1e18, min(1e25, ηy))
                end

            else # if plasticity is not active
                η.ip[iel, ip] = ηTP
            end
        end
    end
end
