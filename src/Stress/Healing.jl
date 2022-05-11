abstract type Healing end
abstract type NoHealing <: Healing end
abstract type AnnealingOnly <: Healing end
abstract type IsotropicDomainOnly <: Healing end
abstract type AnnealingIsotropicDomain <: Healing end
abstract type FabricDestructionOnly <: Healing end
abstract type AnnealingFabricDestruction <: Healing end

# Define isotropic domain
struct IsotropicDomain{T}
    r::T  # depth above which Ω is isotropic
end

# Rate of annealing
struct Annealing{T}
    rate::T # rate of annealing
end

# Destruction of fabric
struct FabricDestruction{T}
    ϵ::T # ratio a1/a2 at which aggregate goes back to its anisotropic form
end

struct FiniteStrain{T} <: Healing
    fse::Matrix{FiniteStrainEllipsoid{Float64}} # matrix containing the FSE at every ip
    annealing::Annealing # rate of annealing
    destruction::FabricDestruction # ratio a1/a2 at which aggregate goes back to its anisotropic form
    isotropic_domain::IsotropicDomain # depth above which Ω is isotropic

    function FiniteStrain(nel; nip=6, ϵ=1e3, annealing_rate=0, r_iso=0)
        # instantiate isotropic finite strain ellipsoid
        fse = [
            FiniteStrainEllipsoid(1.0, 0.0, 0.0, 1.0, 1.0, 1.0) for _ in 1:nel, _ in 1:nip
        ]
        # define active healing mechanism(s)
        if (annealing_rate != 0) && (ϵ != 0)
            type = AnnealingFabricDestruction

        elseif (annealing_rate != 0) && (ϵ == 0)
            type = AnnealingOnly

        elseif (annealing_rate == 0) && (ϵ != 0)
            type = FabricDestructionOnly

        else
            type = NoHealing
        end
        # create object
        return new{type}(fse, Annealing(annealing_rate), IsotropicDomain(r_iso))
    end
end

*(a::Number, annealing::Annealing) = a * annealing.rate
*(annealing::Annealing, a::Number) = a * annealing.rate

#=
Unstrecth the FSE semi-axes by a factor of s.
The area of the ellipse must remain the same
after unstretching
=#
function unstretch_axes(a1, a2, s)
    # a2, a1 are the old semiaxes
    # s is the unstretching factor
    r0 = a2 / a1 # old ratio 
    r = @muladd r0 + (1 - r0) * s # new aspect ratio
    A = π * a2 * a1 # Area of the FSE
    # find new unstrecthed axes such that the   
    # aspect ratio is r and A remains untouched
    a1u = √(A / (π * r))
    a2u = r * a1u

    return a1u, a2u
end

#=
Recover the new F tensor after unstretching the FSE semi-axes.
We know that F = LR, where L is the left-stretch tensor and R⁻¹=Rᵀ 
is the orthogonal rotation tensor. First we recover the new L using
the old eigenvalues (we do not want to rotate the FSE, only unstretch 
it) and new eigenvalues:
    L = PDP⁻¹
where P = [λ₁ λ₃ λ₃] and D = I⋅[a₁ a₃ a₃]ᵀ. R is obtained using the 
old F and L0 tensors:
    R = F\L0
and finally we obtain the new deformation gradient tensor:
    Fn = LR
=#
function recover_F(F, L0, λ11, λ12, λ21, λ22, a1, a2)
    P = @SMatrix [
        λ11 λ21
        λ12 λ22
    ]
    D = @SMatrix [
        a1*a1 0.0
        0.0 a2*a2
    ]
    # recover left-stretch tensor 
    L = P * D \ P
    # rotation tensor R = F\(F*Fᵀ) = F\l0
    R = F \ L0
    # unstretched tensor of deformation
    return Fu = L * R
end
