abstract type Healing end
abstract type NoHealing <: Healing end
abstract type AnnealingOnly <: Healing end
abstract type IsotropicDomainOnly <: Healing end
abstract type AnnealingIsotropicDomain <: Healing end
abstract type FabricDestructionOnly <: Healing end
abstract type AnnealingFabricDestruction <: Healing end

struct FiniteStrainEllipsoid{T}
    x1::T
    x2::T
    y1::T
    y2::T
    a1::T
    a2::T
end

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

    function FiniteStrain(nel; nip=6, ϵ=1e3, r_iso=0)
        # instantiate isotropic finite strain ellipsoid
        fse = [
            FiniteStrainEllipsoid(1.0, 0.0, 0.0, 1.0, 1.0, 1.0) for _ in 1:nel, _ in 1:nip
        ]
        # define active healing mechanism(s)
        # if (annealing_rate != 0) && (ϵ != 0)
        #     type = AnnealingFabricDestruction

        # elseif (annealing_rate != 0) && (ϵ == 0)
        #     type = AnnealingOnly

        # elseif (annealing_rate == 0) && (ϵ != 0)
        #     type = FabricDestructionOnly

        # else
        type = NoHealing

        # end
        # create object
        return new{type}(
            fse,
            Annealing(Matrix{Float64}(undef, nel, nip)),
            FabricDestruction(ϵ),
            IsotropicDomain(r_iso),
        )
    end
end

*(a::Number, annealing::Annealing) = a * annealing.rate
*(annealing::Annealing, a::Number) = a * annealing

function rebuild_FSE(vx1, vx2, vy1, vy2, a1, a2)
    return [
        FiniteStrainEllipsoid(vx1[i], vx2[i], vy1[i], vy2[i], a1[i], a2[i]) for
        i in CartesianIndices(a1)
    ]
end

function isotropic_lithosphere!(F, idx)
    Is = @SMatrix [1.0 0.0; 0.0 1.0]
    Threads.@threads for i in idx
        @inbounds F[i] = Is
    end
end

function healing(F, FSE)
    Is = @SMatrix [1.0 0.0; 0.0 1.0]
    Threads.@threads for i in CartesianIndices(F)
        @inbounds if FSE[i].a1 ./ FSE[i].a2 > 1e2
            F[i] = Is
        end
    end
    return F
end

function eigen_analytic(F)
    evalues = evals(F)
    angles = evects(F)

    imax, imin = eigval_order(evalues) # get the right order of the semi-axis length
    angle1, angle2 = angles[imax], angles[imin]

    FiniteStrainEllipsoid(
        cos(angle1), # vx1
        cos(angle2), # vx2
        sin(angle1), # vy1
        sin(angle2), # vy2
        evalues[imax], # a1 
        evalues[imin], # a2
    )
end

±(a::Number, b::Number) = (a+b, a-b)

function evals(F)
	C1 = sum(Fi*Fi for Fi in F)
	C2 = F[1,1]*F[2,2] - F[1,2]*F[2,1]
	return sqrt.(0.5.*(C1 ± √(C1^2-4C2^2)))
end

function evects(F)
    F11,F12, F21, F22 = F[1,1],F[1,2],F[2,1],F[2,2]
	a = F11*F12 + F21*F22
	b = F12^2 - F11^2 + F22^2 - F21^2

	α = atan.((b ± √(b^2+4a^2))./(2*a))
	ly = @. F21*cos(α) + F22*sin(α)
	lx = @. F11*cos(α) + F[1,2]*sin(α)
	return  @. atan(ly/lx)
end

function getFSE(F, FSE::FiniteStrain{NoHealing})
    # F can grow A LOT in long computations, eventually overflowing at ~1e309
    # Thus we need to normalize F from time to time. We normalize F ∈ Ω w.r.t.
    # the same number (max(F[1])), otherwise a heterogeneous normalization 
    # will screw up the particles interpolation 
    # normalize_F!(F)
    Threads.@threads for iel in CartesianIndices(F)
        @inbounds FSE.fse[iel] = _FSE(F[iel])
    end
    return FSE, F
end

function _FSE(Fi)
    # Compute FSE
    # Fi = normalize_F(F) # normalize F if its too big, otherwise F*F' will be Inf
    L = Fi * Fi' # left-strecth tensor
    # eigval, evect = eigen(Fi * Fi') # get length of FSE semi-axes and orientation
    # imax, imin = eigval_order(eigval) # get the right order of the semi-axis length
    eigval, evect, imax, imin = _eigen(L) # get length of FSE semi-axes and orientation
    # and the right order of eigvals and eigvects

    # Fill FSE
    return FiniteStrainEllipsoid(
        evect[1, imax], # vx1
        evect[1, imin], # vx2
        evect[2, imax], # vy1
        evect[2, imin], # vy2
        √(abs(eigval[imax])), # a1 
        √(abs(eigval[imin])), # a2
    )
end

function normalize_F!(F; ϵ=1e30)
    normalizer = -typemax(Float64) # make sure we take the absolute value, don't want to change signs
    for Fi in F
        @inbounds if abs(Fi[1]) > normalizer
            normalizer = abs(Fi[1])
        end
    end

    # need to check only one component, as the rest will havea similar exponent
    if normalizer > ϵ
        normalizer = 1 / normalizer
        Threads.@threads for i in eachindex(F)
            @inbounds @fastmath F[i] *= normalizer
        end
    end
end

function normalize_F(F; ϵ=1e30)
    Fmax = abs(maximum(F)) # take absolute value because we do not want to change the sign of Fij
    if Fmax > ϵ
        return F ./ Fmax
    else
        return F
    end
end

eigval_order(eigval) = ifelse(@inbounds(eigval[2] > eigval[1]), (2, 1), (1, 2))

function getFSE(F, FSE::FiniteStrain{FabricDestructionOnly})
    Threads.@threads for iel in eachindex(F)
        healing_FSE!(FSE, F, iel)
    end
    return FSE, F
end

@inbounds function healing_FSE!(FSE, F, iel)

    # Compute FSE
    Fi = normalize_F(F[iel]) # normalize F if its too big, otherwise F*F' will be Inf
    eigval, evect = eigen(Fi * Fi') # get length of FSE semi-axes and orientation
    imax, imin = eigval_order(eigval) # get the right order of the semi-axis length

    a1 = √(abs(eigval[imax]))
    a2 = √(abs(eigval[imin]))
    @inbounds if a1 / a2 < FSE.destruction.ϵ
        # Fill FSE
        FSE.fse[iel] = FiniteStrainEllipsoid(
            evect[1, imax]::Float64, # vx1
            evect[1, imin]::Float64, # vx2
            evect[2, imax]::Float64, # vy1
            evect[2, imin]::Float64, # vy2
            a1, # a1 
            a2, # a2
        )

    else
        # destruction of the fabric -> isotropic aggregate
        F[iel] = @SMatrix [1.0 0.0; 0.0 1.0]
        # Fill FSE
        FSE.fse[iel] = FiniteStrainEllipsoid(
            1.0, # vx1
            0.0, # vx2
            0.0, # vy1
            1.0, # vy2
            1.0, # a1 
            1.0, # a2
        )
    end
end

function getFSE(F, FSE::FiniteStrain{AnnealingOnly}, Δt)
    annealing = FSE.annealing.rate
    Threads.@threads for i in eachindex(F)
        @inbounds F[i], FSE.fse[i] = _FSE_annealing(F[i], healing[i])
    end
    return FSE, F
end

unstretch_V(V, h, Δt) = V * exp(-h * Δt) + I * (1 - exp(-h * Δt))

function reconstruct_graddef(F0, L0, B)
    # B = FFᵀ
    R = √(L0) \ F0
    L = √B
    return L * R
end

@inbounds function _FSE_annealing(F, s)

    # Compute FSE
    Fi = normalize_F(F) # normalize F if its too big, otherwise F*F' will be Inf
    L0 = Fi * Fi' # left-strecth tensor

    L = unstretch_V(L0, h, Δt) # unstretch left stretch tensor
    F = reconstruct_graddef(F0, L0, L) # new gradient of deformation tensor

    # eigval, evect = eigen(Fi * Fi') # get length of FSE semi-axes and orientation
    # imax, imin = eigval_order(eigval) # get the right order of the semi-axis length
    eigval, evect, imax, imin = _eigen(L) # get length of FSE semi-axes and orientation
    # and the right order of eigvals and eigvects

    # principal semi-axes
    a1, a2 = √(abs(eigval[imax])), √(abs(eigval[imin]))
    # eigenvectors
    λ11, λ12, λ21, λ22 = evect[1, imax], evect[2, imax], evect[1, imin], evect[2, imin]
    # unstretch semi-axes due to annealing
    a1u, a2u = unstretch_axes(a1, a2, s)
    # Fu = recover_F(Fi, L, λ11, λ12, λ21, λ22, a1u, a2u)
    # Fu = recover_F(Fi, L, λ11, λ12, λ21, λ22, a1, a2)

    # force isotropy in case we unstretched too much
    a1u, a2u = force_isotropy(a1, a2)

    # Fill FSE
    FSE = FiniteStrainEllipsoid(
        λ11, # vx1
        λ21, # vx2
        λ12, # vy1
        λ22, # vy2
        a1u,
        a2u,
    )

    return F, FSE
end

function force_isotropy(a1::T, a2::T) where {T}
    if (a1 ≤ 1) || (a2 ≥ 1)
        a1 = one(T)
        a2 = zero(T)
    end
    return a1, a2
end

function getFSE(F, FSE::FiniteStrain{AnnealingFabricDestruction})
    # F can grow A LOT in long computations, eventually overflowing at ~1e309
    # Thus we need to normalize F from time to time. We normalize F ∈ Ω w.r.t.
    # the same number (max(F[1])), otherwise a heterogeneous normalization 
    # will screw up the particles interpolation 
    normalize_F!(F)
    Threads.@threads for iel in CartesianIndices(F)
        @inbounds FSE.fse[iel], F[iel] = _FSE(F[iel], FSE.destruction, FSE.annealing)
    end
    return FSE, F
end

function _FSE(F, destruction::FabricDestruction, annealing::Annealing)
    ϵ, s = destruction.ϵ, annealing.rate

    # Compute FSE
    Fi = normalize_F(F) # normalize F if its too big, otherwise F*F' will be Inf
    L = Fi * Fi' # left-strecth tensor
    # eigval, evect = eigen(Fi * Fi') # get length of FSE semi-axes and orientation
    # imax, imin = eigval_order(eigval) # get the right order of the semi-axis length
    eigval, evect, imax, imin = _eigen(L) # get length of FSE semi-axes and orientation
    # and the right order of eigvals and eigvects

    # principal semi-axes
    a1, a2 = √(abs(eigval[imax])), √(abs(eigval[imin]))
    # eigenvectors
    λ11, λ12, λ21, λ22 = evect[1, imax], evect[2, imax], evect[1, imin], evect[2, imin]
    # unstretch semi-axes due to annealing
    a1u, a2u = unstretch_axes(a1, a2, s)
    # force isotropy in case we unstretched too much
    a1u, a2u = force_isotropy(a1, a2)

    if a1u / a2u < ϵ # check whether fabric is destroyed or not
        # Fill FSE
        FSE = FiniteStrainEllipsoid(
            λ11, # vx1
            λ21, # vx2
            λ12, # vy1
            λ22, # vy2
            a1u,
            a2u,
        )

    else
        # destruction of the fabric -> isotropic aggregate
        F = @SMatrix [1.0 0.0; 0.0 1.0]
        # Fill FSE
        FSE = FiniteStrainEllipsoid(
            1.0, # vx1
            0.0, # vx2
            0.0, # vy1
            1.0, # vy2
            1.0, # a1 
            1.0, # a2
        )
    end

    return FSE, F
end

function _eigen(A)
    eigval::SVector{2,Float64}, evect::SMatrix{2,2,Float64} = eigen(A) # get length of FSE semi-axes and orientation
    # eigval, evect = eigen(A) # get length of FSE semi-axes and orientation
    imax, imin = eigval_order(eigval) # get the right order of the semi-axis length
    return eigval, evect, imax, imin
end

#=
Unstrecth the FSE semi-axes by a factor of s.
The area of the ellipse must remain the same
after unstretching
=#
function unstretch_axes(a1, a2, s)
    # a2, a1 are the old semiaxes; s is the unstretching factor
    r0 = a2 / a1 # old ratio 
    r = r0 + (1 - r0) * s # new aspect ratio
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

function volume_integral(V, EL2NOD, θ, r)
    ## MODEL PARAMETERS
    nip = 6
    nnodel = size(EL2NOD, 1)
    nnodel = 6
    nel = size(EL2NOD, 2)

    ## INTEGRATION POINTS & DERIVATIVES wrt LOCAL COORDINATES   
    ni, nn, nn3 = Val(nip), Val(nnodel), Val(3)
    N, ∇N, _, w_ip = _get_SF(ni, nn)
    N3, b, dN3ds, c = _get_SF(ni, nn3)

    ## LOOP OVER ELEMENETS
    integral = zeros(nel)
    Threads.@threads for iel in 1:nel
        #= NOTE: For triangular elements with non-curved edges the Jacobian is
              the same for all integration points (i.e. calculated once
              before the integration loop). Further, linear 3-node shape are
              sufficient to calculate the Jacobian. =#

        # Polar coordinates of element nodes
        θ_el = @SVector [θ[i, iel] for i in 1:3]
        r_el = @SVector [r[i, iel] for i in 1:3]
        iel_coords = SMatrix{3,2}([θ_el r_el])

        # Jacobian n. 1 (p:=polar, l:=local): reference element --> current element
        J_pl = dN3ds * iel_coords
        detJ_pl = det(J_pl) # fast |-> unrolled

        # Jacobian ∂ξ∂θ to transform local (ξ, η) into global (θ,r) derivatives
        #     ∂ξ∂θ = [ R_31    -R_21
        #             -Th_31    Th_21] / detJa_PL
        idx = view(EL2NOD, 1:6, iel)
        V_el = SVector{6}(view(V, idx))

        ## INTEGRATION LOOP
        @inbounds for ip in 1:nip
            # Polar coordinates of the integration points
            r_ip = mydot(r_el, N3[ip])

            # Integration weight
            ω = r_ip * detJ_pl * w_ip[ip]

            # field at integration point
            V_ip = mydot(V_el, N[ip])

            # Integral reduction
            integral[iel] += V_ip * ω
        end
    end # end block loop

    return sum(integral)
end # END OF ASSEMBLY FUNCTION

function volume_integral_cartesian(V, gr)
    ## MODEL PARAMETERS
    EL2NOD = gr.e2n
    x = gr.x[gr.e2n]
    z = gr.z[gr.e2n]
    nip = 6
    nnodel = size(EL2NOD, 1)
    nnodel = 6
    nel = size(EL2NOD, 2)

    ## INTEGRATION POINTS & DERIVATIVES wrt LOCAL COORDINATES   
    ni, nn, nn3 = Val(nip), Val(nnodel), Val(3)
    N, ∇N, _, w_ip = _get_SF(ni, nn)
    _, b, dN3ds, c = _get_SF(ni, nn3)

    ## LOOP OVER ELEMENETS
    integral = zeros(nel)
    Threads.@threads for iel in 1:nel
        #= NOTE: For triangular elements with non-curved edges the Jacobian is
              the same for all integration points (i.e. calculated once
              before the integration loop). Further, linear 3-node shape are
              sufficient to calculate the Jacobian. =#

        # Polar coordinates of element nodes
        x_el = @SVector [x[i, iel] for i in 1:3]
        z_el = @SVector [z[i, iel] for i in 1:3]
        iel_coords = SMatrix{3,2}([x_el z_el])

        # Jacobian n. 1 (p:=polar, l:=local): reference element --> current element
        J_pl = dN3ds * iel_coords
        detJ_pl = det(J_pl) # fast |-> unrolled

        idx = view(EL2NOD, 1:6, iel)
        V_el = SVector{6}(view(V, idx))

        ## INTEGRATION LOOP
        @inbounds for ip in 1:nip
            # Integration weight
            ω = detJ_pl * w_ip[ip]

            # field at integration point
            V_ip = mydot(V_el, N[ip])
            # Integral reduction
            integral[iel] += V_ip * ω
        end
    end # end block loop

    return sum(integral)
end # END OF ASSEMBLY FUNCTION
