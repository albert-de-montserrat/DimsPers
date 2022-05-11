struct SymmetricTensor{T}
    xx::Matrix{T}
    zz::Matrix{T}
    xz::Matrix{T}
end

struct Gradient{T}
    ∂x::Matrix{T}
    ∂z::Matrix{T}
end

function initstress(nel; nip=6)
    F = [@SMatrix [1.0 0.0; 0.0 1.0] for _ in 1:nel, _ in 1:nip]
    F0 = deepcopy(F)
    τ = SymmetricTensor(fill(0.0, nel, nip), fill(0.0, nel, nip), fill(0.0, nel, nip))
    ε = SymmetricTensor(fill(0.0, nel, nip), fill(0.0, nel, nip), fill(0.0, nel, nip))
    τII = fill(0.0, nel, nip)
    εII = fill(0.0, nel, nip)
    ∇T = Gradient(fill(0.0, nel, nip), fill(0.0, nel, nip))
    return F, F0, τ, ε, τII, εII, ∇T
end

function _stress!(
    ε,
    εII,
    τ,
    τII,
    η,
    C,
    U,
    iel,
    DoF_U,
    θ,
    r,
    SF_Stress::ShapeFunctionsStress,
    ::Val{Isotropic}
)

    # unpack shape functions
    ∇N, dN3ds, N3 = SF_Stress.∇N, SF_Stress.dN3ds, SF_Stress.N3

    # Polar coordinates of element nodes
    θ_el = @SVector [θ[i, iel] for i in 1:3]
    r_el = @SVector [r[i, iel] for i in 1:3]
    coords = SMatrix{3,2}([θ_el r_el]) # could directly allocate just this one 

    # Jacobian n. 1 (p:=polar, l:=local): reference element --> current element
    J_pl = dN3ds * coords
    # invdetJ_pl = 1/mydet(J_pl)
    invdetJ_pl = 1 / det(J_pl)

    # the Jacobian ∂ξ∂θ to transform local (ξ, η) into global (θ,r) derivatives
    #     ∂ξ∂θ = [ R_31    -R_21
    #             -Th_31    Th_21] / detJa_PL
    R_21 = r_el[2] - r_el[1]  # = -detJa_PL*deta_dth
    R_31 = r_el[3] - r_el[1]  # =  detJa_PL*dxi_dth
    Th_31 = θ_el[3] - θ_el[1] # = -detJa_PL*dxi_dr
    Th_21 = θ_el[2] - θ_el[1] # =  detJa_PL*deta_dr

    Udofs = DoF_U[iel]
    Uel_x = @SVector [
        U[Udofs[1]], U[Udofs[3]], U[Udofs[5]], U[Udofs[7]], U[Udofs[9]], U[Udofs[11]]
    ]
    Uel_z = @SVector [
        U[Udofs[2]], U[Udofs[4]], U[Udofs[6]], U[Udofs[8]], U[Udofs[10]], U[Udofs[12]]
    ]

    # INTEGRATION LOOP
    four_thirds = 4/3
    two_thirds = 2/3
    @inbounds for ip in 1:7

        # Unpack shape functions 
        N3_ip = N3[ip]
        ∇N_ip = ∇N[ip]

        # ρ at ith integration point
        #         η_ip = mydot(SVector{3}([η[e2n[i, iel]] for i in 1:3]), N3_ip)

        # Polar coordinates of the integration points
        θ_ip = dot(θ_el', N3_ip)
        r_ip = dot(r_el', N3_ip)
        invr_ip = 1 / r_ip
        sin_ip, cos_ip = sincos(θ_ip)
        cos_ip_r_ip = cos_ip * invr_ip
        sin_ip_r_ip = sin_ip * invr_ip

        # Build inverse of the 2nd Jacobian
        invJ_double = @SMatrix [
            @muladd(R_31 * cos_ip_r_ip - Th_31 * sin_ip) @muladd(-R_21 * cos_ip_r_ip + Th_21 * sin_ip)
            @muladd(-R_31 * sin_ip_r_ip - Th_31 * cos_ip) @muladd(R_21 * sin_ip_r_ip + Th_21 * cos_ip)
        ]

        # Partial derivatives and strain rate components
        ∂N∂x = invJ_double * ∇N_ip * invdetJ_pl
        # εxx = dot(view(∂N∂x, 1, :), Uel_x) # = ∂Ux∂x 
        # εzz = dot(view(∂N∂x, 2, :), Uel_z) # = ∂Uz∂z 
        # ∂Ux∂z = dot(view(∂N∂x, 2, :), Uel_x)
        # ∂Uz∂x = dot(view(∂N∂x, 1, :), Uel_z)
        ∂N∂x_1 = view(∂N∂x, 1, :)
        ∂N∂x_2 = view(∂N∂x, 2, :)
        εxx = mydot2(Uel_x, ∂N∂x_1) # = ∂Ux∂x 
        εzz = mydot2(Uel_z, ∂N∂x_2) # = ∂Uz∂z 
        ∂Ux∂z = mydot2(Uel_x, ∂N∂x_2)
        ∂Uz∂x = mydot2(Uel_z, ∂N∂x_1)
        εxz = 0.5 * (∂Ux∂z + ∂Uz∂x)

        # update stain rate tensor
        ε.xx[iel, ip] = εxx
        ε.zz[iel, ip] = εzz
        ε.xz[iel, ip] = εxz

        # strain rate second invariant
        εII[iel, ip] = √(@muladd(0.5 * (εxx * εxx + εzz * εzz) + εxz * εxz))

        # deviatoric stress4
        η_ip = η[iel, ip]
        τxx = η_ip * ( four_thirds*εxx + -two_thirds*εzz)
        τzz = η_ip * (-two_thirds*εxx +  four_thirds*εzz)
        τxz = η_ip *   2*εxz
        τ.xx[iel, ip] = τxx
        τ.zz[iel, ip] = τzz
        τ.xz[iel, ip] = τxz
        τII[iel, ip] = √(@muladd(0.5 * (τxx * τxx + τzz * τzz) + τxz * τxz))

    end
end

function _stress!(
    ε,
    εII,
    τ,
    τII,
    η,
    C,
    U,
    iel,
    DoF_U,
    θ,
    r,
    SF_Stress::ShapeFunctionsStress,
    ::Val{Anisotropic}
)

    # unpack shape functions
    ∇N, dN3ds, N3 = SF_Stress.∇N, SF_Stress.dN3ds, SF_Stress.N3

    # Polar coordinates of element nodes
    θ_el = @SVector [θ[i, iel] for i in 1:3]
    r_el = @SVector [r[i, iel] for i in 1:3]
    coords = SMatrix{3,2}([θ_el r_el]) # could directly allocate just this one 

    # Jacobian n. 1 (p:=polar, l:=local): reference element --> current element
    J_pl = dN3ds * coords
    # invdetJ_pl = 1/mydet(J_pl)
    invdetJ_pl = 1 / det(J_pl)

    # the Jacobian ∂ξ∂θ to transform local (ξ, η) into global (θ,r) derivatives
    #     ∂ξ∂θ = [ R_31    -R_21
    #             -Th_31    Th_21] / detJa_PL
    R_21 = r_el[2] - r_el[1]  # = -detJa_PL*deta_dth
    R_31 = r_el[3] - r_el[1]  # =  detJa_PL*dxi_dth
    Th_31 = θ_el[3] - θ_el[1] # = -detJa_PL*dxi_dr
    Th_21 = θ_el[2] - θ_el[1] # =  detJa_PL*deta_dr

    Udofs = DoF_U[iel]
    Uel_x = @SVector [
        U[Udofs[1]], U[Udofs[3]], U[Udofs[5]], U[Udofs[7]], U[Udofs[9]], U[Udofs[11]]
    ]
    Uel_z = @SVector [
        U[Udofs[2]], U[Udofs[4]], U[Udofs[6]], U[Udofs[8]], U[Udofs[10]], U[Udofs[12]]
    ]

    # INTEGRATION LOOP
    # onesixth = 1/6
    @inbounds for ip in 1:7

        # Unpack shape functions 
        N3_ip = N3[ip]
        ∇N_ip = ∇N[ip]

        # ρ at ith integration point
        #         η_ip = mydot(SVector{3}([η[e2n[i, iel]] for i in 1:3]), N3_ip)

        # Polar coordinates of the integration points
        θ_ip = dot(θ_el', N3_ip)
        r_ip = dot(r_el', N3_ip)
        invr_ip = 1 / r_ip
        sin_ip, cos_ip = sincos(θ_ip)
        cos_ip_r_ip = cos_ip * invr_ip
        sin_ip_r_ip = sin_ip * invr_ip

        # Build inverse of the 2nd Jacobian
        invJ_double = @SMatrix [
            @muladd(R_31 * cos_ip_r_ip - Th_31 * sin_ip) @muladd(-R_21 * cos_ip_r_ip + Th_21 * sin_ip)
            @muladd(-R_31 * sin_ip_r_ip - Th_31 * cos_ip) @muladd(R_21 * sin_ip_r_ip + Th_21 * cos_ip)
        ]

        # Partial derivatives and strain rate components
        ∂N∂x = invJ_double * ∇N_ip * invdetJ_pl
        εxx = dot(view(∂N∂x, 1, :), Uel_x) # = ∂Ux∂x 
        εzz = dot(view(∂N∂x, 2, :), Uel_z) # = ∂Uz∂z 
        ∂Ux∂z = dot(view(∂N∂x, 2, :), Uel_x)
        ∂Uz∂x = dot(view(∂N∂x, 1, :), Uel_z)
        εxz = 0.5 * (∂Ux∂z + ∂Uz∂x)

        # update stain rate tensor
        ε.xx[iel, ip] = εxx
        ε.zz[iel, ip] = εzz
        ε.xz[iel, ip] = εxz

        # strain rate second invariant
        εII[iel, ip] = √(@muladd(0.5 * (εxx * εxx + εzz * εzz) + εxz * εxz))

        # deviatoric stress in the anisotropic case
        η_ip = η[iel, ip]
        η11 = C.η11[iel, ip] * η_ip
        η22 = C.η33[iel, ip] * η_ip
        η33 = C.η55[iel, ip] * η_ip
        η12 = C.η13[iel, ip] * η_ip
        η13 = C.η15[iel, ip] * η_ip
        η23 = C.η35[iel, ip] * η_ip
        τxx =
            η12 * ( -εxx/3 + 2*εzz/3) +
            η11 * ( 2εxx/3   - εzz/3) +
            η13 * εxz
        τzz =
            η22 * ( -εxx/3 + 2*εzz/3) +
            η12 * ( 2εxx/3   - εzz/3) +
            η23 * 2εxz
        τxz =
            η23 * ( -εxx/3 + 2*εzz/3) +
            η13 * ( 2εxx/3   - εzz/3) +
            η33 * εxz

        τ.xx[iel, ip] = τxx
        τ.zz[iel, ip] = τzz
        τ.xz[iel, ip] = τxz
        τII[iel, ip] = √(@muladd(0.5 * (τxx * τxx + τzz * τzz) + τxz * τxz))

    end
end

function stress!(ε, εII, τ, τII, η, C, U, nel, DoF_U, coordinates, SF_Stress)
    @batch per = core for iel in 1:nel
        _stress!(ε, εII, τ, τII, η.ip, C, U, iel, DoF_U, coordinates.θ, coordinates.r, SF_Stress, Val(η))
        # _stress!(F, ε, εII, τ, η.node, U, e2n, iel, DoF_U, coordinates.θ, coordinates.r, SF_Stress, Δt)
    end
end

function _deformation_gradient!(F, U, iel, DoF_U, θ, r, SF_Stress::ShapeFunctionsStress, Δt)

    # unpack shape functions
    ∇N, dN3ds, N3 = SF_Stress.∇N, SF_Stress.dN3ds, SF_Stress.N3

    # Polar coordinates of element nodes
    θ_el = @SVector [θ[i, iel] for i in 1:3]
    r_el = @SVector [r[i, iel] for i in 1:3]
    coords = SMatrix{3,2}([θ_el r_el]) # could directly allocate just this one 

    # Jacobian n. 1 (p:=polar, l:=local): reference element -> current element
    J_pl = dN3ds * coords
    invdetJ_pl = 1 / det(J_pl)

    # the Jacobian ∂ξ∂θ to transform local (ξ, η) into global (θ,r) derivatives
    #     ∂ξ∂θ = [ R_31    -R_21
    #             -Th_31    Th_21] / detJa_PL
    R_21 = r_el[2] - r_el[1]  # = -detJa_PL*deta_dth
    R_31 = r_el[3] - r_el[1]  # =  detJa_PL*dxi_dth
    Th_31 = θ_el[3] - θ_el[1] # = -detJa_PL*dxi_dr
    Th_21 = θ_el[2] - θ_el[1] # =  detJa_PL*deta_dr

    Udofs = DoF_U[iel]
    Uel_x = @SVector [
        U[Udofs[1]], U[Udofs[3]], U[Udofs[5]], U[Udofs[7]], U[Udofs[9]], U[Udofs[11]]
    ]
    Uel_z = @SVector [
        U[Udofs[2]], U[Udofs[4]], U[Udofs[6]], U[Udofs[8]], U[Udofs[10]], U[Udofs[12]]
    ]

    # INTEGRATION LOOP
    onesixth = 1 / 6
    @inbounds for ip in 1:7

        # Unpack shape functions 
        N3_ip = N3[ip]
        ∇N_ip = ∇N[ip]

        # Polar coordinates of the integration points
        θ_ip = mydot2(θ_el, N3_ip)
        r_ip = mydot2(r_el, N3_ip)
        invr_ip = 1 / r_ip
        sin_ip, cos_ip = sincos(θ_ip)
        cos_ip_r_ip = cos_ip * invr_ip
        sin_ip_r_ip = sin_ip * invr_ip

        # Build inverse of the 2nd Jacobian
        invJ_double = @SMatrix [
            @muladd(R_31 * cos_ip_r_ip - Th_31 * sin_ip) @muladd(-R_21 * cos_ip_r_ip + Th_21 * sin_ip)
            @muladd(-R_31 * sin_ip_r_ip - Th_31 * cos_ip) @muladd(R_21 * sin_ip_r_ip + Th_21 * cos_ip)
        ]

        # Partial derivatives and strain rate components
        ∂N∂x = invJ_double * ∇N_ip * invdetJ_pl
        ∂Ux∂x = mydot2(Uel_x, view(∂N∂x, 1, :))
        ∂Uz∂z = mydot2(Uel_z, view(∂N∂x, 2, :))
        ∂Ux∂z = mydot2(Uel_x, view(∂N∂x, 2, :))
        ∂Uz∂x = mydot2(Uel_z, view(∂N∂x, 1, :))

        # transpose of the velocity gradient
        ∇Uᵀ = @SMatrix [
            ∂Ux∂x ∂Ux∂z
            ∂Uz∂x ∂Uz∂z
        ]

        # # as in McKenzie
        # δij = @SMatrix [
        #     1.0 0.0
        #     0.0 1.0
        # ]
        # A = δij - 0.5*Δt*∇Uᵀ
        # B = δij + 0.5*Δt*∇Uᵀ
        # F[iel, ip] *= A\B

        # precompute velocity gradient × Δt
        Δt∇Uᵀ = Δt * ∇Uᵀ

        # Integrate (RK4) and update gradient of deformation tensor F
        F0 = F[iel, ip]
        k1 = Δt∇Uᵀ * F0
        Fi = k1 * 0.5 + F0
        k2 = Δt∇Uᵀ * Fi
        Fi = k2 * 0.5 + F0
        k3 = Δt∇Uᵀ * Fi
        Fi = k3 + F0
        k4 = Δt∇Uᵀ * Fi
        
        F[iel, ip] = @muladd(F0 + (k1 + 2 * (k2 + k3) + k4) * onesixth)
    end
end

function deformation_gradient!(F, U, nel, DoF_U, coordinates, SF_Stress, Δt)
    # @batch per = core 
    for iel in 1:nel
        _deformation_gradient!(
            F, U, iel, DoF_U, coordinates.θ, coordinates.r, SF_Stress, Δt
        )
    end
end

secondinvariant(xx, zz, xz) = sqrt(0.5 * (xx^2 + zz^2) + xz^2)

@inline function secondinvariant(A::SymmetricTensor)
    II = Matrix{Float64}(undef, size(A.xx))
    xx, zz, xz = A.xx, A.zz, A.xz
    @batch for i in eachindex(xx)
        II[i] = √(@muladd(0.5 * (xx[i] * xx[i] + zz[i] * zz[i]) + xz[i] * xz[i]))
    end
    return II
end

@inline function secondinvariant!(II, A::SymmetricTensor)
    xx, zz, xz = A.xx, A.zz, A.xz
    @batch for i in eachindex(xx)
        @inbounds II[i] = √(@muladd(0.5 * (xx[i] * xx[i] + zz[i] * zz[i]) + xz[i] * xz[i]))
    end
end

function shearheating(τ::SymmetricTensor{T}, ε::SymmetricTensor{T}) where {T}
    shear_heating = fill(0.0, size(τ.xx))
    @tturbo for i in CartesianIndices(shear_heating) # faster than threading for current matrices size
        shear_heating[i] =
            τ.xx[i] * ε.xx[i] + τ.zz[i] * ε.zz[i] + 2τ.xz[i] * ε.xz[i]
    end
    return shear_heating
end

function rowdot!(C, A, B)
    @inbounds for i in 1:size(A, 1)
        C[i] = mydot(view(A, i, :), view(B, i, :))
    end
end
