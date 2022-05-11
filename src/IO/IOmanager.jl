
struct IOs{T,I}
    path::T
    folder::T
    filename::T
    iplot::I
end

function setup_output(path, folder)
    filename = "file"
    iplot = Int32(0)
    OUT = IOs(path, folder, filename, iplot)
    mkpath(joinpath(path, folder))
    return OUT, iplot
end

function savedata(
    OUT,
    Upolar,
    Ucartesian,
    T,
    η,
    𝓒,
    ρ,
    ε,
    F,
    FSE,
    nθ,
    nr,
    particle_fields,
    particle_info,
    time2save::Float64,
    ::Val{Isotropic},
)

    # unpack
    path, folder, filename, iplot = OUT.path, OUT.folder, OUT.filename, OUT.iplot

    # output path 
    filename = string(filename, "_", iplot)
    final_path = string(joinpath(pwd(), path, folder, filename), ".h5")

    # Prepare F
    Fxx = [F[i, j][1, 1] for i in axes(F, 1), j in axes(F, 2)]
    Fzz = [F[i, j][2, 2] for i in axes(F, 1), j in axes(F, 2)]
    Fxz = [F[i, j][1, 2] for i in axes(F, 1), j in axes(F, 2)]
    Fzx = [F[i, j][2, 1] for i in axes(F, 1), j in axes(F, 2)]

    # Prepare FSE
    a1 = [FSE[i, j].a1 for i in axes(FSE, 1), j in axes(FSE, 2)]
    a2 = [FSE[i, j].a2 for i in axes(FSE, 1), j in axes(FSE, 2)]
    x1 = [FSE[i, j].x1 for i in axes(FSE, 1), j in axes(FSE, 2)]
    x2 = [FSE[i, j].x2 for i in axes(FSE, 1), j in axes(FSE, 2)]
    y1 = [FSE[i, j].y1 for i in axes(FSE, 1), j in axes(FSE, 2)]
    y2 = [FSE[i, j].y2 for i in axes(FSE, 1), j in axes(FSE, 2)]

    # Prepare output variables
    Uθ, Ur = getvelocity(Upolar)
    Ux, Uz = getvelocity(Ucartesian)

    # Unpack particle info 
    np = length(particle_info)
    xp = [particle_info[i].CCart.x for i in 1:np]
    zp = [particle_info[i].CCart.z for i in 1:np]
    t = [particle_info[i].t for i in 1:np]
    Tp = particle_fields.T

    # Save output
    h5open(final_path, "w") do file

        # create groups
        MESH = create_group(file, "MESH") # create a group
        VAR = create_group(file, "VAR") # create a group
        PART = create_group(file, "PART") # create a group
        Time = create_group(file, "Time") # create a group

        # time 
        Time["t"] = time2save

        # mesh variables
        MESH["nθ"] = nθ # create a scalar dataset inside the group
        MESH["nr"] = nr # create a scalar dataset inside the group

        # physical variables
        save_viscosity!(VAR, η)
        VAR["strain_xx"] = ε.xx
        VAR["strain_zz"] = ε.zz
        VAR["strain_xz"] = ε.xz
        VAR["Ux"] = Ux
        VAR["Uz"] = Uz
        VAR["Utheta"] = Uθ
        VAR["Ur"] = Ur
        VAR["T"] = T
        # VAR["nu"] = η.node
        VAR["density"] = ρ
        VAR["Fxx"] = Fxx
        VAR["Fzz"] = Fzz
        VAR["Fxz"] = Fxz
        VAR["Fzx"] = Fzx
        VAR["a1"] = a1
        VAR["a2"] = a2
        VAR["x1"] = x1
        VAR["x2"] = x2
        VAR["y1"] = y1
        VAR["y2"] = y2

        #         # particle variables
        #         PART["x"] = xp
        #         PART["z"] = zp
        #         PART["t"] = t
        #         PART["T"] = Tp

    end
end

function save_viscosity!(VAR, η::TemperatureDependantPlastic)
    VAR["nu_node"] = η.node
    return VAR["nu_ip"] = η.ip
end

function save_viscosity!(VAR, η::Arrhenius)
    VAR["nu_node"] = η.node
    return VAR["nu_ip"] = η.ip
end

function save_viscosity!(VAR, η::Isoviscous{Isotropic})
    return VAR["nu_node"] = η.val
end

function save_viscosity!(VAR, η)
    return VAR["nu_node"] = η.node
end

function savedata(
    OUT,
    Upolar,
    Ucartesian,
    T,
    η,
    𝓒,
    ρ,
    ε,
    F,
    FSE,
    nθ,
    nr,
    particle_fields,
    particle_info,
    time2save,
    ::Val{Anisotropic},
)

    # unpack
    path, folder, filename, iplot = OUT.path, OUT.folder, OUT.filename, OUT.iplot

    # output path 
    filename = string(filename, "_", iplot)
    final_path = string(joinpath(pwd(), path, folder, filename), ".h5")

    # Prepare output variables
    Uθ, Ur = getvelocity(Upolar)
    Ux, Uz = getvelocity(Ucartesian)

    # Prepare F
    Fxx = [F[i, j][1, 1] for i in axes(F, 1), j in axes(F, 2)]
    Fzz = [F[i, j][2, 2] for i in axes(F, 1), j in axes(F, 2)]
    Fxz = [F[i, j][1, 2] for i in axes(F, 1), j in axes(F, 2)]
    Fzx = [F[i, j][2, 1] for i in axes(F, 1), j in axes(F, 2)]

    # Prepare FSE
    a1 = [FSE[i, j].a1 for i in axes(FSE, 1), j in axes(FSE, 2)]
    a2 = [FSE[i, j].a2 for i in axes(FSE, 1), j in axes(FSE, 2)]
    x1 = [FSE[i, j].x1 for i in axes(FSE, 1), j in axes(FSE, 2)]
    x2 = [FSE[i, j].x2 for i in axes(FSE, 1), j in axes(FSE, 2)]
    y1 = [FSE[i, j].y1 for i in axes(FSE, 1), j in axes(FSE, 2)]
    y2 = [FSE[i, j].y2 for i in axes(FSE, 1), j in axes(FSE, 2)]

    # Unpack particle info 
    np = length(particle_info)
    xp = [particle_info[i].CCart.x for i in 1:np]
    zp = [particle_info[i].CCart.z for i in 1:np]
    t = [particle_info[i].t for i in 1:np]
    Tp = particle_fields.T

    # Save output
    h5open(final_path, "w") do file

        # create groups
        MESH = create_group(file, "MESH") # create a group
        VAR = create_group(file, "VAR") # create a group
        PART = create_group(file, "PART") # create a group
        Time = create_group(file, "Time") # create a group

        # time 
        Time["t"] = time2save

        # mesh variables
        MESH["nθ"] = nθ                    # create a scalar dataset inside the group
        MESH["nr"] = nr                     # create a scalar dataset inside the group

        # physical variables
        save_viscosity!(VAR, η)
        VAR["strain_xx"] = ε.xx
        VAR["strain_zz"] = ε.zz
        VAR["strain_xz"] = ε.xz
        VAR["Ux"] = Ux
        VAR["Uz"] = Uz
        VAR["Utheta"] = Uθ
        VAR["Ur"] = Ur
        VAR["T"] = T
        VAR["nu"] = η.node
        VAR["density"] = ρ
        VAR["nu11"] = 𝓒.η11
        VAR["nu33"] = 𝓒.η33
        VAR["nu55"] = 𝓒.η55
        VAR["nu13"] = 𝓒.η13
        VAR["nu15"] = 𝓒.η15
        VAR["nu35"] = 𝓒.η35

        VAR["Fxx"] = Fxx
        VAR["Fzz"] = Fzz
        VAR["Fxz"] = Fxz
        VAR["Fzx"] = Fzx

        VAR["a1"] = a1
        VAR["a2"] = a2
        VAR["x1"] = x1
        VAR["x2"] = x2
        VAR["y1"] = y1
        VAR["y2"] = y2

        #         # particle variables
        #         PART["x"] = xp
        #         PART["z"] = zp
        #         PART["t"] = t
        #         PART["T"] = Tp

    end
end

function reloader(fname)
    fid = h5open(fname, "r")

    # -- Load fields
    V = fid["VAR"]
    T = read(V, "T")

    Fxx = read(V["Fxx"])
    Fzz = read(V["Fzz"])
    Fxz = read(V["Fxz"])
    Fzx = read(V["Fzx"])

    F = [[Fxx[u] Fxz[u]; Fzx[u] Fzz[u]] for u in CartesianIndices(Fxx)]
    return T, F
end

function load_particles(fname)
    fid = h5open(fname, "r")

    # -- Load fields
    V = fid["PART"]
    px = read(V, "x")
    pz = read(V, "z")

    return px, pz
end

function setup_metrics(gr, path, folder; load=false)
    ScratchNu = ScratchNusselt(gr)
    stats_file = joinpath(pwd(), path, folder, "statistics.txt")
    if !load && isfile(stats_file)
        rm(stats_file)
    end
    return ScratchNu, stats_file
end

function write_stats(U, T, np, gr, Time, ScratchNu, stats_file)
    # V = ones(size(S))
    # A = volume_integral_cartesian(V, gr) # total area
    r_in, r_out = extrema(gr.r)
    Area = π * (r_out^2 - r_in^2)
    S = @. @views U[1:2:end]^2 + U[2:2:end]^2
    Urms = sqrt(volume_integral_cartesian(S, gr) ./ Area) # velocity root mean square
    Tav = volume_integral_cartesian(T, gr) ./ Area # volume averaged temperature

    # -- Nusselt number
    Ttop_minus, Tbot_minus = T[ScratchNu.itop_minus], T[ScratchNu.ibot_minus]
    ∇T_top = @. (0.0 - Ttop_minus) / ScratchNu.Δr
    ∇T_bot = @. (Tbot_minus - 1.0) / ScratchNu.Δr
    Nu_top = ScratchNu.A * ScratchNu.Δθ * sum(∇T_top) * r_out
    Nu_bot = ScratchNu.A * ScratchNu.Δθ * sum(∇T_bot) * r_in

    open(stats_file, "a") do io
        println(
            io,
            Float32(Time),
            " ",
            Float32(Urms),
            " ",
            Float32(Tav),
            " ",
            Float32(Nu_top),
            " ",
            Float32(Nu_bot),
            " ",
            np,
        )
    end

    return println(
        "Urms = ",
        Urms,
        " ⟨T⟩ = ",
        Tav,
        " Nusselt top =",
        Nu_top,
        " Nusselt bot =",
        Nu_bot,
        " #particles =",
        np,
    )
end

struct Nusselt{T,N}
    A::T
    f::T
    Δr::T
    Δθ::T
    itop_minus::Vector{N}
    ibot_minus::Vector{N}
end

function ScratchNusselt(gr)
    θ, r = gr.θ, gr.r
    r_in, r_out = 1.22, 2.22

    runique = unique(r)
    sort!(runique)
    itop_minus, ibot_minus = findall(x -> x == runique[end - 1], r),
    findall(x -> x == runique[2], r)
    θtop_minus = view(θ, itop_minus)

    Δr = runique[2] - runique[1]
    Δθ = θtop_minus[1] - θtop_minus[2]
    f = r_in / r_out
    A = log(f) / (2π * r_out * (1 - f))

    return Nusselt(A, f, Δr, Δθ, itop_minus, ibot_minus)
end
