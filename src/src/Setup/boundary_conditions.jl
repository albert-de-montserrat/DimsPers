struct BC{T,N}
    Ω::Vector{T}
    ifree::Vector{T}
    vfix::Vector{N}
end

function init_BCs(
    gr, IDs; Ttop=0.0, Tbot=1.0, velocity_type=:free_slip, temperature_type=:heated
)
    TBC = temperature_bcs(gr, IDs; Ttop=Ttop, Tbot=Tbot, type=temperature_type)
    UBC = velocity_bcs(gr, IDs; type=velocity_type)
    return TBC, UBC
end

function velocity_bcs(M::Grid, ids; type=:free_slip)
    if type == :free_slip
        Ωu, ufix = free_slip(ids)
    elseif type == :no_slip
        Ωu, ufix = no_slip(ids)
    end

    nU = maximum(M.e2n)
    ifree = setdiff(collect(1:(2nU)), Ωu)
    return BC(Ωu, ifree, ufix)
end

function free_slip(ids)
    top_nodes = findall(x -> x == :outter, ids)
    bot_nodes = findall(x -> x == :inner, ids)
    nodes = vcat(top_nodes, bot_nodes)

    # fix radial velocity only
    Ωu = @. 2 * nodes
    ufix = fill(0.0, length(Ωu))

    return Ωu, ufix
end

function free_slip(nr::Int, ids)
    top_nodes = findall(x -> x == :outter, ids)
    bot_nodes = findall(x -> x == :inner, ids)
    nodes = vcat(top_nodes, bot_nodes)

    # fix radial velocity only
    Ωu = @. 2 * nodes
    ufix = fill(0.0, length(Ωu))

    # fix tangential velocity of one single node
    push!(Ωu, 2 * nr - 1)
    push!(ufix, 0.0)

    return Ωu, ufix
end

function no_slip(ids)
    nodes = findall(x -> (x == :outter) || (x == :inner), ids)
    # bot_nodes = findall(x-> x == :inner, ids)
    # nodes = vcat(top_nodes, bot_nodes)

    # fix radial velocity only
    Ωu1 = @. 2 * nodes
    Ωu2 = @. 2 * nodes - 1
    Ωu = vcat(Ωu1, Ωu2)
    ufix = fill(0.0, length(Ωu))

    return Ωu, ufix
end

function temperature_bcs(M::Grid, ids; Ttop=0.0, Tbot=1.0, type=:heated)
    if type == :heated
        return heated_bottom(M, ids, Ttop, Tbot)
    elseif type == :insulated
        return insulated_bottom(M, ids, Ttop)
    end
end

function heated_bottom(M::Grid, ids, Ttop=0.0, Tbot=1.0)
    # Top nodes & temperature
    top_nodes = findall(x -> x == :outter, ids)
    top_T = fill(Ttop, length(top_nodes))

    # Bottom nodes & temperature
    bot_nodes = findall(x -> x == :inner, ids)
    bot_T = fill(Tbot, length(top_nodes))

    # Concatenate DoFs and T
    ΩT = vcat(top_nodes, bot_nodes)
    tfix = vcat(top_T, bot_T)

    nT = length(M.x)
    tfree = setdiff(collect(1:nT), ΩT)
    return BC(ΩT, tfree, tfix)
end

function insulated_bottom(M::Grid, ids, Ttop=0.0)
    # Top nodes & temperature
    ΩT = findall(x -> x == :outter, ids)
    tfix = fill(Ttop, length(ΩT))

    nT = length(M.x)
    tfree = setdiff(collect(1:nT), ΩT)
    return BC(ΩT, tfree, tfix)
end
