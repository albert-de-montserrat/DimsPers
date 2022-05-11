abstract type Cartesian end
abstract type Polar end

mutable struct Point2D{T}
    x::Float64
    z::Float64
end

struct Grid{A,B,C,D}
    x::Vector{A}
    z::Vector{A}
    θ::Vector{A}
    r::Vector{A}
    Ro::A
    Ri::A
    e2n::Matrix{B}
    e2n_p1::Matrix{B}
    e2nP::Matrix{B}
    nθ::C
    nr::C
    nel::C
    nnod::C
    nVnod::C
    nvert::C
    neighbours::D
    neighbours_p1::D
end

struct ElementCoordinates{T}
    θ::T
    r::T
end

cartesian2polar(x, z) = (atan(x, z), sqrt(x^2 + z^2))

polar2cartesian(x, z) = (z * sin(x), z * cos(x))

function init_grid(N; split=1, type=:adimensional)
    r_out, r_in = type == :adimensional ? (2.22, 1.22) : (6.351e6, 6.351e6 - 2.89e6)

    if split == 1
        nr = Int(1 + 2^N)
        nθ = Int(12 * 2^N)
        gr = Grid_split1(nθ, nr)
    else
        nr = Int(1 + 2^N)
        nθ = Int(12 * 2^N)
        if N == 5
            nr = Int(1 + 32)
            nθ = Int(256)
        end
        gr = Grid(nθ, nr, r_out, r_in)
    end
    IDs = point_ids(gr)

    return gr, IDs
end

function Grid(nθ::Int64, nr::Int64, r_out, r_in)
    nn = nr * (nθ)# total number of nodes
    nels = (nr - 1) * nθ
    r = fill(0.0, nn)
    θ = fill(0.0, nn)
    angle = deg2rad(0)
    dθ = 2 * π / nθ

    # -- Nodal positions
    Ir = fill(1.0, nr)
    @inbounds for ii in 1:nθ
        idx = @. (1:nr) + nr * (ii - 1)
        r[idx] = LinRange(r_in, r_out, nr)
        @. (θ[idx] = angle + dθ * (ii - 1) * Ir)
    end

    # -- Quadrilateral elements
    id_el = Array{Int64,2}(undef, nels, 4)
    @inbounds for ii in 1:nθ
        if ii < nθ
            idx = @. (1:(nr - 1)) + (nr - 1) * (ii - 1)
            idx1 = @. (1:(nr - 1)) + nr * (ii - 1)
            idx2 = @. (1:(nr - 1)) + nr * (ii)
            id_el[idx, :] .= [idx1 idx2 idx2 .+ 1 idx1 .+ 1]
        else
            idx = @. (1:(nr - 1)) + (nr - 1) * (ii - 1)
            idx1 = @. (1:(nr - 1)) + nr * (ii - 1)
            idx2 = @. 1:(nr - 1)
            id_el[idx, :] .= [idx1 idx2 idx2 .+ 1 idx1 .+ 1]
        end
    end

    # Split quadrilateral in 2 triangles
    ntri = nels * 4
    id_els_tri = Array{Int64,2}(undef, ntri, 3)
    dθ /= 2
    angle = dθ
    dr = r[2] - r[1]
    r_c, θ_c = fill(0.0, nels), fill(0.0, nels)
    id_centers = nn .+ (1:nels)
    #         id_el       = [id_el id_centers']
    i1 = [1, 2]
    i2 = [2, 3]
    i3 = [3, 4]
    i4 = [4, 1]
    @inbounds for ii in 1:nels
        # connectivity matrix
        idx = @. (1:4) + 4 * (ii - 1)
        id_els_tri[idx[1], :] = [id_el[ii, i1]' id_centers[ii]]
        id_els_tri[idx[2], :] = [id_el[ii, i2]' id_centers[ii]]
        id_els_tri[idx[3], :] = [id_el[ii, i3]' id_centers[ii]]
        id_els_tri[idx[4], :] = [id_el[ii, i4]' id_centers[ii]]
        # coords of central node            
        θ_c[ii] = θ[id_el[ii, 1]] + dθ
        r_c[ii] = r[id_el[ii, 1]] + dr / 2
    end

    r = vcat(r, r_c)
    θ = vcat(θ, θ_c)
    EL2NOD = Array(id_els_tri')
    neighbours = element_neighbours(EL2NOD)
    θ, r, EL2NOD = add_midpoints(EL2NOD, neighbours, θ, r)
    EL2NOD_P1 = p2top1(EL2NOD)
    neighbours_p1 = element_neighbours(EL2NOD_P1)
    nel = size(EL2NOD, 2)
    nVnod = maximum(view(EL2NOD, 1:3, :))
    # EL2NOD, θ, r =  sixth_node(EL2NOD, θ, r)
    x, z = polar2cartesian(θ, r)
    nnod = length(x)
    # EL2NODP = Matrix(reshape(1:3*nel,3,nel))
    EL2NODP = EL2NOD[1:3, :]

    return Grid(
        x,
        z,
        θ,
        r,
        r_out,
        r_in,
        EL2NOD,
        EL2NOD_P1,
        EL2NODP,
        nθ,
        nr,
        nel,
        nnod,
        nVnod,
        3,
        neighbours,
        neighbours_p1,
    )
end

function Grid_split1(nθ::Int64, nr::Int64)
    # 1 = split rectangle into 2 triangles
    # 2 = split rectangle into 4 triangles
    R = 2.22
    r_out = R
    r_in = 1.22
    nn = nr * (nθ)# total number of nodes
    nels = (nr - 1) * nθ
    r = fill(0.0, nn)
    θ = fill(0.0, nn)
    angle = deg2rad(0)
    # angle   = 0
    dθ = 2 * π / nθ

    # -- Nodal positions
    Ir = fill(1.0, nr)
    @inbounds for ii in 1:nθ
        idx = @. (1:nr) + nr * (ii - 1)
        r[idx] = LinRange(r_in, r_out, nr)
        @. (θ[idx] = angle + dθ * (ii - 1) * Ir)
    end

    # -- Quadrilateral elements
    id_el = Array{Int64,2}(undef, nels, 4)
    @inbounds for ii in 1:nθ
        if ii < nθ
            idx = @. (1:(nr - 1)) + (nr - 1) * (ii - 1)
            idx1 = @. (1:(nr - 1)) + nr * (ii - 1)
            idx2 = @. (1:(nr - 1)) + nr * (ii)
            id_el[idx, :] .= [idx1 idx2 idx2 .+ 1 idx1 .+ 1]
        else
            idx = @. (1:(nr - 1)) + (nr - 1) * (ii - 1)
            idx1 = @. (1:(nr - 1)) + nr * (ii - 1)
            idx2 = @. 1:(nr - 1)
            id_el[idx, :] .= [idx1 idx2 idx2 .+ 1 idx1 .+ 1]
        end
    end

    # -- Split quadrilateral in 2 triangles
    ntri = nels * 2
    id_els_tri = Array{Int64,2}(undef, ntri, 3)
    i1 = [1, 2, 4]
    i2 = [2, 3, 4]
    @inbounds for ii in 1:nels
        # -- connectivity matrix
        idx = @. (1:2) + 2 * (ii - 1)
        id_els_tri[idx[1], :] .= id_el[ii, i1]
        id_els_tri[idx[2], :] .= id_el[ii, i2]
    end

    EL2NOD = Array(id_els_tri')
    neighbours = element_neighbours(EL2NOD)
    θ, r, EL2NOD = add_midpoints(EL2NOD, neighbours, θ, r)
    EL2NOD_P1 = p2top1(EL2NOD)
    neighbours_p1 = element_neighbours(EL2NOD_P1)
    nel = size(EL2NOD, 2)
    nVnod = maximum(view(EL2NOD, 1:3, :))
    # EL2NOD, θ, r =  sixth_node(EL2NOD, θ, r)
    x, z = polar2cartesian(θ, r)
    nnod = length(x)
    # EL2NODP = Matrix(reshape(1:3*nel,3,nel))
    EL2NODP = EL2NOD[1:3, :]

    nvert = 3

    return Grid(
        x,
        z,
        θ,
        r,
        EL2NOD,
        EL2NOD_P1,
        EL2NODP,
        nθ,
        nr,
        nel,
        nnod,
        nVnod,
        nvert,
        neighbours_p1,
    )
end

function element_area(gr::Grid)
    xel = view(gr.x, gr.e2n)
    zel = view(gr.z, gr.e2n)
    @. @views 0.5 * (
        xel[1, :] * (zel[2, :] - zel[3, :]) +
        xel[2, :] * (zel[3, :] - zel[1, :]) +
        xel[3, :] * (zel[1, :] - zel[2, :])
    )
end

function element_area(x, z, e2n)
    xel = view(x, e2n)
    zel = view(z, e2n)
    @. @views 0.5 * (
        xel[1, :] * (zel[2, :] - zel[3, :]) +
        xel[2, :] * (zel[3, :] - zel[1, :]) +
        xel[3, :] * (zel[1, :] - zel[2, :])
    )
end

function inradius(gr::Grid)
    # https://en.wikibooks.org/wiki/Trigonometry/Circles_and_Triangles/The_Incircle
    x, z, nel, e2n = gr.x, gr.z, gr.nel, gr.e2n_p1
    r = Vector{Float64}(undef, nel)
    @inbounds @fastmath for iel in 1:nel
        xv = ntuple(i -> x[e2n[i, iel]], 3) # verteices x-coords
        zv = ntuple(i -> z[e2n[i, iel]], 3) # verteices z-coords
        a = distance(xv[1], zv[1], xv[2], zv[2]) # distance of side AB
        b = distance(xv[1], zv[1], xv[3], zv[3]) # distance of side AC
        c = distance(xv[3], zv[3], xv[2], zv[2]) # distance of side BC
        s = (a + b + c) * 0.5 # semiperimeter
        r[iel] = √((s - a) * (s - b) * (s - c) / s) # inradius
    end
    return minimum(r)
end

function inrectangle(gr::Grid)
    # define a percentage of the diagonal
    x, z, nel, e2n = gr.x, gr.z, gr.nel * 4, gr.e2n_p1
    # x, z, nel, e2n = gr.x, gr.z, gr.nel, gr.e2n
    r = Vector{Float64}(undef, nel)
    @inbounds @fastmath for iel in 1:nel
        xv = ntuple(i -> x[e2n[i, iel]], 3) # verteices x-coords
        zv = ntuple(i -> z[e2n[i, iel]], 3) # verteices z-coords
        xmin, xmax = extrema(xv)
        zmin, zmax = extrema(zv)
        r[iel] = √((xmin - xmax)^2 + (zmin - zmax)^2) # inradius
    end

    return minimum(r)
end

function sixth_node(EL2NOD, θ, r)
    nel = size(EL2NOD, 2)
    nnod = maximum(EL2NOD)
    θel = θ[EL2NOD[1:3, :]]
    rel = r[EL2NOD[1:3, :]]
    fixangles!(θel)
    θ_new = mean(θel; dims=1)
    r_new = mean(rel; dims=1)
    θ = vcat(θ, vec(θ_new))
    r = vcat(r, vec(r_new))
    return vcat(EL2NOD, transpose(nnod .+ (1:nel))), θ, r
end

function p2top1(EL2NOD)
    nel = size(EL2NOD, 2)
    EL2NOD_P1 = Array{Int64,2}(undef, 3, nel * 4)
    local_map = [1 4 6; 4 2 5; 6 5 3; 4 5 6]'
    @inbounds for iparent in 1:nel
        local_element = view(EL2NOD, :, iparent)
        child_elements = view(local_element, local_map)
        iglobal = (1:4) .+ 4 * (iparent - 1)
        EL2NOD_P1[:, iglobal] .= child_elements
    end
    return EL2NOD_P1
end

function element_neighbours(e2n)
    els = size(e2n, 1) == 3 ? e2n : view(e2n, 1:3, :)
    nel = size(els, 2)

    # Incidence matrix
    I, J, V = Int[], Int[], Bool[]
    @inbounds for i in axes(els, 1), j in axes(els, 2)
        node = els[i, j]
        push!(I, node)
        push!(J, j)
        push!(V, true)
    end
    incidence_matrix = sparse(J, I, V)

    # Find neighbouring elements
    neighbour = [Int64[] for _ in 1:nel]
    nnod = I[end]
    @inbounds for node in 1:nnod
        # Get elements sharing node
        r = nzrange(incidence_matrix, node)
        el_neighbour = incidence_matrix.rowval[r]
        # Add neighbouring elements to neighbours map
        for iel1 in el_neighbour
            current_neighbours = neighbour[iel1]
            for iel2 in el_neighbour
                # check for non-self neighbour and avoid repetitions
                if (iel1 != iel2) && (iel2 ∉ current_neighbours)
                    push!(neighbour[iel1], iel2)
                end
            end
        end
    end

    return neighbour
end

function edge_connectivity(EL2NOD, neighbours)
    nel = size(EL2NOD, 2)
    local_map = [1 2 3; 2 3 1]
    global_idx = 0
    el2edge = fill(0, 3, nel)
    edge2node = Vector{Int64}[]
    edge_neighbour = Array{Int64,2}(undef, 2, 3)

    @inbounds for iel in 1:nel
        # local edge
        edge = EL2NOD[local_map, iel]
        sort!(edge; dims=1)

        # neighbours of local element
        el_neighbours = neighbours[iel]
        edge_neighbours = [EL2NOD[local_map, i] for i in el_neighbours]

        # check edge by edge
        for iedge in 1:3
            if el2edge[iedge, iel] == 0
                global_idx += 1
                el2edge[iedge, iel] = global_idx
                push!(edge2node, view(edge, :, iedge))

                # edges the neighbours
                for (c, ieln) in enumerate(el_neighbours)
                    edge_neighbour .= edge_neighbours[c]
                    sort!(edge_neighbour; dims=1)

                    # check wether local edge is in neighbouring element
                    for i in 1:3
                        if (edge[1, iedge] == edge_neighbour[1, i]) &&
                            (edge[2, iedge] ∈ edge_neighbour[2, i])
                            el2edge[i, ieln] = global_idx
                            break
                        end
                    end
                end
            end
        end
    end

    return el2edge, edge2node
end

function add_midpoints(EL2NOD, neighbours, θ, r)
    el2edge, edges2node = edge_connectivity(EL2NOD, neighbours)
    # make mid points
    nnods = size(edges2node, 1)
    θmid = Vector{Float64}(undef, nnods)
    rmid = similar(θmid)
    @inbounds for i in 1:nnods
        θbar = @MVector [θ[edges2node[i][1]], θ[edges2node[i][2]]]
        if abs.(θbar[1] - θbar[2]) > π
            for j in 1:2
                if θbar[j] < π
                    θbar[j] += 2π
                end
            end
        end
        θmid[i] = 0.5 * (θbar[1] + θbar[2])
        rmid[i] = 0.5 * (r[edges2node[i][1]] + r[edges2node[i][2]])
    end

    old_nnods = maximum(EL2NOD)
    EL2NOD_new = el2edge .+ old_nnods

    return vcat(θ, θmid), vcat(r, rmid), vcat(EL2NOD, EL2NOD_new)
end

function point_ids(M::Grid)
    top = :outter
    bot = :inner
    inner = :inside

    nnod = length(M.r)
    IDs = Vector{Symbol}(undef, nnod)

    # rmin, rmax = extrema(M.r)
    rmin, rmax = extrema(M.r)

    @inbounds for (i, ri) in enumerate(M.r)
        if ri == rmax
            IDs[i] = top
        elseif ri == rmin
            IDs[i] = bot
        else
            IDs[i] = inner
        end
    end

    return IDs
end
