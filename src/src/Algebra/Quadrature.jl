struct ScratchThermalGlobal{T}
    ndim::T # model dimension
    nvert::T # number of vertices per element
    nnodel::T # number of nodes per element
    nel::T # number of elements
    nip::T # number of vertices per element
end

struct ShapeFunctionsThermal{T}
    N::Vector{SMatrix{1,6,T,6}}
    ∇N::Vector{SMatrix{2,6,T,12}}
    dN3ds::SMatrix{2,3,T,6}
    w_ip::SVector{7,T}
    N3::Vector{SMatrix{1,3,T,3}}
end

struct ShapeFunctionsStokes{T}
    N::Vector{SMatrix{1,6,T,6}}
    ∇N::Vector{SMatrix{2,6,T,12}}
    NP::Vector{SMatrix{1,3,Float64,3}}
    dN3ds::SMatrix{2,3,T,6}
    w_ip::SVector{7,T}
    N3::Vector{SMatrix{1,3,T,3}}
end

struct ShapeFunctionsStress{A,B,C,D,E,F}
    N::A
    ∇N::B
    NP::C
    dN3ds::D
    w_ip::E
    N3::F
end

@inline function _get_SF(nip, nnodel)
    x_ip, w_ip = ip_triangle(nip)
    # local coordinates and weights of points for integration of
    # velocity/pressure matrices
    N, dNds = shape_functions_triangles(x_ip, nnodel)
    # velocity shape functions and their derivatives
    dN3ds = @SMatrix [
        -1.0 1.0 0.0 # w.r.t. r
        -1.0 0.0 1.0
    ]
    # derivatives of linear (3-node) shape functions; used to calculate
    # each element's Jacobian  

    return N, dNds, dN3ds, w_ip
end

function ip_triangle(nip)
    if nip === 3
        ipx, ipw = ip_triangle3()
    elseif nip === 6
        ipx, ipw = ip_triangle6()
    elseif nip === 7
        ipx, ipw = ip_triangle7()
    end
    return ipx, ipw
end

ip_triangle(::Val{3}) = ip_triangle3()
ip_triangle(::Val{6}) = ip_triangle6()
ip_triangle(::Val{7}) = ip_triangle7()

function ip_triangle3()
    # -- Integration point coordinates
    ipx = @SMatrix [
        1/6 1/6
        2/3 1/6
        1/6 2/3
    ]

    # -- Weights
    ipw = @SVector [1 / 6, 1 / 6, 1 / 6]

    return ipx, ipw
end

function ip_triangle6()

    # -- Integration point coordinates
    g1 = (8.0 - sqrt(10.0) + sqrt(38.0 - 44.0 * sqrt(2.0 / 5.0))) / 18.0
    g2 = (8.0 - sqrt(10.0) - sqrt(38.0 - 44.0 * sqrt(2.0 / 5.0))) / 18.0
    ipx = @SMatrix [
        1.0-2.0 * g1 g1
        g1 1.0-2.0 * g1
        g1 g1
        1.0-2.0 * g2 g2
        g2 1.0-2.0 * g2
        g2 g2
    ]

    # -- Weights
    w1 = (620.0 + sqrt(213125.0 - 53320.0 * sqrt(10.0))) / 3720.0
    w2 = (620.0 - sqrt(213125.0 - 53320.0 * sqrt(10.0))) / 3720.0
    ipw = @SVector [
        w1,                       #0.223381589678011;
        w1,                       #0.223381589678011;
        w1,                       #0.223381589678011;
        w2,                       #0.109951743655322;
        w2,                       #0.109951743655322;
        w2,                       #0.109951743655322;
    ]

    return ipx, 0.5 * ipw
end

function ip_triangle7()

    # -- Integration point coordinates
    g1 = (6.0 - sqrt(15.0)) / 21.0
    g2 = (6.0 + sqrt(15.0)) / 21.0
    ipx = @SMatrix [
        1.0/3.0     1.0/3.0
        1.0-2.0*g1  g1
        g1          1.0-2.0 * g1
        g1          g1
        1.0-2.0*g2  g2
        g2          1.0-2.0 * g2
        g2          g2
    ]

    # -- Weights
    w1 = (155.0 - sqrt(15.0)) / 1200.0
    w2 = (155.0 + sqrt(15.0)) / 1200.0
    ipw = @SVector [
        0.225,
        w1,                       #0.223381589678011;
        w1,                       #0.223381589678011;
        w1,                       #0.223381589678011;
        w2,                       #0.109951743655322;
        w2,                       #0.109951743655322;
        w2,                       #0.109951743655322;
    ]

    return ipx, 0.5 * ipw
end

function shape_functions_triangles(lc, nnodel)
    r = lc[:, 1]
    s = lc[:, 2]
    npt = length(r)
    N, dN = get_N_∇N(r, s, npt, nnodel)
    return N, dN
end

function get_N_∇N(r, s, npt, ::Val{3})
    N = [sf_N_tri3(r[ip], s[ip]) for ip in 1:npt]
    dN = [sf_dN_tri3(r[ip], s[ip]) for ip in 1:npt]
    return N, dN
end

function get_N_∇N(r, s, npt, ::Val{6})
    N = [sf_N_tri6(r[ip], s[ip]) for ip in 1:npt]
    dN = [sf_dN_tri6(r[ip], s[ip]) for ip in 1:npt]
    return N, dN
end

function get_N_∇N(r, s, npt, ::Val{7})
    N = [sf_N_tri7(r[ip], s[ip]) for ip in 1:npt]
    dN = [sf_dN_tri7(r[ip], s[ip]) for ip in 1:npt]
    return N, dN
end

function sf_N_tri3(r, s)
    # Find shape functions and their derivatives at given points on the
    # master element for 3 node triangle
    # 3-node triangle (node numbering is important)
    #
    #        3
    #        | \
    # s-axis |   \
    #        |     \
    #        1 - - - 2
    #          r axis -->
    t = 1.0 - r - s
    N = @SMatrix [t r s]   # N3 at coordinate (r,s)
    return N
end

function sf_dN_tri3(r, s)
    # Find shape functions and their derivatives at given points on the
    # master element for 3 node triangle
    # 3-node triangle (node numbering is important)
    #
    #        3
    #        | \
    # s-axis |   \
    #        |     \
    #        1 - - - 2
    #          r axis -->
    t = 1.0 - r - s
    dN = @SMatrix [
        -1.0 1.0 0.0 # w.r.t. r
        -1.0 0.0 1.0
    ] # w.r.t. s
    return dN
end

function sf_N_tri7(r, s)
    # Find shape functions and their derivatives at given points on the
    # master element for a 7 node triangle
    # 7-node triangle (node numbering is important)
    #
    #        3
    #        | \
    # s-axis 6   5
    #        | 7   \
    #        1 - 4 - 2
    #          r-axis
    t = 1 - r - s
    N = @SMatrix [t * (2 * t - 1) + 3 * r * s * t r * (2 * r - 1) + 3 * r * s * t s * (
        2 * s - 1
    ) + 3 *
                                                                                  r *
                                                                                  s *
                                                                                  t 4 *
                                                                                    r *
                                                                                    t -
                                                                                    12 *
                                                                                    r *
                                                                                    s *
                                                                                    t 4 *
                                                                                      r *
                                                                                      s -
                                                                                      12 *
                                                                                      r *
                                                                                      s *
                                                                                      t 4 *
                                                                                        s *
                                                                                        t -
                                                                                        12 *
                                                                                        r *
                                                                                        s *
                                                                                        t 27 *
                                                                                          r *
                                                                                          s *
                                                                                          t]

    return N
end

function sf_dN_tri7(r, s)
    # Find shape functions and their derivatives at given points on the
    # master element for a 7 node triangle
    # 7-node triangle (node numbering is important)
    #
    #        3
    #        | \
    # s-axis 6   5
    #        | 7   \
    #        1 - 4 - 2
    #          r-axis
    t = 1 - r - s
    dN = @SMatrix [
        1 - 4 * t + 3 * s * t-3 * r * s -1 + 4 * r + 3 * s * t-3 * r * s 3 * s * t-3 * r * s 4 * t - 4 * r + 12 * r * s-12 * s * t 4 * s + 12 * r * s-12 * s * t -4 * s + 12 * r * s-12 * s * t -27 * r * s+27 * s * t
        1 - 4 * t + 3 * r * t-3 * r * s 3 * r * t-3 * r * s -1 + 4 * s + 3 * r * t-3 * r * s -4 * r - 12 * r * t+12 * r * s 4 * r - 12 * r * t+12 * r * s 4 * t - 4 * s - 12 * r * t+12 * r * s 27 * r * t-27 * r * s
    ]

    return dN
end

function sf_N_tri6(r, s)
    # Find shape functions and their derivatives at given points on the
    # master element for a 6 node triangle (node numbering is important)
    #
    #        3
    #        | \
    # s-axis 6   5
    #        |     \
    #        1 - 4 - 2
    #          r-axis
    #
    t = 1.0 - r - s
    # N1 at coordinate (r,s), N2 at coordinate (r,s), etc
    N = @SMatrix [t * (2.0 * t - 1.0) r * (2.0 * r - 1.0) s * (2.0 * s - 1.0) 4.0 * r * t 4.0 *
                                                                                          r *
                                                                                          s 4.0 *
                                                                                            s *
                                                                                            t]
    #     dN1       dN2    dN3    dN4       dN5    dN6

    return N
end

function sf_dN_tri6(r, s)
    # Find shape functions and their derivatives at given points on the
    # master element for a 6 node triangle (node numbering is important)
    #
    #        3
    #        | \
    # s-axis 6   5
    #        |     \
    #        1 - 4 - 2
    #          r-axis
    #
    t = 1.0 - r - s
    # N1 at coordinate (r,s), N2 at coordinate (r,s), etc
    #     dN1       dN2    dN3    dN4       dN5    dN6
    dN = @SMatrix [
        -(4.0 * t - 1.0) 4.0 * r-1 0.0 4.0*(t - r) 4.0*s -4.0*s     # w.r.t. r
        -(4.0 * t - 1.0) 0.0 4.0 * s-1.0 -4.0*r 4.0*r 4.0*(t - s)
    ] # w.r.t. s

    return dN
end
