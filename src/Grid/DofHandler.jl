struct DoFHandler{N,T}
    DoF::Vector{NTuple{N,T}}
    nDofs::T
end

Base.getindex(a::DoFHandler, I::Int64) = a.DoF[I]
Base.getindex(a::DoFHandler, I::Int32) = a.DoF[I]
Base.getindex(a::DoFHandler, I::Int8) = a.DoF[I]

function DoFs_Thermal(EL2NOD, nnodel)
    DoF = [ntuple(j -> EL2NOD[j, i], nnodel) for i in axes(EL2NOD, 2)]
    nDofs = maximum(view(EL2NOD, 1:nnodel, :))
    return DoFHandler(DoF, nDofs)
end

function DoFs_Pressure(e2nP)
    DoF = [ntuple(j -> e2nP[j, i], 3) for i in axes(e2nP, 2)]
    nDofs = maximum(e2nP)
    return DoFHandler(DoF, nDofs)
end

function DoFs_Velocity(EL2NOD)
    dummy = Matrix{Int64}(undef, size(EL2NOD, 2), 12)
    @views dummy[:, 1:2:end] .= (@. 2 * (EL2NOD - 1) + 1)'
    @views dummy[:, 2:2:end] .= (@. 2 * (EL2NOD))'

    DoF = [ntuple(j -> dummy[i, j], 12) for i in axes(EL2NOD, 2)]
    nDofs = maximum(EL2NOD) * 2
    return DoFHandler(DoF, nDofs)
end
