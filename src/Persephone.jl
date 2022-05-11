module Persephone

using StaticArrays
using LinearAlgebra
using SuiteSparse
using SparseArrays
using LoopVectorization
using IterativeSolvers
using Preconditioners
using HDF5
using TimerOutputs
using Polyester
# using Pardiso
using Fabrics
using MuladdMacro
using Krylov
using ThreadedSparseArrays
using LimitedLDLFactorizations

import Statistics: mean
import SpecialFunctions: erf, erfc
import .Base: *

const yr = 3600 * 365.25 * 24
export yr

include("Grid/mesher.jl")
export Grid_split1
export Grid
export point_ids
export inradius
export inrectangle
export cartesian2polar
export polar2cartesian
export Cartesian
export Polar
export Point2D
export ElementCoordinates
export init_grid

include("Grid/coloring.jl")
export color_mesh

include("Grid/DofHandler.jl")
export DoFHandler

include("Rheology/Rheology.jl")
export Isotropic
export Anisotropic
export Isoviscous
export TemperatureDependant
export TemperatureDependantPlastic
export Nakagawa
export getviscosity
export getviscosity!
export state_equation
export state_equation!

include("Stokes/StokesSolver.jl")
export solveStokes
export initvelocity
export rotation_matrix
export assembly_stokes_cylindric

include("ThermalDiffusion/ThermalSolver.jl")
export _prepare_matrices!
export _apply_bcs!

include("Algebra/Quadrature.jl")
export ScratchThermalGlobal
export ShapeFunctionsThermal
export ShapeFunctionsStokes
export _get_SF

include("Stress/FSE.jl")
export FiniteStrainEllipsoid
export volume_integral
export getFSE
export getFSE_healing
export getFSE_annealing!
export rebuild_FSE
export isotropic_lithosphere!
export healing
export FiniteStrain
export Annealing
export Healing
export NoHealing
export AnnealingOnly
export IsotropicDomainOnly
export AnnealingIsotropicDomain
export FabricDestructionOnly
export AnnealingFabricDestruction

include("IterativeSolvers.jl")

include("PardisoSolvers.jl")

include("Utilities.jl")
export elementcoordinate
export fixangles!
export fixangles6!
export getspeed
export calculate_Δt
export getips

include("Algebra/Sparsity.jl")
export Stokes
export Thermal
export Sparsity
export sparsitystokes
export sparsitythermal

include("Algebra/Algebra.jl")
export gemm
export gemmt
export gemm!
export gemmt!
export mydot
export mynorm

include("IO/IOmanager.jl")
export IOs
export write_stats
export reloader
export Nusselt
export ScratchNusselt
export savedata
export setup_output
export setup_metrics

include("Setup/Setup.jl")
export ThermalParameters
export thermal_parameters
export init_temperature
export init_particle_temperature!
export fixT!
export init_U_P

include("Rheology/anisotropy.jl")
export StiffnessTensor
export Parameterisation
export 𝓒init
export DEM
export getDEM
export anisotropic_tensor

include("PiC/tsearch.jl")
export PINFO
export PWEIGHTS
export PVAR
export init_pinfo
export init_pweights
export init_pvars
export purgeparticles
export check_corruption!
export addreject
export particles_generator
export tsearch_parallel

include("PiC/PiC.jl")
export Fij2particle
export F2particle
export ip2node
export velocities2ip
export getvelocity
export initial_particle_temperature!
export interpolate_temperature!
export T2node
export applybounds!
export applybounds
export F2ip
export T2particle

include("Advection/Advection.jl")
export advection_RK2
export advection_RK4
export advection!

include("ThermalDiffusion/AssemblerThermal.jl")
export solveDiffusion_threaded
export diffusion_immutables
export assembly_threaded!

include("Stokes/AssemblerStokes.jl")
export solve_stokes_threaded
export stokes_immutables
export stress_shape_functions

include("Setup/boundary_conditions.jl")
export BC
export init_BCs
export velocity_bcs
export temperature_bcs

include("PiC/CubicInterpolations.jl")
export quasicubic_interpolation

include("Stress/StressCalculation.jl")
export SymmetricTensor
export Gradient
export initstress
export secondinvariant
export secondinvariant!
export stress!
export deformation_gradient!
export stress

end # module
