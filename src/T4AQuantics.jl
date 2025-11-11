module T4AQuantics

using ITensors
import ITensors
using ITensors.SiteTypes: siteinds as ITensorsSiteTypes_siteinds
import ITensors.NDTensors: Tensor, BlockSparseTensor, blockview
import T4AITensorCompat
using T4AITensorCompat: TensorTrain
using T4AITensorCompat: findsite, linkinds, linkind, findsites, truncate!, siteinds, truncate

import SparseIR: Fermionic, Bosonic, Statistics
import LinearAlgebra: I, inv
using StaticArrays
import SparseArrays: sparse

import QuanticsTCI
import TensorCrossInterpolation as TCI
import T4APartitionedMPSs
using T4AITensorCompat: contract, Algorithm, apply, product

using EllipsisNotation

function __init__()
end

include("util.jl")
include("tag.jl")
include("binaryop.jl")
include("mul.jl")
include("mps.jl")
include("fouriertransform.jl")
include("imaginarytime.jl")
include("transformer.jl")
include("affine.jl")
include("partitionedmps.jl")

end
