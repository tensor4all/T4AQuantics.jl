using ITensors
import T4AITensorCompat: random_mps, TensorTrain
using Random

function _random_mpo(
        rng::AbstractRNG, sites::AbstractVector{<:AbstractVector{Index{T}}}; linkdims::Int=1
) where {T}
    sites_ = collect(Iterators.flatten(sites))
    Ψ = random_mps(rng, sites_; linkdims)
    tensors = ITensor[]
    pos = 1
    for i in 1:length(sites)
        push!(tensors, prod(Ψ[pos:(pos + length(sites[i]) - 1)]))
        pos += length(sites[i])
    end
    return TensorTrain(tensors)
end

function _random_mpo(
        sites::AbstractVector{<:AbstractVector{Index{T}}}; linkdims::Int=1
) where {T}
    return _random_mpo(Random.default_rng(), sites; linkdims)
end
