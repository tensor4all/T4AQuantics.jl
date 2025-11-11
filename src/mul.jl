abstract type AbstractMultiplier end

#===
Matrix multiplication
===#
struct MatrixMultiplier{T} <: AbstractMultiplier where {T}
    sites_row::Vector{Index{T}}
    sites_shared::Vector{Index{T}}
    sites_col::Vector{Index{T}}

    function MatrixMultiplier(sites_row::Vector{Index{T}},
            sites_shared::Vector{Index{T}},
            sites_col::Vector{Index{T}}) where {T}
        new{T}(sites_row, sites_shared, sites_col)
    end
end

function MatrixMultiplier(site_row::Index{T},
        site_shared::Index{T},
        site_col::Index{T}) where {T}
    return MatrixMultiplier([site_row], [site_shared], [site_col])
end

function preprocess(mul::MatrixMultiplier{T}, M1::TensorTrain, M2::TensorTrain) where {T}
    for (site_row, site_shared, site_col) in zip(mul.sites_row, mul.sites_shared,
        mul.sites_col)
        M1, M2 = combinesites(M1, site_row, site_shared),
        combinesites(M2, site_col, site_shared)
    end
    return M1, M2
end

function postprocess(mul::MatrixMultiplier{T}, M::TensorTrain)::TensorTrain where {T}
    tensors = collect(M)
    for (site_row, site_col) in zip(mul.sites_row, mul.sites_col)
        p = findfirst(hasind(site_row), tensors)
        hasind(tensors[p], site_col) ||
            error("$site_row and $site_col are not on the same site")

        # Recompute links after each modification
        links = length(tensors) > 1 ? [commoninds(tensors[n], tensors[n + 1])[1] for n in 1:(length(tensors) - 1)] : Index[]

        indsl = [site_row]
        if p > 1 && (p - 1) <= length(links)
            push!(indsl, links[p - 1])
        end

        indsr = [site_col]
        if p < length(tensors) && p <= length(links)
            push!(indsr, links[p])
        end

        Ml, Mr = split_tensor(tensors[p], [indsl, indsr])

        deleteat!(tensors, p)
        insert!(tensors, p, Ml)
        insert!(tensors, p + 1, Mr)
    end

    return TensorTrain(Vector{ITensor}(tensors))
end

#===
Elementwise multiplication
===#
struct ElementwiseMultiplier{T} <: AbstractMultiplier where {T}
    sites::Vector{Index{T}}
    function ElementwiseMultiplier(sites::Vector{Index{T}}) where {T}
        new{T}(sites)
    end
end

"""
Convert an MPS tensor to an MPO tensor with a diagonal structure
"""
function _asdiagonal(t, site::Index{T}; baseplev=0)::ITensor where {T<:Number}
    ITensors.hasinds(t, site') && error("Found $(site')")
    links = ITensors.uniqueinds(ITensors.inds(t), site)
    rawdata = Array(t, links..., site)
    tensor = zeros(eltype(t), size(rawdata)..., ITensors.dim(site))
    for i in 1:ITensors.dim(site)
        tensor[.., i, i] = rawdata[.., i]
    end
    return ITensor(
        tensor, links..., ITensors.prime(site, baseplev + 1), ITensors.prime(
            site, baseplev)
    )
end

function _todense(t, site::Index{T}) where {T<:Number}
    links = uniqueinds(inds(t), site, site'')
    newdata = zeros(eltype(t), dim.(links)..., dim(site))
    if length(links) == 2
        olddata = Array(t, links..., site, site'')
        for i in 1:dim(site)
            newdata[:, :, i] = olddata[:, :, i, i]
        end
    elseif length(links) == 1
        olddata = Array(t, links..., site, site'')
        for i in 1:dim(site)
            newdata[:, i] = olddata[:, i, i]
        end
    else
        error("Too many links found: $links")
    end
    return ITensor(newdata, links..., site)
end

function preprocess(mul::ElementwiseMultiplier{T}, M1::TensorTrain, M2::TensorTrain) where {T}
    tensors1 = collect(M1)
    tensors2 = collect(M2)
    for s in mul.sites
        p = findfirst(hasind(s), tensors1)
        hasinds(tensors2[p], s) || error("ITensor of M2 at $p does not have $s")
        #tensors1[p] = replaceprime(_asdiagonal(tensors1[p], s), 0 => 1, 1 => 2)
        tensors1[p] = _asdiagonal(tensors1[p], s)
        replaceind!(tensors1[p], s' => s'')
        replaceind!(tensors1[p], s => s')
        tensors2[p] = _asdiagonal(tensors2[p], s)
    end
    return TensorTrain(Vector{ITensor}(tensors1)), TensorTrain(Vector{ITensor}(tensors2))
end

function postprocess(mul::ElementwiseMultiplier{T}, M::TensorTrain)::TensorTrain where {T}
    tensors = collect(M)
    for s in mul.sites
        p = findfirst(hasind(s), tensors)
        tensors[p] = _todense(tensors[p], s)
    end
    return TensorTrain(Vector{ITensor}(tensors))
end

"""
By default, elementwise multiplication will be performed.
"""
function automul(M1::TensorTrain, M2::TensorTrain; tag_row::String="", tag_shared::String="",
        tag_col::String="", alg="naive", cutoff=1e-30, kwargs...)
    if in(:maxbonddim, keys(kwargs))
        error("Illegal keyward parameter: maxbonddim. Use maxdim instead!")
    end

    sites1_flat = collect(Iterators.flatten(siteinds(M1)))
    sites2_flat = collect(Iterators.flatten(siteinds(M2)))
    sites1_vec = Vector{Index{Int}}(sites1_flat)
    sites2_vec = Vector{Index{Int}}(sites2_flat)
    
    sites_row = findallsiteinds_by_tag(sites1_vec; tag=tag_row)
    sites_shared = findallsiteinds_by_tag(sites1_vec; tag=tag_shared)
    sites_col = findallsiteinds_by_tag(sites2_vec; tag=tag_col)
    sites_matmul = Set(Iterators.flatten([sites_row, sites_shared, sites_col]))

    if sites_shared != findallsiteinds_by_tag(sites2_vec; tag=tag_shared)
        error("Invalid shared sites for MatrixMultiplier")
    end

    matmul = MatrixMultiplier(sites_row, sites_shared, sites_col)
    ewmul = ElementwiseMultiplier([s for s in sites1_vec if s ∉ sites_matmul])

    M1_, M2_ = preprocess(matmul, M1, M2)
    M1_, M2_ = preprocess(ewmul, M1_, M2_)

    # Convert alg to Algorithm type
    alg_ = alg isa Algorithm ? alg : Algorithm(alg)
    M = contract(M1_, M2_; alg=alg_, cutoff=cutoff, kwargs...)

    M = T4AQuantics.postprocess(matmul, M)
    M = T4AQuantics.postprocess(ewmul, M)

    if in(:maxdim, keys(kwargs))
        truncate!(M; maxdim=kwargs[:maxdim], cutoff=cutoff)
    end

    return M
end
