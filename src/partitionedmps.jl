# Integration with T4APartitionedMPSs
# This file contains methods for T4APartitionedMPSs types (SubDomainMPS, PartitionedMPS)

using ITensors
import ITensors
using ITensors.SiteTypes: siteinds
import ITensors.NDTensors: Tensor, BlockSparseTensor, blockview
using T4AITensorCompat: TensorTrain
using T4AITensorCompat: findsite, linkinds, linkind, findsites

import T4APartitionedMPSs: T4APartitionedMPSs, SubDomainMPS, PartitionedMPS, isprojectedat,
                        project

function makesitediagonal(subdmps::SubDomainMPS, site::Index)
    return _makesitediagonal(subdmps, site; baseplev=0)
end

function makesitediagonal(subdmps::SubDomainMPS, sites::AbstractVector{Index})
    return _makesitediagonal(subdmps, sites; baseplev=0)
end

function makesitediagonal(subdmps::SubDomainMPS, tag::String)
    mps_diagonal = makesitediagonal(TensorTrain(subdmps), tag)
    subdmps_diagonal = SubDomainMPS(mps_diagonal)

    target_sites = findallsiteinds_by_tag(
        unique(ITensors.noprime.(Iterators.flatten(siteinds(subdmps)))); tag=tag
    )

    newproj = deepcopy(subdmps.projector)
    for s in target_sites
        if isprojectedat(subdmps.projector, s)
            newproj[ITensors.prime(s)] = newproj[s]
        end
    end

    return project(subdmps_diagonal, newproj)
end

function _makesitediagonal(
        subdmps::SubDomainMPS, sites::AbstractVector{Index{IndsT}}; baseplev=0
) where {IndsT}
    M_ = deepcopy(TensorTrain(subdmps))
    for site in sites
        target_site::Int = only(findsites(M_, site))
        M_[target_site] = _asdiagonal(M_[target_site], site; baseplev=baseplev)
    end
    return project(M_, subdmps.projector)
end

function _makesitediagonal(subdmps::SubDomainMPS, site::Index; baseplev=0)
    return _makesitediagonal(subdmps, [site]; baseplev=baseplev)
end

function extractdiagonal(
        subdmps::SubDomainMPS, sites::AbstractVector{Index{IndsT}}
) where {IndsT}
    tensors = collect(subdmps.data)
    for i in eachindex(tensors)
        for site in intersect(sites, ITensors.inds(tensors[i]))
            sitewithallplevs = _find_site_allplevs(tensors[i], site)
            tensors[i] = if length(sitewithallplevs) > 1
                tensors[i] = _extract_diagonal(tensors[i], sitewithallplevs...)
            else
                tensors[i]
            end
        end
    end

    projector = deepcopy(subdmps.projector)
    for site in sites
        if site' in keys(projector.data)
            delete!(projector.data, site')
        end
    end
    return SubDomainMPS(TensorTrain(Vector{ITensor}(tensors)), projector)
end

function extractdiagonal(subdmps::SubDomainMPS, site::Index{IndsT}) where {IndsT}
    return extractdiagonal(subdmps, [site])
end

function extractdiagonal(subdmps::SubDomainMPS, tag::String)::SubDomainMPS
    targetsites = findallsiteinds_by_tag(
        unique(ITensors.noprime.(T4APartitionedMPSs._allsites(subdmps))); tag=tag
    )
    return extractdiagonal(subdmps, targetsites)
end

function rearrange_siteinds(subdmps::SubDomainMPS, sites)
    return T4APartitionedMPSs.rearrange_siteinds(subdmps, sites)
end

function rearrange_siteinds(partmps::PartitionedMPS, sites)
    return T4APartitionedMPSs.rearrange_siteinds(partmps, sites)
end

"""
Make the PartitionedMPS diagonal for a given site index `s` by introducing a dummy index `s'`.
"""
function makesitediagonal(obj::PartitionedMPS, site)
    return PartitionedMPS([_makesitediagonal(prjmps, site; baseplev=0)
                           for prjmps in values(obj)])
end

function _makesitediagonal(obj::PartitionedMPS, site; baseplev=0)
    return PartitionedMPS([_makesitediagonal(prjmps, site; baseplev=baseplev)
                           for prjmps in values(obj)])
end

"""
Extract diagonal of the PartitionedMPS for `s`, `s'`, ... for a given site index `s`,
where `s` must have a prime level of 0.
"""
function extractdiagonal(obj::PartitionedMPS, site)
    return PartitionedMPS([extractdiagonal(prjmps, site) for prjmps in values(obj)])
end

"""
By default, elementwise multiplication will be performed.
"""
function automul(
        M1::PartitionedMPS,
        M2::PartitionedMPS;
        tag_row::String="",
        tag_shared::String="",
        tag_col::String="",
        alg="naive",
        maxdim=typemax(Int),
        cutoff=1e-25,
        kwargs...
)
    all(length.(siteinds(M1)) .== 1) || error("M1 should have only 1 site index per site")
    all(length.(siteinds(M2)) .== 1) || error("M2 should have only 1 site index per site")

    sites_row = _findallsiteinds_by_tag(M1; tag=tag_row)
    sites_shared = _findallsiteinds_by_tag(M1; tag=tag_shared)
    sites_col = _findallsiteinds_by_tag(M2; tag=tag_col)
    sites_matmul = Set(Iterators.flatten([sites_row, sites_shared, sites_col]))

    sites1 = only.(siteinds(M1))
    sites1_ewmul = setdiff(only.(siteinds(M1)), sites_matmul)
    sites2_ewmul = setdiff(only.(siteinds(M2)), sites_matmul)
    sites2_ewmul == sites1_ewmul || error("Invalid sites for elementwise multiplication")

    M1 = _makesitediagonal(M1, sites1_ewmul; baseplev=1)
    M2 = _makesitediagonal(M2, sites2_ewmul; baseplev=0)

    sites_M1_diag = Vector{Vector{Index{Int}}}([collect(x) for x in siteinds(M1)])
    sites_M2_diag = Vector{Vector{Index{Int}}}([collect(x) for x in siteinds(M2)])

    M1 = rearrange_siteinds(
        M1, combinesites(sites_M1_diag, Vector{Index{Int}}(sites_row), Vector{Index{Int}}(sites_shared)))

    M2 = rearrange_siteinds(
        M2, combinesites(sites_M2_diag, Vector{Index{Int}}(sites_shared), Vector{Index{Int}}(sites_col)))

    M = T4APartitionedMPSs.contract(M1, M2; alg=alg, kwargs...)

    M = extractdiagonal(M, sites1_ewmul)

    ressites = Vector{eltype(siteinds(M1)[1])}[]
    for s in siteinds(M)
        s_ = unique(ITensors.noprime.(s))
        if length(s_) == 1
            push!(ressites, s_)
        else
            if s_[1] ∈ sites1
                push!(ressites, [s_[1]])
                push!(ressites, [s_[2]])
            else
                push!(ressites, [s_[2]])
                push!(ressites, [s_[1]])
            end
        end
    end
    return T4APartitionedMPSs.truncate(
        rearrange_siteinds(M, ressites); cutoff=cutoff, maxdim=maxdim)
end

function _findallsiteinds_by_tag(M::PartitionedMPS; tag::String="")
    return findallsiteinds_by_tag(only.(siteinds(M)); tag=tag)
end

