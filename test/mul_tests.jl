@testitem "test_mul.jl/preprocess_matmul" begin
    using Test
    import T4AQuantics
    using ITensors
    import T4AITensorCompat: random_mps, TensorTrain

    @testset "_preprocess_matmul" begin
        N = 2
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]
        sitesz = [Index(2, "z=$n") for n in 1:N]
        sites1 = collect(Iterators.flatten(zip(sitesx, sitesy)))
        sites2 = collect(Iterators.flatten(zip(sitesy, sitesz)))
        M1 = random_mps(sites1)
        M2 = random_mps(sites2)

        mul = T4AQuantics.MatrixMultiplier(sitesx, sitesy, sitesz)

        M1, M2 = T4AQuantics.preprocess(mul, M1, M2)

        flag = true
        for n in 1:N
            flag = flag && hasinds(M1[n], sitesx[n], sitesy[n])
            flag = flag && hasinds(M2[n], sitesy[n], sitesz[n])
        end
        @test flag
    end

    @testset "postprocess_matmul" begin
        N = 2
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]
        sitesz = [Index(2, "z=$n") for n in 1:N]
        mul = T4AQuantics.MatrixMultiplier(sitesx, sitesy, sitesz)

        links = [Index(1, "Link,l=$l") for l in 0:N]
        sites_pairs = [[sitesx[n], sitesz[n]] for n in 1:N]
        M = TensorTrain(ComplexF64, sites_pairs, 1)
        for n in 1:N
            inds_list = [links[n], links[n + 1], sitesx[n], sitesz[n]]
            M[n] = randomITensor(inds_list...)
        end

        M = T4AQuantics.postprocess(mul, M)

        flag = true
        for n in 1:N
            flag = flag && hasind(M[2 * n - 1], sitesx[n])
            flag = flag && hasind(M[2 * n], sitesz[n])
        end
        @test flag
    end
end

@testitem "mul_tests.jl/matmul" begin
    using Test
    import T4AQuantics
    using ITensors
    import T4AITensorCompat: random_mps, contract, Algorithm

    @testset "matmul" for T in [Float64, ComplexF64]
        N = 3
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]
        sitesz = [Index(2, "z=$n") for n in 1:N]
        mul = T4AQuantics.MatrixMultiplier(sitesx, sitesy, sitesz)

        sites1 = collect(Iterators.flatten(zip(sitesx, sitesy)))
        sites2 = collect(Iterators.flatten(zip(sitesy, sitesz)))
        M1 = random_mps(T, sites1)
        M2 = random_mps(T, sites2)

        # preprocess
        M1, M2 = T4AQuantics.preprocess(mul, M1, M2)

        # MPO-MPO contraction
        M = contract(M1, M2; alg=Algorithm("naive"))

        # postprocess
        M = T4AQuantics.postprocess(mul, M)

        M_mat_reconst = reshape(Array(reduce(*, M), [reverse(sitesx)..., reverse(sitesz)]),
            2^N, 2^N)

        # Reference data
        M1_mat = reshape(Array(reduce(*, M1), [reverse(sitesx)..., reverse(sitesy)]), 2^N,
            2^N)
        M2_mat = reshape(Array(reduce(*, M2), [reverse(sitesy)..., reverse(sitesz)]), 2^N,
            2^N)
        M_mat_ref = M1_mat * M2_mat

        @test M_mat_ref ≈ M_mat_reconst
    end

    @testset "elementwisemul" for T in [Float64, ComplexF64]
        N = 5
        sites = [Index(2, "n=$n") for n in 1:N]
        mul = T4AQuantics.ElementwiseMultiplier(sites)

        M1_ = random_mps(T, sites)
        M2_ = random_mps(T, sites)
        M1 = M1_
        M2 = M2_

        # preprocess
        M1, M2 = T4AQuantics.preprocess(mul, M1, M2)

        # MPO-MPO contraction
        M = contract(M1, M2; alg=Algorithm("naive"))

        # postprocess
        M = T4AQuantics.postprocess(mul, M)

        # Comparison with reference data
        M_reconst = Array(reduce(*, M), sites)
        M1_reconst = Array(reduce(*, M1_), sites)
        M2_reconst = Array(reduce(*, M2_), sites)

        @test M_reconst ≈ M1_reconst .* M2_reconst
    end
end

@testitem "mul_tests.jl/batchedmatmul" begin
    using Test
    import T4APartitionedTT: project, Projector, SubDomainTT, PartitionedTT, isprojectedat
    import T4AQuantics
    using ITensors
    import T4AITensorCompat: random_mps, TensorTrain, siteinds

    """
    Reconstruct 3D matrix
    """
    function _tomat3(a)
        sites_vec = siteinds(a)
        sites_flat = collect(Iterators.flatten(sites_vec))
        N = length(sites_flat)
        Nreduced = N ÷ 3
        sites_ = [sites_flat[1:3:N]..., sites_flat[2:3:N]..., sites_flat[3:3:N]...]
        result = reduce(*, a)
        return reshape(Array(result, sites_), 2^Nreduced, 2^Nreduced, 2^Nreduced)
    end

    @testset "batchedmatmul" for T in [Float64, ComplexF64]
        """
        C(x, z, k) = sum_y A(x, y, k) * B(y, z, k)
        """
        nbit = 2
        D = 2
        sx = [Index(2, "Qubit,x=$n") for n in 1:nbit]
        sy = [Index(2, "Qubit,y=$n") for n in 1:nbit]
        sz = [Index(2, "Qubit,z=$n") for n in 1:nbit]
        sk = [Index(2, "Qubit,k=$n") for n in 1:nbit]

        sites_a = collect(Iterators.flatten(zip(sx, sy, sk)))
        sites_b = collect(Iterators.flatten(zip(sy, sz, sk)))

        a = random_mps(T, sites_a; linkdims=D)
        b = random_mps(T, sites_b; linkdims=D)

        # Reference data
        a_arr = _tomat3(a)
        b_arr = _tomat3(b)
        ab_arr = zeros(T, 2^nbit, 2^nbit, 2^nbit)
        for k in 1:(2^nbit)
            ab_arr[:, :, k] .= a_arr[:, :, k] * b_arr[:, :, k]
        end

        ab = T4AQuantics.automul(a, b; tag_row="x", tag_shared="y", tag_col="z", alg="fit")
        ab_arr_reconst = _tomat3(ab)
        @test ab_arr ≈ ab_arr_reconst
    end

    @testset "PartitionedTT" begin
        @testset "batchedmatmul" for T in [Float64, ComplexF64]
            """
            C(x, z, k) = sum_y A(x, y, k) * B(y, z, k)
            """
            nbit = 2
            D = 2
            cutoff = 1e-25
            maxdim = typemax(Int)
            sx = [Index(2, "Qubit,x=$n") for n in 1:nbit]
            sy = [Index(2, "Qubit,y=$n") for n in 1:nbit]
            sz = [Index(2, "Qubit,z=$n") for n in 1:nbit]
            sk = [Index(2, "Qubit,k=$n") for n in 1:nbit]

            sites_a = collect(Iterators.flatten(zip(sx, sy, sk)))
            sites_b = collect(Iterators.flatten(zip(sy, sz, sk)))

            a = random_mps(T, sites_a; linkdims=D)
            b = random_mps(T, sites_b; linkdims=D)

            # Reference data
            a_arr = _tomat3(a)
            b_arr = _tomat3(b)
            ab_arr = zeros(T, 2^nbit, 2^nbit, 2^nbit)
            for k in 1:(2^nbit)
                ab_arr[:, :, k] .= a_arr[:, :, k] * b_arr[:, :, k]
            end

            a_ = PartitionedTT([
                project(a, Projector(Dict(sx[1] => 1, sy[1] => 1))),
                project(a, Projector(Dict(sx[1] => 1, sy[1] => 2))),
                project(a, Projector(Dict(sx[1] => 2, sy[1] => 1))),
                project(a, Projector(Dict(sx[1] => 2, sy[1] => 2)))
            ])

            b_ = PartitionedTT([
                project(b, Projector(Dict(sy[1] => 1, sz[1] => 1))),
                project(b, Projector(Dict(sy[1] => 1, sz[1] => 2))),
                project(b, Projector(Dict(sy[1] => 2, sz[1] => 1))),
                project(b, Projector(Dict(sy[1] => 2, sz[1] => 2)))
            ])

            @test a ≈ TensorTrain(a_)
            @test b ≈ TensorTrain(b_)

            ab = T4AQuantics.automul(
                a_, b_; tag_row="x", tag_shared="y", tag_col="z", alg="fit", cutoff, maxdim
            )
            ab_ref = T4AQuantics.automul(
                a, b; tag_row="x", tag_shared="y", tag_col="z", alg="fit", cutoff, maxdim
            )

            @test TensorTrain(ab)≈ab_ref rtol=10 * sqrt(cutoff)
        end
    end
end
