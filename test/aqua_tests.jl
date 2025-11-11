@testitem "code quality test" begin
    using Test
    using Aqua
    import T4AQuantics

    @testset "Aqua" begin
        Aqua.test_stale_deps(T4AQuantics)
    end
end
