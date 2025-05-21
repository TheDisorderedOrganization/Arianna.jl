using SafeTestsets

@safetestset "Distribution Test" begin
    include("distribution_test.jl")
end

@safetestset "PGMC Test" begin
    include("pgmc_test.jl")
end

@safetestset "PGMG AD Backends" begin
    include("ad_backends_test.jl")
end

@safetestset "Quality Assurance" begin
    using Aqua
    Aqua.test_all(Arianna)
end