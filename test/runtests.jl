using RRCF
using Test

@testset "RRCF.jl" begin
    tree = RCTree([rand(Float32, 256) for i in 1:200])

    tree = RCTree(256)
    for i in 1:200
        RRCF.insert_point(tree, rand(Float32, 256), UInt32(i))
    end

    print(tree)
    N = 1000
    @time tree = RCTree([rand(Float32, 256) for i in 1:N]);
    print([RRCF.codisp(tree, i) for i in 1:N])
    print([RRCF.disp(tree, i) for i in 1:N])
end
