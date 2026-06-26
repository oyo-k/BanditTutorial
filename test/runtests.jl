using DrWatson, Test
@quickactivate "BanditTutorial"

using Random, Statistics

include(srcdir("bandit.jl"))

# Run test suite
println("Starting tests")
ti = time()

@testset "BanditTutorial tests" begin
    @testset "softmax_choice" begin
        # 選ばれる腕は必ず有効なインデックス
        q = [0.0, 1.0]
        for _ in 1:100
            c = softmax_choice(q, 5.0)
            @test c in 1:length(q)
        end
        # beta が大きいと Q 値最大の腕に強く偏る
        Random.seed!(1)
        choices = [softmax_choice([0.0, 1.0], 50.0) for _ in 1:200]
        @test mean(choices .== 2) > 0.9
    end

    @testset "moving_average" begin
        @test moving_average([1.0, 1.0, 1.0], 2) == [1.0, 1.0, 1.0]
        ma = moving_average([0.0, 1.0], 2)
        @test ma[1] == 0.0
        @test ma[2] == 0.5
    end

    @testset "run_bandit" begin
        Random.seed!(42)
        acc = run_bandit(100, [0.2, 0.8], 0.3, 5.0)
        @test length(acc) == 100
        @test all(x -> x in (0.0, 1.0), acc)
        # 学習が進めば後半は最適腕を高い割合で選ぶ
        @test mean(acc[end-19:end]) > 0.8
    end
end

ti = time() - ti
println("\nTest took total time of:")
println(round(ti/60, digits = 3), " minutes")
