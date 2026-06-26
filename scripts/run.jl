import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)
Pkg.instantiate(; io=devnull)

using DrWatson
using CSV, DataFrames
using Random, Statistics

# シミュレーション関数本体は src/bandit.jl
include(srcdir("bandit.jl"))

# 使い方:  julia scripts/run.jl SEED N_TRIALS N_REPS ALPHA BETA
#
# 解析の主体は notebooks/bandit_qlearning.ipynb。
# このスクリプトは 1 つのパラメータ条件についてシミュレーションを実行し、
# 各試行で最適腕を選んだ割合（N_REPS 回平均）を datadir("sims") へ CSV 保存する。

const REWARD_PROBS = [0.2, 0.8]   # 2 本腕の報酬確率（固定）
const DEFAULTS     = ["42", "100", "200", "0.3", "5.0"]

function main(args)
    seed     = parse(Int,     args[1])
    n_trials = parse(Int,     args[2])
    n_reps   = parse(Int,     args[3])
    alpha    = parse(Float64, args[4])
    beta     = parse(Float64, args[5])

    # n_reps 回の独立シミュレーションを平均して学習曲線を滑らかにする
    acc = zeros(n_trials)
    for r in 1:n_reps
        Random.seed!(seed + r)
        acc .+= run_bandit(n_trials, REWARD_PROBS, alpha, beta)
    end
    p_optimal = acc ./ n_reps

    df = DataFrame(
        trial     = 1:n_trials,
        p_optimal = p_optimal,
        alpha     = alpha,
        beta      = beta,
        seed      = seed,
        n_trials  = n_trials,
        n_reps    = n_reps,
    )

    params = @strdict alpha beta seed n_trials n_reps
    outdir = datadir("sims")
    mkpath(outdir)
    fpath  = joinpath(outdir, savename("bandit", params, "csv"))
    CSV.write(fpath, df)
    println("Saved: ", fpath)
end

main(isempty(ARGS) ? DEFAULTS : ARGS)
