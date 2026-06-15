import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)
Pkg.instantiate(; io=devnull)

using DrWatson
using Plots
using Random
using Statistics

# パラメータ
const SEED       = 42
const N_TRIALS   = 100
const REWARD_PROBS = [0.2, 0.8]   # 2本腕の報酬確率
const ALPHA      = 0.3             # 学習率
const BETA       = 5.0             # 逆温度（softmax の鋭さ）

Random.seed!(SEED)

# softmax 選択
function softmax_choice(q, beta)
    w = exp.(beta .* q)
    p = w ./ sum(w)
    return findfirst(cumsum(p) .>= rand())
end

# Q 学習シミュレーション
function run_bandit(n_trials, reward_probs, alpha, beta)
    n_arms   = length(reward_probs)
    q        = fill(0.0, n_arms)
    accuracy = zeros(n_trials)

    for t in 1:n_trials
        choice       = softmax_choice(q, beta)
        reward       = rand() < reward_probs[choice] ? 1 : 0
        q[choice]   += alpha * (reward - q[choice])
        accuracy[t]  = choice == argmax(reward_probs) ? 1.0 : 0.0
    end

    return accuracy
end

accuracy = run_bandit(N_TRIALS, REWARD_PROBS, ALPHA, BETA)

# 移動平均で学習曲線を平滑化
window = 10
smoothed = [mean(accuracy[max(1, t-window+1):t]) for t in 1:N_TRIALS]

# プロット生成・保存
p = plot(smoothed;
    xlabel = "Trial",
    ylabel = "P(optimal)",
    title  = "Q-learning on 2-arm bandit (α=$(ALPHA), β=$(BETA))",
    ylim   = (0, 1),
    legend = false,
    lw     = 2,
)
hline!(p, [0.5]; linestyle=:dash, color=:gray)

params = @dict ALPHA BETA SEED
fname  = savename("learning_curve", params, "png")
savefig(p, plotsdir(fname))
println("Saved: ", plotsdir(fname))
