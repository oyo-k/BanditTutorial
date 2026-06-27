# ============================================================
# scripts/run.jl — 1 パラメータ条件のシミュレーション実行スクリプト
# ============================================================
#
# 使い方:
#   julia scripts/run.jl SEED N_TRIALS N_REPS ALPHA BETA P1 P2 [P3 ...]
#
# 引数:
#   SEED       乱数シード（再現性のために固定する整数）
#   N_TRIALS   1 回のシミュレーションで行う試行数
#   N_REPS     シミュレーションを繰り返す回数（平均化用）
#   ALPHA      Q 学習の学習率（0〜1）
#   BETA       softmax の逆温度（大きいほど貪欲）
#   P1 P2 ...  各腕の報酬確率（2 本以上、スペース区切り）
#
# 例:
#   julia scripts/run.jl 42 100 200 0.3 5.0 0.2 0.8
#
# 出力:
#   data/sims/bandit_alpha=…_beta=…_n_arms=…_….csv
# ============================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)  # プロジェクト環境を有効化
Pkg.instantiate(; io=devnull)                        # 未インストールのパッケージを補完

using DrWatson       # projectdir / datadir / savename などのパス管理
using Random, Statistics

include(srcdir("bandit.jl"))  # モデル: softmax_choice / run_bandit
include(srcdir("io.jl"))      # 保存: save_sim

function main(args)
    # 引数が足りない場合は使い方を表示して終了
    if length(args) < 7
        println(stderr, """
            引数が設定されていません。

            使い方: julia scripts/run.jl SEED N_TRIALS N_REPS ALPHA BETA P1 P2 [P3 ...]

            SEED       乱数シード
            N_TRIALS   試行数
            N_REPS     繰り返し回数（平均化用）
            ALPHA      学習率（0〜1）
            BETA       softmax 逆温度
            P1 P2 ...  各腕の報酬確率（2 本以上）

            例: julia scripts/run.jl 42 100 200 0.3 5.0 0.2 0.8
            """)
        exit(1)
    end

    seed         = parse(Int,     args[1])
    n_trials     = parse(Int,     args[2])
    n_reps       = parse(Int,     args[3])
    alpha        = parse(Float64, args[4])
    beta         = parse(Float64, args[5])
    reward_probs = parse.(Float64, args[6:end])  # 残りすべてが報酬確率

    # --------------------------------------------------------
    # N_REPS 回のシミュレーションを実行して平均を取る
    # --------------------------------------------------------
    # seed を毎回 seed+r にずらすことで、各繰り返しが
    # 独立した乱数列を使うようにしている。
    acc = zeros(n_trials)
    reg = zeros(n_trials)
    for r in 1:n_reps
        Random.seed!(seed + r)
        result = run_bandit(n_trials, reward_probs, alpha, beta)
        acc .+= result.accuracy
        reg .+= result.regret
    end
    p_optimal   = acc ./ n_reps  # 各試行の最適腕選択率（0〜1）
    mean_regret = reg ./ n_reps  # 各試行の平均後悔（累積は notebook 側で cumsum）

    # 結果の保存は src/io.jl の save_sim に委譲する
    save_sim(p_optimal, mean_regret, reward_probs, alpha, beta, seed, n_trials, n_reps)
end

main(ARGS)
