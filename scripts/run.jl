# ============================================================
# scripts/run.jl — 1 パラメータ条件のシミュレーション実行スクリプト
# ============================================================
#
# 使い方:
#   julia scripts/run.jl SEED N_TRIALS N_REPS ALPHA BETA
#
# 引数:
#   SEED      乱数シード（再現性のために固定する整数）
#   N_TRIALS  1 回のシミュレーションで行う試行数
#   N_REPS    シミュレーションを繰り返す回数（平均化用）
#   ALPHA     Q 学習の学習率（0〜1）
#   BETA      softmax の逆温度（大きいほど貪欲）
#
# 出力:
#   data/sims/bandit_alpha=…_beta=…_….csv
#
# 役割:
#   このスクリプトは「1 つのパラメータ条件」だけを担当する。
#   複数条件の sweep やグラフ描画は notebooks/bandit_qlearning.ipynb が行う。
#   ノートブックがこのスクリプトをサブプロセスとして条件ごとに呼び出す設計。
# ============================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)  # プロジェクト環境を有効化
Pkg.instantiate(; io=devnull)                        # 未インストールのパッケージを補完

using DrWatson       # projectdir / datadir / savename などのパス管理
using Random, Statistics

include(srcdir("bandit.jl"))  # モデル: softmax_choice / run_bandit
include(srcdir("io.jl"))      # 保存: save_sim

# 2 本腕の報酬確率（腕 1 = 0.2、腕 2 = 0.8）
# この値はスクリプト全体で固定。変えたい場合はここを編集する。
const REWARD_PROBS = [0.2, 0.8]

# 引数なしで実行したときに使うデフォルト値
# （ノートブックから呼ぶときは引数を渡すので通常は使われない）
const DEFAULTS = ["42", "100", "200", "0.3", "5.0"]

function main(args)
    # コマンドライン引数を適切な型に変換する
    seed     = parse(Int,     args[1])
    n_trials = parse(Int,     args[2])
    n_reps   = parse(Int,     args[3])
    alpha    = parse(Float64, args[4])
    beta     = parse(Float64, args[5])

    # --------------------------------------------------------
    # N_REPS 回のシミュレーションを実行して平均を取る
    # --------------------------------------------------------
    # 1 回のシミュレーションだけでは乱数の偶然に左右されて
    # 結果がぶれてしまう。N_REPS 回繰り返して平均することで
    # 「このパラメータ条件の典型的な学習曲線」が得られる。
    #
    # seed を毎回 seed+r にずらすことで、各繰り返しが
    # 独立した乱数列を使うようにしている。
    acc = zeros(n_trials)  # accuracy（最適腕選択率）の累積
    reg = zeros(n_trials)  # regret（後悔）の累積
    for r in 1:n_reps
        Random.seed!(seed + r)
        result = run_bandit(n_trials, REWARD_PROBS, alpha, beta)
        acc .+= result.accuracy  # 試行ごとに加算（後で平均する）
        reg .+= result.regret
    end
    p_optimal   = acc ./ n_reps  # 各試行の最適腕選択率（0〜1）
    mean_regret = reg ./ n_reps  # 各試行の平均後悔（累積は notebook 側で cumsum）

    # 結果の保存は src/io.jl の save_sim に委譲する
    save_sim(p_optimal, mean_regret, alpha, beta, seed, n_trials, n_reps)
end

# ARGS（コマンドライン引数）が空ならデフォルト値で実行
main(isempty(ARGS) ? DEFAULTS : ARGS)
