# ============================================================
# src/io.jl — シミュレーション結果の保存
# ============================================================
#
# シミュレーション結果（p_optimal, regret）を受け取り、
# DrWatson の命名規則に従って data/sims/ へ CSV として書き出す。
#
# モデル（bandit.jl）と保存ロジックを分離することで、
# 将来フォーマットを変えたい場合（HDF5、JLD2 など）でも
# このファイルだけを修正すればよい。
# ============================================================

using DrWatson
using CSV, DataFrames

"""
    save_sim(p_optimal, mean_regret, reward_probs, alpha, beta, seed, n_trials, n_reps) -> String

シミュレーション結果を `data/sims/` へ CSV 保存し、保存先パスを返す。

ファイル名は DrWatson の `savename` で自動生成される。
例: `bandit_alpha=0.3_beta=5.0_n_arms=2_n_reps=200_n_trials=100_seed=42.csv`
"""
function save_sim(p_optimal, mean_regret, reward_probs, alpha, beta, seed, n_trials, n_reps)
    n_arms = length(reward_probs)

    # 結果を 1 行 = 1 試行の DataFrame にまとめる
    # パラメータ列も一緒に保存しておくと、複数 CSV を結合したときに
    # どの条件か識別できる（notebook のセクション 4 で活用）
    df = DataFrame(
        trial     = 1:length(p_optimal),
        p_optimal = p_optimal,
        regret    = mean_regret,
        alpha     = alpha,
        beta      = beta,
        seed      = seed,
        n_trials  = n_trials,
        n_reps    = n_reps,
    )

    # 各腕の報酬確率を列として追加（p_arm1, p_arm2, ...）
    for (i, p) in enumerate(reward_probs)
        df[!, Symbol("p_arm$i")] .= p
    end

    # @strdict でパラメータの Dict を作り、savename でファイル名を生成する
    # n_arms をファイル名に含めることで腕の数が違う条件を区別できる
    params = @strdict alpha beta n_arms seed n_trials n_reps
    outdir = datadir("sims")
    mkpath(outdir)  # data/sims/ がなければ作成
    fpath  = joinpath(outdir, savename("bandit", params, "csv"))

    CSV.write(fpath, df)
    println("Saved: ", fpath)
    return fpath
end
