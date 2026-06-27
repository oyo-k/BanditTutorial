# BanditTutorial

[Julia](https://julialang.org/) と [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/) を使った多腕バンディットQ学習のチュートリアルプロジェクト。

作者: Kura

---

## 概要

2本腕バンディット課題上でQ学習エージェントを動かし、学習率α・逆温度βのパラメーターsweepによる学習曲線と累積regretの変化を可視化する。

- 報酬確率: アーム1 = 0.2、アーム2 = **0.8**（固定）
- 行動選択: softmax（逆温度 β）
- Q値更新: δ則（`Q ← Q + α(r − Q)`）
- 評価指標: 最適腕選択率 `p_optimal` と累積後悔 `cumulative regret`

---

## ワークフロー図

```mermaid
flowchart TD
    A([START]) --> B[notebooks/bandit_qlearning.ipynb を開く]
    B --> C{パラメータ sweep<br/>α × β の全組み合わせ}
    C --> D["scripts/run.jl をサブプロセス実行<br/>(SEED, N_TRIALS, N_REPS, α, β)"]
    D --> E[src/bandit.jl で N_REPS 回シミュレーション]
    E --> F[p_optimal を CSV として data/sims/ へ保存]
    F --> C
    C --> G[全 CSV を DataFrame に集計]
    G --> H[学習曲線を比較プロット]
    H --> I[plots/ へ PNG 保存]
    I --> Z([END])

    style D fill:#dbeafe
    style E fill:#dbeafe
    style F fill:#fef9c3
    style G fill:#fef9c3
    style H fill:#dcfce7
    style I fill:#dcfce7
```

---

## データフロー図

```mermaid
flowchart LR
    subgraph Params["パラメータ入力"]
        P1[α 学習率]
        P2[β 逆温度]
        P3[seed / N_TRIALS / N_REPS]
    end

    subgraph Sim["シミュレーション ×N_REPS (src/bandit.jl)"]
        direction TB
        S1["Q 値初期化<br/>Q = [0, 0]"]
        S2["softmax_choice(Q, β)<br/>→ choice"]
        S3["報酬サンプリング<br/>r ~ Bernoulli(p_arm)"]
        S4["Q 値更新<br/>Q[choice] += α(r − Q[choice])"]
        S5{次の試行へ}
        S1 --> S2 --> S3 --> S4 --> S5
        S5 -->|t ≤ N_TRIALS| S2
        S5 -->|t > N_TRIALS| S6[accuracy ベクトル]
    end

    subgraph Agg["集計 (scripts/run.jl)"]
        A1["N_REPS 回の accuracy を平均<br/>→ p_optimal[t]"]
        A2["DataFrame 化<br/>(trial, p_optimal, α, β, ...)"]
        A3["CSV 保存<br/>data/sims/bandit_....csv"]
        A1 --> A2 --> A3
    end

    subgraph Viz["可視化 (notebook)"]
        V1["CSV 読み込み・結合"]
        V2["α × β ごとに学習曲線を描画"]
        V4["plots/ へ PNG 保存"]
        V1 --> V2 --> V3
    end

    Params --> Sim
    S6 --> Agg
    Agg --> Viz
```

---

## ファイル構成

```
BanditTutorial/
├── notebooks/
│   └── bandit_qlearning.ipynb   # メイン。パラメータ sweep → 集計 → 作図
├── scripts/
│   └── run.jl                   # 1 条件をシミュレーションして CSV 保存
├── src/
│   ├── bandit.jl                # モデル: softmax_choice / run_bandit
│   └── io.jl                    # 保存: save_sim（CSV 書き出し）
├── test/
│   └── runtests.jl              # ユニットテスト
├── data/
│   └── sims/                    # 生成された CSV（git 管理外）
└── plots/                       # 生成された図（git 管理外）
```

### 主要ファイルの役割

| ファイル | 役割 |
|---|---|
| [`notebooks/bandit_qlearning.ipynb`](notebooks/bandit_qlearning.ipynb) | sweep 定義・集計・作図のメイン |
| [`scripts/run.jl`](scripts/run.jl) | 1パラメーター条件のシミュレーション実行（`bandit.jl` + `io.jl` を呼ぶ） |
| [`src/bandit.jl`](src/bandit.jl) | モデル: `softmax_choice` / `run_bandit` |
| [`src/io.jl`](src/io.jl) | 保存: `save_sim`（DataFrame 構築〜CSV 書き出し） |
| [`test/runtests.jl`](test/runtests.jl) | 各関数のユニットテスト |

---

## セットアップ

```julia
# 1. DrWatson をグローバルにインストール
julia> using Pkg
julia> Pkg.add("DrWatson")

# 2. プロジェクトを有効化してパッケージを揃える
julia> Pkg.activate("path/to/BanditTutorial")
julia> Pkg.instantiate()
```

### notebook の起動

```julia
julia> using IJulia; notebook(dir="notebooks")
```

Juliaカーネルは **`julia-1.11`** を選択する。

### スクリプト単体実行

```bash
julia scripts/run.jl SEED N_TRIALS N_REPS ALPHA BETA P1 P2 [P3 ...]
# 例: julia scripts/run.jl 42 100 200 0.3 5.0 0.2 0.8
```

引数を省略した場合は使い方のメッセージが表示される。

### テスト実行

```bash
julia --project=. test/runtests.jl
```

---

## パラメーター

| 引数 | 型 | 説明 |
|---|---|---|
| `SEED` | Int | 乱数シード |
| `N_TRIALS` | Int | 1 回のシミュレーションの試行数 |
| `N_REPS` | Int | 反復回数（平均化用） |
| `ALPHA` (α) | Float64 | Q 学習の学習率（0〜1） |
| `BETA` (β) | Float64 | softmax の逆温度 |
| `P1 P2 ...` | Float64... | 各腕の報酬確率（2 本以上、スペース区切り） |

---

## CI

GitHub ActionsでJulia 1.11 / ubuntu-latest / x64による自動テストを実行している（`.github/workflows/CI.yml`）。
`main`へのプッシュおよびPull Requestでトリガーされる。
