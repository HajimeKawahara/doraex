# M8 `sigma_log_p` free 問題と highest-precision NUTS の引き継ぎ記録

日付: 2026-08-20
関連PR: [#11 Add M8 on-the-fly pressure retrieval and resolve free-sigma precision pathology](https://github.com/HajimeKawahara/doraex/pull/11)

## 1. 要約

M8 の凍結済み「雲圧力摂動の一次近似＋空間GPの解析的周辺化」targetでは、
`sigma_log_p` をfreeにしたときのNUTS病理は実用上解消した。

原因は、`sigma_log_p` というscale座標やHalfNormal priorそのものではなく、
57,344行×767列のlow-rank factorを使うfloat32尤度で、default matmul精度の
reverse-mode勾配が数値的に不整合になったことだった。専用sampling processで

```text
JAX_DEFAULT_MATMUL_PRECISION=highest
```

を指定すると、float32（`x64=false`）のまま局所scoreがJVPおよびfloat64参照値と
整合し、同じfree-`sigma_log_p` targetを効率よくsampleできた。

長期controlのv10（warmup 2,000 + retained 1,500、1 chain、seed 0）は、
divergence 0、tree-depth cap 0/1,500、retained step中央値15で完了した。
これは線形化target上のsampler問題についての結論であり、一次近似の物理的妥当性や
非線形forward posteriorの妥当性を示すものではない。

誤読を避けるため、計算内容を先に整理すると次のとおりである。

| 項目 | v10での扱い |
|---|---|
| 各chipの基準RTスペクトル | sampling stateごとにon-the-fly計算 |
| 768 pixelごとの完全RT | 行わない |
| pixel間の雲圧力差 | Taylor一次近似 |
| 767個のmap係数 | NUTSでsampleせず解析的周辺化 |
| `sigma_log_p` | NUTSでsample |
| parameter-space spectrum grid / DiffGrid / Hermite補間 | 使用しない |
| 波長grid・圧力層grid・opacity cache | 使用する |

## 2. v10で計算しているforward

### 2.1 on-the-flyである部分

大気スペクトルは、各尤度評価で現在の全球大気parameterを用いてExoJAXから
on-the-flyに計算する。4つのchipそれぞれについて、現在の

- `T0`
- `alpha`
- `log VMR(CO)`
- `log VMR(H2O)`
- `log VMR(CH4)`
- `log VMR(HF)`
- 全球平均 `log_p_cloud`

における基準局所スペクトルと
`dS / dlog10(Pcloud)`を`jax.jvp`で同時に評価する。`T0`、`alpha`、分子存在量、
全球平均雲圧力は固定gridから補間せず、sampling中に非線形forwardへ直接入る。

opacityはExoJAXの`OpaPremodit`と波長範囲ごとのcacheを使用する。
通常の計算gridとして、各chip 4,500点の波数gridと101層の圧力gridは存在する。
一方、parameterから完成スペクトルを取り出す補間grid、`OpaDiffgrid`、
Hermite補間は使用していない。

### 2.2 一次近似している部分

HEALPix NSIDE=8の768 surface pixelすべてについて完全な放射輸送を計算している
わけではない。pixelごとの雲圧力摂動だけを、現在の全球平均雲圧力の周りで

```text
S_pixel(lambda) ~= S_0(lambda)
                  + q_pixel * dS/dlog10(Pcloud)
```

と一次近似する。基準スペクトルと圧力微分はsampling stateごとに再計算されるため、
「一度作ったスペクトルgridをsampling中ずっと使う」モデルではない。

零平均制約を持つ768-pixel圧力mapの有効自由度は767である。このmap係数はNUTSで
sampleせず、固定相関長の空間GPとして解析的に周辺化する。観測数は

```text
4 chips × 14 phases × 1,024 wavelengths = 57,344
```

であり、尤度は概念的に

```text
LowRankMultivariateNormal(
    loc=baseline[57,344],
    cov_factor=F[57,344,767],
    cov_diag=noise[57,344],
)
```

となる。4 chipは同じ空間mapを共有するため、周辺化後の尤度にはchip間相関も含まれる。

## 3. sampling parameterと固定量

NUTSがsampleする自由度は12で、full dense mass matrixをadaptする。

| site | 次元 | 内容 |
|---|---:|---|
| `atmosphere_rotated` | 7 | `T0`, `alpha`, CO/H2O/CH4/HFのlog VMR、全球平均`log_p_cloud`を固定直交回転した座標 |
| `sigma_log_p` | 1 | 零平均圧力mapのscale、`HalfNormal(0.3)` |
| `A` | 4 | chipごとの規格化、`Uniform(1.0, 1.2)` |

主な固定量は次のとおり。

- `logg = 4.86`
- 空間GP相関長 `ell_b = 0.4 rad`
- `log_w[4,14]` と `sigma_d[4]`
- 回転、傾斜角、limb darkeningなどの幾何parameter
- HEALPix `nside = 8`
- GP jitter `5e-7`
- 零平均圧力map

767個のmap係数は「固定parameter」ではなく、尤度内で解析的に積分消去された潜在変数
である。

## 4. 問題の切り分け

### 4.1 historical M8 v1

v1はfree `sigma_log_p`、warmup 2,000 + retained 1,500、seed 0、full dense
adaptationの長期runだったが、NUTSはほぼ常に最大tree depthへ到達した。

| 指標 | v1 |
|---|---:|
| divergence | 0 |
| 2,047-step cap | 1,497 / 1,500 = 99.8% |
| retained step中央値 | 2,047 |
| retained step総数 | 3,067,428 |
| final step size | 0.00151033 |
| runtime | 407,483.96 s = 113.19 h |
| `sigma_log_p` ESS（NumPyro、一chain） | 約6.54 |

fixed-`sigma_log_p` controlでは改善したが、長期v3でもcapは540/1,500（36%）残った。
したがってfree scaleのcouplingは病理を悪化させたものの、scale自由度だけを固定すれば
根本解決するという結果ではなかった。

### 4.2 v5–v7探索診断の位置づけ

v5–v7ではprofile、有限差分、HVP、LowRank再構成を用いて原因を探索した。
value-levelのreplayは概ねfloat32の1–2 ULPで一致した一方、計算graphや微分modeが
異なるscore/Hessian比較は符号反転や非有限値を示した。これらのrunは最終的な
integrity gateを通っていないため、真のposterior geometryの定量結果としては採用しない。

ただしv7のraw captureでは、NUTSが使うreverse scoreがpure JVP、value secant、
float64解析値から系統的にずれることが確認され、v8のprecision分離実験につながった。

### 4.3 v8 LowRank precision診断

v8は同じ凍結targetの一状態とその近傍で、full target、明示的full-factor LowRank、
reduced float64参照を比較した。

| 比較 | default | highest | reduced float64 |
|---|---:|---:|---:|
| full targetの最大 `abs(reverse - JVP)` | 3.44824 | 0.0462208 | - |
| isolated full-factorの最大差 | 3.51436 | 0.0587349 | - |
| reduced系の差 | - | - | 1.376e-11 |

default精度では、sigma方向scoreの局所的な傾きから得た曲率の符号までfloat64参照と
反対になった。highestではfull target、isolated LowRankの両方でscoreと曲率が
float64参照へ戻った。sigma prior項のreverse/JVP差は無視できる大きさであり、問題は
主として巨大なfloat32 LowRank reductionへ局在した。

この診断は一状態近傍の結果であり、parameter空間全域についての数学的証明ではない。
また`highest`はLowRank部分だけでなく専用Python process内の全matmulへ作用する。

## 5. highest-precision control

### 5.1 v9短期control

v9はv1と同じ意図target、初期値、seed、NUTS設定を保ち、process-global matmul精度だけを
`highest`にした短期screeningである。

| 指標 | v9 |
|---|---:|
| chain / seed | 1 / 0 |
| warmup / retained | 200 / 20 |
| 初期点 `abs(reverse - JVP)` | 0.00529385 |
| divergence | 0 |
| cap | 0 / 20 |
| retained step min / median / max | 15 / 31 / 31 |
| retained step総数 | 604 |
| final step size | 0.100733 |
| mean acceptance | 0.950445 |

v9はsampling挙動のscreenであり、20 drawをposterior推定には使用しない。

### 5.2 v10長期control

v10は同じ介入でwarmup 2,000 + retained 1,500を実行した。

| 指標 | v10 |
|---|---:|
| chain / seed | 1 / 0 |
| precision | float32, `JAX_DEFAULT_MATMUL_PRECISION=highest` |
| warmup / retained | 2,000 / 1,500 |
| runtime (`mcmc.run`) | 8,730.30 s = 2.425 h |
| ExoJAX 4-chip setup（別計測） | 43.55 s |
| divergence | 0 |
| 2,047-step cap | 0 / 1,500 |
| retained step min / median / mean / max | 7 / 15 / 12.893 / 15 |
| retained step総数 | 19,340 |
| final step size | 0.334684 |
| mean acceptance（target 0.95） | 0.957909 |

prespecified screening条件（divergence 0、cap率5%未満、step中央値512未満）はすべて
満たした。

主要posterior量の中央値とcentral 90%区間は次のとおり。

| parameter | median | central 90% interval |
|---|---:|---:|
| `T0` [K] | 1184.699 | [1177.645, 1191.875] |
| `alpha` | 0.118625 | [0.112715, 0.124368] |
| `log_vmr_co` | -3.160736 | [-3.191824, -3.130965] |
| `log_vmr_h2o` | -3.403715 | [-3.427814, -3.379871] |
| `log_vmr_ch4` | -4.734905 | [-4.750618, -4.719307] |
| `log_vmr_hf` | -7.334660 | [-7.390124, -7.282669] |
| `log_p_cloud` | 1.354211 | [1.327609, 1.383842] |
| `sigma_log_p` | 0.387554 | [0.298114, 0.509381] |

`sigma_log_p`のrank-normalized within-chain ESSはbulk 1,322、tail 1,163だった。
12個のactive NUTS座標のbulk ESSは約1,284–2,004で、明瞭なchain内driftや
stickinessは見られなかった。ESSがdraw数を超える場合があるのは負の自己相関による。
一chainなので通常のchain間R-hatは定義できない。

## 6. 速度向上の意味

v10はv1より46.67倍短時間で終了し、retained step総数は158.6分の1になった。
これはDiffGridや1回のforward計算の高速化ではなく、勾配の数値整合性が回復して
NUTSのtrajectoryが短くなった効果である。

| 指標 | v1 default | v10 highest | 変化 |
|---|---:|---:|---:|
| runtime | 113.19 h | 2.425 h | 46.67倍短縮 |
| retained step総数 | 3,067,428 | 19,340 | 158.6分の1 |
| step中央値 | 2,047 | 15 | 136.5分の1 |
| final step size | 0.001510 | 0.334684 | 221.6倍 |
| cap率 | 99.8% | 0% | 解消 |

progress logから見積もったretained phaseの1 leapfrog stepは、v1で約0.0996 s、
v10で約0.1115 sだった。highestの1 stepはむしろ約12%遅い。したがって全体の高速化は
1 stepの演算速度ではなく、必要step数の減少で説明される。

約0.1115 sは4 chip、57,344観測点、スペクトルと圧力JVP、Doppler/design演算、
LowRank尤度、reverse-mode勾配を含む一回のfull potential-gradient評価の時間である。
v10は`--no-preflight-autodiff`で実行されたため、独立した「1 spectrumあたりの計算時間」
は保存されていない。0.1115 sを4で割った値を単一スペクトル時間として扱うことはできない。

## 7. 確立した事項と適用範囲

### 確立した事項

- 凍結済みの一次線形化・解析的周辺化M8 targetでは、free `sigma_log_p` のNUTS病理は
  不可避なscale-coordinate問題ではない。
- 大規模float32 LowRank尤度のdefault matmul精度によるreverse-mode score誤差が
  病理の主因である。
- process-global `highest`は、同じfloat32 targetの局所score整合性と長期sampling効率を
  回復した。
- `sigma_log_p`を固定しなくても、実際に0.2145–0.7136の範囲を探索しながらcapなしで
  sampleできた。
- 劇的なruntime短縮はNUTS step数の減少によるものであり、DiffGridの効果ではない。

### この結果からは確立していない事項

- 1 chain・1 seedのため、chain間収束、初期値依存、未発見modeの不在は証明していない。
- total Hamiltonian energyを保存していないため、E-BFMIは評価できない。
- historical v1はambient matmul policyをartifactへ保存していない。v1がdefault/unsetだった
  という解釈はlauncher sourceとv8の現環境再現に基づく。
- `highest`はfloat64化ではなく、LowRank演算だけに限定した介入でもない。
- この結果はTaylor一次近似の妥当性を検証しない。既存のexact-forward screenでは、
  v10が推定した`約0.30–0.51`の`sigma_log_p`領域の多くで線形化誤差が無視できない。
- したがってv10 posteriorは「凍結済み線形targetに対するposterior」であり、検証済みの
  非線形・物理posteriorではない。

## 8. コードとartifact

主なtracked source:

- [on-the-fly retrieval本体](../src/doraex/workflows/on_the_fly_pressure_retrieval.py)
- [ExoJAX forward](../src/doraex/spectra/exojax_forward.py)
- [M8 v1 sampling設定](../examples/luhman16b_yama/m8_v1_run.py)
- [v8 LowRank precision診断](../examples/luhman16b_yama/check_m8_v8_free_sigma_lowrank_precision.py)
- [v9短期control](../examples/luhman16b_yama/check_m8_v9_free_sigma_highest_precision_control.py)
- [v10長期control](../examples/luhman16b_yama/check_m8_v10_free_sigma_highest_precision_long_control.py)
- [v8 launcher](../csh/exe_m8_v8_check_free_sigma_lowrank_precision.csh)
- [v9 launcher](../csh/exe_m8_v9_free_sigma_highest_precision_v17_control.csh)
- [v10 launcher](../csh/exe_m8_v10_free_sigma_highest_precision_long.csh)
- [英語版の正式結論](m8_free_sigma_highest_precision_conclusion.md)

外部artifact path:

- v8: `results/m8/v8/free_sigma_lowrank_precision/`
- v9: `results/m8/v9/free_sigma_highest_precision_seed0/`
- v10: `results/m8/v10/free_sigma_highest_precision_long_seed0/`

主要SHA-256:

```text
7db88c6fff97d4fa1421898598d8e1d5bff99ca204cd03d70cacc303023d6efc  v8 summary
a1a1000166981637e4a8ac988d713c184b44664431d384dd7e94f10a179c8f81  v9 summary
b9b703f494af44f495182cc65784e7aceb5b99e62d996ccdde96cdaff21ae7b7  v10 summary
c79d8f2c24e25a35bd5827628af8b2e0e5b14202ca7e8229f4545b338e46209f  v10 samples
140b52e99977c82f931da150f7d9c265cd909ba5f9f28be801290aa9d07a26d3  v10 diagnostics
```

result directory、run log、sentinel、provenance manifestはGitに含めず、外部artifactとして
保持されている。artifactless CIでは、このbundleを直接replayする15件だけを理由付きで
skipし、同じmodule内のsynthetic unit testは実行する。凍結済みv8–v10 sourceとそのSHA
manifestはCI対応のために書き換えていない。tracked noteとPR本文には、再解釈に必要な
主要数値と限定条件を記録した。
