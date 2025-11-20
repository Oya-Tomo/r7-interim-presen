#set page(
  paper: "jis-b1",
  margin: (
    top: 2cm,
    bottom: 2cm,
    left: 3cm,
    right: 3cm,
  )
)

#set text(
  font: "Noto Sans CJK JP",
  size: 28pt,
)

#show link: it => {
  text(rgb("#0088ff"))[
    #it
  ]
}

#show figure.where(
  kind: table
): set figure.caption(position: top)

#align(center)[
  #box(
    fill: rgb(200, 230, 255, 255),
    inset: (
      top: 2cm,
      bottom: 2cm,
    ),
    width: 100%
  )[
    = #text(48pt)[
      不整地・障害物フィールドに対応する\
      4脚ロボット制御のための強化学習手法の実装と比較
    ]
    #v(1cm)
    #text(weight: "bold")[村上研究室 f21040 5J13 小山田智典]
  ]
]

#v(15pt, weak: true)

#let section_box(title, width, height, content) = {
  rect(
    width: width,
    height: height,
    stroke: none,
  )[
    #place(
      center + top,
      dy: 25pt,
    )[
      #rect(
        width: 100%,
        height: 100% - 20pt,
        stroke: rgb(200, 230, 255, 255) + 4pt,
      )
    ]
    #pad(
      top: 0pt,
      left: 20pt,
      right: 20pt,
      bottom: 20pt,
    )[
      #rect(
        stroke: rgb(200, 230, 255, 255) + 4pt,
        fill: white,
        inset: 15pt
      )[
        == #text(weight: "bold", size: 28pt)[#title]
      ]
      #v(1cm, weak: true)
      #content
    ]
  ]
  v(25pt, weak: true)
}

#let bg_section = section_box("背景", 100%, 9%)[
  　近年，脚式ロボット (e.g. Unitree Go2，ANYmal，Unitree G1) の市販流通が進んでおり，畑や山などの不整地環境での活用が期待されている．
  特に4脚ロボットはその安定性から実環境での試験運用が進んでいる．
  しかし，不整地歩行のための強化学習モデルは学習済みのモデルウェイトが公開されておらず，ほとんどの場合，独自に訓練する必要がある．
  そこで，4脚ロボットの不整地歩行を可能にするソフトウェア・ハードウェアの基盤を構築することを目的とし，既存手法の調査と再実装，機能の比較を行う．
]


#let impl_section = section_box("実装", 100%, 40.5%)[
  検証のために，モデルおよび強化学習のアルゴリズムとシミュレーション環境を実装した．

  *実装に使用したライブラリ*
  - PyTorch: 深層学習フレームワーク
  - Genesis: ロボットシミュレーション環境構築ライブラリ

  *強化学習環境の実装*\
  - ロボットモデル : Unitree Go2 (4脚ロボット)
  - フィールド : 平坦な地面，不整地フィールド
  - 観測情報 : 60次元の状態情報
    - ボディの速度 $v_"xyz"$ (ボディ座標系)
    - ボディの回転角速度 $omega_"xyz"$ (ボディ座標系)
    - 関節の角度 $theta$ ，角速度 $theta'$ ，トルク $tau$
    - 一つ前の行動 $a_(t-1)$
    - 目標動作コマンド $c$ (前進速度，横移動速度，回転速度)

  - 報酬設計 : 12項目
    - xy速度，yaw角速度 : 目標動作コマンドとの一致度
    - 足の先端位置 : 移動軌跡との近さ
    - 足の接地状態 : 対角線の足が接地しているか
    - 肩の高さ : 地面と一定の距離を保っているか
    - ボディの安定性 : 重力ベクトルのz成分の大きさ
    - 行動の最小化 : 初期姿勢からの行動変化の抑制
    - 縦揺れ抑制，回転抑制 : ボディの不必要な振動と回転を抑制する
    - 足先，関節の円滑性 : 足の先端と関節角度の変化の滑らかさ
    - 衝突回避 : ボディと地面の衝突を避ける

]

#let comp = section_box("比較・考察", 100%, 21.5%)[
  Table 1 に示す3つの手法を実装し，歩行性能の比較を行った．

  #figure(
    gap: 20pt,
    table(
      columns: 3,
      align: horizon + center,
      inset: 10pt,
      table.header(
        [手法], [平地歩行], [不整地歩行]
      ),
      [PPO + MLP], [○], [✕],
      [PPO + RNN], [○], [✕],
      [SLR], [○], [○],
    ),
    caption: "平地・不整地歩行の可否検証結果",
  )

  　MLPでは過去の情報を用いず，RNNでは効率的な過去情報の伝搬を学習することができなかったため，不整地歩行は困難であった．
  一方でSLRは，過去の観測情報を時系列的に一貫性のある潜在表現に変換して，効率的に利用することで路面状況を推測することができ，不整地歩行が可能になったと考えられる．
]

#let ppo_section = section_box("強化学習手法: Proximal Policy Optimization", 100%, 18.5%)[
  　Proximal Policy Optimization (以下，PPO) はActor-Critic法をベースとする強化学習であり，ロボット制御に限らず様々な分野の強化学習タスクで高い性能を発揮している．
  PPOは，方策関数 (Policy) の勾配を安定的に更新することを目的としており，クリッピング手法を用いることで急激な方策更新による性能の劣化を防いでいる．
  PPOの最大化目的は以下の式で表される．(出典: @schulman2017proximalpolicyoptimizationalgorithms)

  #grid(
    columns: (65%, 34%),
    gutter: -30pt,
    $
      hat(A)_t = delta_t + ( gamma lambda ) delta_(t+1) + ... + ( gamma lambda )^(T-t+1) delta_(T-1)\
      "where" space.quad delta_t = r_t + gamma V( s_(t+1) ) - V( s_t )
    $,
    $
      r_(t)(θ) = frac( pi_(θ)( a_t | s_t ), pi_(θ_"old")( a_t | s_t ))
    $
  )
  $
    L^"CLIP" (θ) = hat(EE)_t [ min( r_(t)(θ) hat(A)_t, "clip"( r_(t)(θ), 1 - ϵ, 1 + ϵ ) hat(A)_t ) ]
  $
]


#let slr_section = section_box("先行研究: Self-learning Latent Representation", 100%, 38.5%)[
  　Self-learning Latent Representation (以下，SLR) はPPOをベースとした手法であり，過去の観測情報を埋め込んでMLPに入力することで，不整地歩行に必要な情報を効率的に学習することを目的としている．

  #align(center)[
    #figure(
      caption: [SLRの学習フレームワーク (出典: @chen2024slrlearningquadrupedlocomotion)],
    )[
      #image("assets/slr_arch.png", width: 75%)
    ]
  ]

  *SLRが導入するモデル*
  - Encoder : 過去の観測情報を低次元の潜在表現に変換するモデル\

    #align(center)[
      $z_t = phi (o_t^H) space.quad$ ※ $o_t^H$ は ${t-h:t}$の範囲の観測情報
    ]

  - TransModel : 潜在表現の時系列変化を予測するモデル\
    $
    tilde(z)_(t+1) = mu (z_t, a_t)
    $

  　SLRの損失関数は，PPOの損失関数に加えて，潜在表現 $z$ の時系列整合性を保つためのトリプレット損失を導入している．
  $
    cal(L)_"trip" = max(||z_(t+1) - tilde(z)_(t+1)|| - ||z_(t+1) - z_(t+n)|| + m, 0), space.quad s.t. space.quad n != 1\
  $
]

#let plan_section = section_box("今後の展望", 100%, 13%)[
  　現状の手法では，RGBカメラやDepthカメラなどの視覚情報を利用していないため，障害物の認識が困難であり複雑な不整地歩行には対応できていない．\
  　今後は，より走破性の高い方策モデルの構築を目指し，視覚情報を活用した手法 (e.g. @Miki_2022) の実装を行い，
  構築した方策モデルを用いて，実機ロボットへの移植や機能追加 (e.g 障害物回避，音声制御) を行う予定である．
]

#let ref_section = section_box("参考文献", 100%, 18%)[
  #bibliography("ref.bib", title: none, style: "ieee")
]

// render layout

#bg_section

#grid(
  columns: (50% - 28pt, 50%),
  gutter: 28pt,
  [
    #impl_section
    #comp
    #plan_section
  ],
  [
    #ppo_section
    #slr_section
    #ref_section
  ]
)


