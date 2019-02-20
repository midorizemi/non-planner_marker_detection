# 非平面マーカ検出
非平面マーカを検出し，マーカの形状を得るためにマーカをメッシュに分割する．
しかし，メッシュ上には画像特徴が十分でない場合がある．
そこで，メッシュ上の特徴点の数を周辺のメッシュ上の特徴点数と比較しメッシュを統合する．

実験スクリプト　　expt_split_affinesim_combinable.py

さらに，非平面なマーカはセルフオクルージョンが存在しメッシュの位置推定できない場合がある．
そこで，位置推定でいたメッシュの句形頂点の座標を用いて，推定できなかったメッシュの位置推定を行う．

実験スクリプト  expt_split_affinesim_combinable_with_interporation.py

## 機能について

### common
ディレクトリパス，時間計測などのコードがこの中に入っている．

- affine_base  
    こちらは，ASIFTのアルゴリズムを使っている．重要

### makedatabase
命名がわかりにくいが，こちらが主要なコードである．
次のコードが重要である．

- split_affinesim
- split_affinesim_combinable
- mesh_interpolation

上記以外は，ほとんど使っていない．わかりにくいので消してもいい．
問題が発生したらgitで戻って欲しい．

## その他
実際の実験はshスクリプトを実行する．
手法はexpt_から始まるスクリプトごとに異なる．
メンテナンスされているパイソンスクリプトは次のとおりである．

expt_split_affinesim
expt_mesh_interpolation
expt_split_affinesim_combinable
expt_split_affinesim_combinable_with_interporation

それ以外は，ちょい書きで作ったものなので動作保証はない．
トリッキーなことはしていないので，ソースコードを読んでもらえたらわかると思われる．

