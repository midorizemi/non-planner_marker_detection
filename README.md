# 非平面マーカ検出
非平面マーカを検出し，マーカの形状を得るためにマーカをメッシュに分割する．
しかし，メッシュ上には画像特徴が十分でない場合がある．
そこで，メッシュ上の特徴点の数を周辺のメッシュ上の特徴点数と比較しメッシュを統合する．

実験スクリプト　　expt_split_affinesim_combinable.py

さらに，非平面なマーカはセルフオクルージョンが存在しメッシュの位置推定できない場合がある．
そこで，位置推定でいたメッシュの句形頂点の座標を用いて，推定できなかったメッシュの位置推定を行う．

実験スクリプト  expt_split_affinesim_combinable_with_interporation.py

## 実行環境
Python3.6をインストールしておくこと
PiPの最新バージョンをインストールしておくこと

`requirements` ファイルを用意したので，Linux環境であればターミナルを開いて
各自のシェル環境で下記を実行する．
Pythonスクリプト実行時にパッケージでエラーが出たら都度インストールする．
```
pip install -r requirements
```

### Windowsの場合
※ 注意 ※
この方法をとった場合，Anaconda等のほかディストリビューションのpythonと互換性はなくなります．
必ず，WSLターミナルからpythonを実行するようにしてください．

まずは，このページを参考にWSLを使えるようにする．
https://qiita.com/Aruneko/items/c79810b0b015bebf30bb
次に，プロキシ設定や日本語化，タイムゾーンを東京に合わせる．
参考ページ：（https://linuxfan.info/wsl-setup-guide#Ubuntu-3）

``` bash
domain=proxy.uec.ac.jp:8080
echo "export http_proxy=http://${domain}" >> ~/.bashrc
echo "export https_proxy=http://${domain}" >> ~/.bashrc
sudo sh -c "echo 'http_proxy=http://${domain}' >> /etc/environment"
sudo sh -c "echo 'https_proxy=http://${domain}' >> /etc/environment"
echo "proxy = proxy.uec.ac.jp:8080" > ~/.curlrc
sudo dpkg-reconfigure tzdata
sudo apt install -y language-pack-ja
sudo update-locale LANG=ja_JP.UTF-8
sudo apt-get update
```

続いて下記を実行する．

``` bash
sudo apt-get install python3 python3-dev python3-venv libsm6 libxrender1 libfontconfig1
cd /mnt/c/Users/<user-name>/<path to makedb directoy>
python3 -m venv --without-pip venv && cd $_
source bin/activate
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
```

ローカルディレクトリで独自のpython環境を構築する準備が整いました．
下記を実行してください．

``` bash
echo "alias venvact='source venv/bin/activate'" >> ~/.bashrc && source $_
cd /mnt/c/Users/<user-name>/<path to makedb directoy>
python3 -m venv venv
venvact
pip install -r requirements
```

## 実行
```bash
bash expt_split_affinesim_combinable_with_interporation.py
```

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
実際の実験はbashスクリプトを実行する．
手法はexpt_から始まるスクリプトごとに異なる．
メンテナンスされているパイソンスクリプトは次のとおりである．

expt_split_affinesim
expt_mesh_interpolation
expt_split_affinesim_combinable
expt_split_affinesim_combinable_with_interporation

それ以外は，ちょい書きで作ったものなので動作保証はない．
トリッキーなことはしていないので，ソースコードを読んでもらえたらわかると思われる．

## 最後に
わからないことがあったら，こちらのリポジトリにissueを投げてください．
https://github.com/midorizemi/non-planner_marker_detection