
## ARマーカーを用いた位置情報の取得


### はじめに
カメラ画像からサンプル位置を自動推定する。
ARマーカーを4つ配置して、各ARマーカーの座標を読み取り画像サイズ（ピクセル）を設定し、射影変換をすることにより実座標系における位置計測を行う。

### 参考
[Python+OpenCVで、カメラ画像から机上の物体位置（実座標系）を計測してみる](https://qiita.com/code0327/items/c6e468da7007734c897f)



ARマーカ

[ARマーカー認識プログラム](https://qiita.com/hsgucci/items/37becbb8bfe04330ce14)

### 環境と環境構築
試した環境

Win10 Pro 64bit

Anaconda

Python3.7

#### モジュールインストール

opencv =version4以上  

```Python
conda install -c conda-forge opencv
pip install pyzbar

```
conda-forgeからインストールできます。
(conda-forgeからのインストールが設定されている場合は、-c conda-forge　は不要です。）

QRコードリーダーはpipからインストールしてください。
```Python
pip install pyzbar

```

### 使い方

ARマーカーを4つ配置して、各ARマーカーの座標を読み取り画像サイズ（ピクセル）を設定し、射影変換をすることにより実座標系における位置計測を行う。

ar_libに関数がまとまっています。

ArBase.image_show(image_file, posetime=2, image_save=False)

画像を表示する関数

ArBase.qr_reader(image_file, posetime=2, image_save=False)

OpenCVによるQRコードを読む関数（QRコードは複数あるとエラーになる）

ArBase.ar_marker_make(variety_num=4,output_dir='./marker')

ARマーカーを作成する関数

ArBase.ar_find_overay(image_file, posetime=2, image_save=False)

ARマーカーを見つけて表示する関数

ArBase.real_size_ar(image_file,output_path='./data',size=None,ar_cut_position='edge')

4つあるARマーカーから射影変換を行い真上から見たイメージに変換する関数

ArBase.real_size_ar_with_indicator(image_file,output_path='./data',size= (150,150),ar_cut_position='edge')

4つのARマーカーでサンプルの大きさを決定し、もう一つのARマーカーで測定位置推定する関数

テストコードは、test.ipynbファイルに記載しています。


### その他
QRコードについてはqr_libに関数がまとまっています。
