# 画像認識 課題4

---

## 1. 課題内容

MNIST のテスト画像1 枚を入力とし，3 層ニューラルネットワークを用いて，0～9 の値のうち1 つを出力するプログラムを作成せよ．

---

## 2. 作成したプログラムについて

課題1 で作成した`my_nn_test.py`に改良を加えた。

課題1から追加、変更になった点は以下の通りである。

- 課題2, 3において作成した学習後のパラメータを記録したファイルを読み込み、識別を行う
- 標準入力から不正な値が入力された場合、ファイルが存在しなかった場合などの例外処理

はじめに、`main()`部分についての説明を行う。`main()`のソースコードは以下の コード1 の通りである。

```Python
def main():
    """
    This is the main function.
    """

    nn = NNTest()
    err_time = 0
    while True:
        if err_time >= 3:
            print("プログラムを終了します...")
            sys.exit(0)

        try:
            print("0以上9999以下の整数を1つ入力してください.")
            idx = int(sys.stdin.readline(), 10)

            if 0 <= idx < 10000:
                break
            else:
                err_time = err_time + 1
                print("Error: 0以上9999以下の整数ではありません")

        except Exception as e:
            err_time = err_time + 1
            print(e)

    # load parameter
    err_time = 0
    while True:
        try:
            if err_time >= 3:
                print("プログラムを終了します...")
                sys.exit(0)

            print("パラメータを保存してあるファイル名を入力して下さい.")
            print("読み込まない場合は何も入力せずに Enter を押してください")
            filename = str(sys.stdin.readline())
            filename = filename.replace('\n', '')
            filename = filename.replace('\r', '')
            if filename == "":
                print("パラメータをランダムに初期化してテストを行います")
                break
            load_param = np.load(filename)

        except Exception as e:
            print("エラー: {0}".format(e))
            err_time = err_time + 1

        else:
            nn.network['w1'] = load_param['w1']
            nn.network['w2'] = load_param['w2']
            nn.network['b1'] = load_param['b1']
            nn.network['b2'] = load_param['b2']
            break

    # forwarding
    forward_data = forward(nn, idx)
    y = np.argmax(forward_data['y'], axis=0)

    # -- for showing images --
    plt.imshow(nn.X[idx], cmap=cm.gray)
    print("Recognition result -> {0} \n Correct answer -> {1}".format(y, nn.Y[idx]))
    plt.show()
```
<center><small>コード1 main 関数部</small></center><br>

この`main()`内では主に以下のことを行なっている。

1. `NNTest` クラスインスタンスの作成
2. 標準入力から 0 以上 9999 以下の値を 1 つ受け取る。それ以外の値が入力された時はエラーを出力し、3回エラーを出力した場合はプログラムを終了する。
3. 学習したパラメータを利用して識別を行う場合は、ファイル名を標準入力から入力、課題1のようにランダムに初期化されたパラメータを用いて識別を行う場合は何も入力せずに進む。ファイルが存在しない場合はエラーとし、3回エラーを出力した場合はプロゴウラムを終了する。
4. パラメータをファイルから読み込んだ場合は、インスタンス作成時に初期化した各パラメータを読み込んだパラメータに書き換える。
5. 画像の識別を行い、結果を標準出力に出力する。

## 3. 実行結果

課題3で学習（600 * 30 epoch = 18000回)させたパラメータを保存したファイル `param_30epoch.npz` を標準入力から読み込ませ、0以上9999以下の値を何度か適当に選び、出力結果を確認した。

入力した整数と正解ラベル、および識別器の識別結果をまとめたものを以下の表1に示す。

<table align="center">
    <caption>表1 入力値、正解ラベル、認識結果</caption>
    <tr>
        <td>入力値</td>
        <td>正解ラベル</td>
        <td>認識結果</td>
    </tr>
    <tr>
        <td>0</td>
        <td>7</td>
        <td>7</td>
    </tr>
    <tr>
        <td>334</td>
        <td>3</td>
        <td>3</td>
    </tr>
    <tr>
        <td>777</td>
        <td>1</td>
        <td>1</td>
    </tr>
    <tr>
        <td>1000</td>
        <td>9</td>
        <td>9</td>
    </tr>
    <tr>
        <td>1729</td>
        <td>1</td>
        <td>1</td>
    </tr>
    <tr>
        <td>4126</td>
        <td>8</td>
        <td>8</td>
    </tr>
    <tr>
        <td>5887</td>
        <td>7</td>
        <td>0</td>
    </tr>
    <tr>
        <td>9999</td>
        <td>6</td>
        <td>6</td>
    </tr>
</table>

また、MNIST テストデータ10000枚すべてを用いて、ランダムに初期化されたパラメータと学習後のパラメータ（`param_30epoch.npz`）のそれぞれについて認識精度を計算したところ、以下の表2の結果が得られた。

<table align="center">
    <caption>表2 認識精度</caption>
    <tr>
        <td>学習前</td>
        <td>学習後</td>
    </tr>
    <tr>
        <td>8.92%</td>
        <td>91.1%</td>
    </tr>
</table>

表1、表2 よりプログラムが正しく動作していることと、パラメータがうまく学習できていると考えられる。

## 4. 工夫点、問題点

例外処理を行うことによって、ファイル名の入力ミスがあってもすぐにプログラムが終了しないようにした。また、誰がプログラムを実行してもスムーズに進められるように、メッセージを充実させたり、認識結果と正解ラベルとテスト画像をすべて表示することによって、正しい識別が行われたかどうかを確認しやすくしたりしている。