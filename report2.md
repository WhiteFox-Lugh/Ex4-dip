# 画像認識 課題2

---

## 1. 課題内容

[課題1] のコードをベースに，ミニバッチ（＝複数枚の画像）を入力可能とするように改良し，さらにクロスエントロピー誤差を計算するプログラムを作成せよ．

---

## 2. 作成したプログラムについて

作成したプログラムのファイルは`my_nn_learn.py`である。

課題1 で `my_nn_test.py` に `NNTest` というクラスを作成してある。このコードをベースに `NNLearn`というクラスを作成した。`NNLearn`クラスの中身は以下の コード1 のようになっている。

```Python
class NNLearn:
    """ Class of Neural Network (learning).

    Attributes:
        network: data and parameters in NN.

    """
    d = 28 * 28
    img_div = 255
    c = 10
    eta = 0.01
    m = 200
    batch_size = 100
    per_epoch = 60000 // batch_size
    epoch = 15
    p = ProgressBar()
    X, Y = mndata.load_training()
    X = np.array(X)
    X = X.reshape((X.shape[0], 28, 28))
    Y = np.array(Y)

    def __init__(self):
        self.network = {}
        w1_tmp = np.random.normal(0, math.sqrt(1 / self.d), self.m * self.d)
        self.network['w1'] = w1_tmp.reshape((self.m, self.d))
        w2_tmp = np.random.normal(0, math.sqrt(1 / self.m), self.c * self.m)
        self.network['w2'] = w2_tmp.reshape((self.c, self.m))
        b1_tmp = np.random.normal(0, math.sqrt(1 / self.d), self.m)
        self.network['b1'] = b1_tmp.reshape((self.m, 1))
        b2_tmp = np.random.normal(0, math.sqrt(1 / self.m), self.c)
        self.network['b2'] = b2_tmp.reshape((self.c, 1))

```
<center><small>コード1 NNLearn クラスのソースコード</small></center><br>

クラス変数については以下のように設定してある。

- `d` ... 画像の次元
- `img_div` ... 画素値を正規化するための変数。今回は白黒画像で画素値の範囲が 0~255 であるため、255 に設定してある。
- `c` ... クラス数。今回は 0~9の数字の識別を行うため、10 に設定してある。
- `eta` ... 学習係数(learning rate)。
- `m` ... 中間層のノード数。
- `batch_size` ... バッチサイズ。
- `per_epoch` ... 1 エポックあたりの学習回数。課題 3 で用いる。
- `epoch` ... エポック数。課題 3 で用いる。
- `p` ... 学習の進捗状況を表すプログレスバーであり、学習自体には関係ない。
- `X, Y` ... MNIST 学習データ。X には画像、Y には正解ラベルが格納される。

インスタンス変数については以下のように設定してある。

- `network` ... パラメータを格納するディクショナリ。課題1では重みのパラメータ`w1`,`w2`,`b1`,`b2`を格納している。初期値は課題1の仕様と同様である。

---

次に`main()`部分についての説明を行う。`main()`のソースコードは以下の コード2 の通りである。実際のソースコードにはのちの課題用にコメントアウトしている部分があるが割愛する。

```Python
def main():
    """
    This is the main function.
    """

    nn = NNLearn()
    nums = list(range(0, nn.X.size // nn.d))

    for itr in range(1):
        # init
        input_img = np.array([], dtype='int32')
        t_label = np.array([], dtype='int32')

        # select from training data
        choice_nums = np.random.choice(nums, nn.batch_size, replace=False)

        # data input
        for i in range(nn.batch_size):
            tmp_img = nn.X[choice_nums[i]]
            input_img = np.append(input_img, tmp_img)
            t_label = np.append(t_label, nn.Y[choice_nums[i]])

        nn.network['t_label'] = t_label
        nn.network['t_label_one_hot'] = one_hot_vector(t_label, nn.c)

        # forwarding
        forward_data = forward(nn, input_img)

        # print cross entropy
        print("average cross entropy -> {0}".format(forward_data['avg_entropy']))
```
<center><small>コード2 main 関数部</small></center><br>

`main()`内では主に以下のことを行なっている。

1. `NNLearn` クラスインスタンスの作成
2. バッチサイズの数だけ学習データからランダムに画像を選ぶ
3. 正解ラベルを one-hot vector 表現に変換する
4. 順伝播を行う
5. クロスエントロピー誤差の平均値を標準出力に出力

2 について、コード2 の `# init` から `# data input` 以下の for ループ内までが該当する。変数 `choice_nums` に指定したバッチサイズの数だけランダムに選んだ数字（重複はなし）を格納する。その後、`input_img` に 1 行 784 次元に直した画像データ、`t_label` に正解ラベルを格納する。

3 については、関数 `one_hot_vector()` に引数 `t_label` を渡すことで実装している。
関数 `one_hot_vector()` については以下の コード3 のようになっている。

```Python
def one_hot_vector(t: ndarray, c: int) -> ndarray:
    """ Make one-hot vector

    Args:
        t: correct label

    Returns:
        correct label (in one-hot vector expression)

    """
    return np.identity(c)[t]
```
<center><small>コード3 one_hot_vector 関数</small></center><br>

4 の順伝播については `forward()`部分が該当する。
`forward()`について課題 1 から変更、追加になった点を中心に説明する。
`forward()`内は以下の コード4 のようになっている。

```Python
def forward(nn: NNLearn, input_img: ndarray):
    """ Forwarding

    Args:
        nn: Class NNLearn
        input_img: selected images

    Returns:
        Dictionary data including the calculation result in each layer.

    """

    data_forward = {}

    # input_layer : (1, batch_size * d) -> (d = 784, batch_size)
    output_input_layer = input_layer(nn, input_img)
    # mid_layer : (d = 784, batch_size) -> (m, batch_size)
    a_mid_layer = affine_transformation(nn.network['w1'], output_input_layer, nn.network['b1'])
    z_mid_layer = mid_layer_activation(a_mid_layer)
    # output_layer : (m, batch_size) -> (c = 10, batch_size)
    a_output_layer = affine_transformation(nn.network['w2'], z_mid_layer, nn.network['b2'])
    result = output_layer_apply(a_output_layer)

    # find cross entropy
    entropy = cal_cross_entropy(nn, result, nn.network['t_label_one_hot'])
    entropy_average = np.sum(entropy, axis=0) / nn.batch_size

    data_forward['x1'] = output_input_layer
    data_forward['a1'] = a_mid_layer
    data_forward['z1'] = z_mid_layer
    data_forward['a2'] = a_output_layer
    data_forward['y'] = result
    data_forward['avg_entropy'] = entropy_average

    return data_forward
```
<center><small>コード4 forward 関数</small></center><br>

変数名と3層ニューラルネットワークの対応は以下の通りである。

- `output_input_layer` ... 入力層からの出力。784 行 `batch_size` 列の行列である。
- `a_mid_layer` ... 中間層でのアフィン変換後の値。`m` 行 `batch_size` 列の行列である。
- `z_mid_layer` ... `a_mid_layer` に活性化関数（課題 2 ではシグモイド関数）を適用したもの。`m` 行 `batch_size 列の行列である。
- `a_output_layer` ... 出力層でのアフィン変換後の値。
- `result` ... `a_output_layer` にソフトマックス関数を適用したもの。

中間層、出力層においてアフィン変換を適用する関数`affine_transformation()`、中間層においてシグモイド関数を適用する関数`mid_layer_activation()`、シグモイド関数`f_sigmoid()`、出力層においてソフトマックス関数を適用する関数`output_layer_apply()`、ソフトマックス関数`f_softmax()`の実装については、非常に単純な実装となっているため説明を割愛する。

クロスエントロピー誤差は `cal_cross_entropy()` において行っている。
この関数の実装は以下の コード5 の通りとなっている。

```Python
def cal_cross_entropy(nn: NNLearn, prob: ndarray, label: ndarray) -> ndarray:
    """ Calculate cross entropy

    Args:
        nn: NNLearn
        prob: output from output layer
        label: correct label (one-hot vector expression)

    Returns:
        cross entropy value of each recognition result

    """

    e = np.array([], dtype="float32")
    y_p = prob.T
    for j in range(nn.batch_size):
        tmp_e = 0
        for k in range(nn.c):
            tmp_e += (-label[j][k] * np.log(y_p[j][k]))
        e = np.append(e, tmp_e)
    return e
```
<center><small>コード5 cal_cross_entropy 関数</small></center><br>

この関数の引数 `prob` は出力層からの出力、`label` は正解ラベルを one-hot vector 表現にしたものである。変数`e`に各画像のクロスエントロピー誤差を格納し、これを返す。

`cal_cross_entropy()` で計算された各画像のクロスエントロピー誤差は、`forward()` 内の変数 `entropy` に格納される。変数 `entropy_average` にはクロスエントロピー誤差の平均が格納される。

順伝播のすべての計算が終わった後、各層における計算結果、およびクロスエントロピー誤差の平均を `data_forward` ディクショナリに格納し、これを`main()`に返す。

---

## 3. 実行結果

`my_nn_learn.py` を実行したところ、以下の 実行結果1 のようになった。

```zsh
average cross entropy -> 2.4316492330287436

Process finished with exit code 0
```
<center><small>実行結果1 my_nn_learn.py</small></center><br>

クロスエントロピー誤差の平均の値は、乱数のシード値を固定していないため実行するたびに変わる。
10 クラス識別器において、入力された画像をランダムに識別した場合、認識精度がおおよそ10%になると仮定すると、クロスエントロピーの平均は -log(0.1) = 2.3025850929940455 となると考えられる。今回の計算結果もおおよそこの値の近くとなっているため、正しく計算ができていると考えられる。

## 4. 工夫点、問題点

順伝搬において途中の計算結果をディクショナリ内に格納することで、変数の受け渡しを容易にした。またできる限り for 文を利用しないように心がけた。