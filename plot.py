import matplotlib.pyplot as plt
import numpy as np
import sys


def main():
    """
    This is the main function.
    """
    while True:
        try:
            print("プロットするデータを選んでください")
            print("0 -> loss(train data), 1 -> accuracy(train data)")
            mode = int(sys.stdin.readline())
            break
        except Exception as e:
            print("エラー: {0}".format(e))

    while True:
        try:
            # load file
            print("パラメータを保存してあるファイル名を入力して下さい.")
            print("読み込まない場合は何も入力せずに Enter を押してください")
            filename = str(sys.stdin.readline())
            filename = filename.replace("\n", "")
            filename = filename.replace("\r", "")
            tmp_name = filename.replace("param_", "")
            legend_name = tmp_name.replace(".npz", "")
            load_param = np.load(filename)
            itr_plot_x = load_param['loss'].size
            iteration = np.arange(0, itr_plot_x, 1)
            loss = load_param['loss']

            # plot loss
            if mode == 0:
                plt.plot(iteration, loss, label=legend_name, lw=0.5)
                plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=18)
                plt.title("cross entropy error")
                plt.grid(True)
                plt.xlabel("itr")
                plt.ylabel("error avg")

            elif mode == 1:
                # plot accuracy
                iteration_train = load_param['t_acc_itr']
                accuracy_train = load_param['t_acc']
                plt.plot(iteration_train, accuracy_train)
                plt.title("accuracy for train data")
                plt.grid(True)
                plt.xlabel("epoch")
                plt.ylabel("accuracy")

            # continue or exit
            print("続けて plot する場合は1, 終了する場合は0を入力してください.")
            cont = int(sys.stdin.readline())
            if cont == 0:
                plt.show()
                break

        except Exception as e:
            print("エラー: {0}".format(e))


if __name__ == "__main__":
    main()