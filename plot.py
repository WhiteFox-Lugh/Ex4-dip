import matplotlib.pyplot as plt
import numpy as np
import sys


def main():
    """
    This is the main function.
    """
    plot_time = 0
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
            filename = str(sys.stdin.readline())
            filename = filename.replace("\n", "")
            filename = filename.replace("\r", "")
            tmp_name = filename.replace("param_", "")
            legend_name = tmp_name.replace(".npz", "")
            load_param = np.load(filename)
            itr_plot_x = load_param['loss'].size
            iteration = np.arange(0, itr_plot_x, 1)
            loss = load_param['loss']
            # color name
            print("凡例のカラー名を入力して下さい.")
            print("r,g,b,c,m,y,k などが使えます. 何も入力しなければデフォルトになります.")
            color_name = sys.stdin.readline()
            color_name = color_name.replace('\n', '')
            color_name = color_name.replace('\r', '')

            # plot loss
            if mode == 0:
                if color_name == "":
                    plt.plot(iteration, loss, label=legend_name, lw=0.5)

                else:
                    plt.plot(iteration, loss, label=legend_name, color=color_name, lw=0.5)

                plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=18)
                plt.title("cross entropy error")
                plt.grid(True)
                plt.xlabel("itr")
                plt.ylabel("error avg")
                plt.ylim(0.0, 0.3)

            elif mode == 1:
                # plot accuracy
                iteration_train = load_param['t_acc_itr']
                accuracy_train = load_param['t_acc']
                plt.plot(iteration_train, accuracy_train, label=legend_name, lw=2.0)
                plt.legend(bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=1, fontsize=18)
                plt.title("accuracy for train data")
                plt.grid(True)
                plt.xlabel("epoch")
                plt.ylabel("accuracy")
                print("final accuracy -> {0}".format(accuracy_train[-1]))

            # continue or exit
            print("続けて plot する場合は1, 終了する場合は0を入力してください.")
            cont = int(sys.stdin.readline())
            if cont == 0:
                plt.show()
                break
            else:
                plot_time = plot_time + 1

        except Exception as e:
            print("エラー: {0}".format(e))


if __name__ == "__main__":
    main()