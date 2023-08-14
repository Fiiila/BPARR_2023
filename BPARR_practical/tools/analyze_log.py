import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df_alg = pd.read_csv("../comparison/logs/logging_alg_validation.csv")
    df_nn = pd.read_csv("../comparison/logs/logging_nn_validation.csv")
    df_true = pd.read_csv("../comparison/logs/logging_validation_true.csv")

    # Time stages comparison
    columns = df_alg.columns[1:4]
    titles = ["Předzpracování", "Extrakce", "Následné zpracování"]
    fig, ax = plt.subplots(len(columns), 1, layout="tight", figsize=(7,6))
    for i, c in enumerate(columns):
        ax[i].plot(df_alg.index, df_alg[c], label=f"alg")
        ax[i].plot(df_nn.index, df_nn[c], label=f"nn")
        print(f"Alg-{c}-mean: {df_alg[c].mean()} s")
        print(f"Nn-{c}-mean: {df_nn[c].mean()} s")
        # ax[i].set_xlabel("snímek")
        # ax[i].set_ylabel("čas [s]")
        ax[i].set_title(f"{titles[i]}")
        ax[i].legend(loc="upper right")
    fig.suptitle("Porovnání výpočetních časů", fontsize=16)
    fig.supxlabel("číslo snímku")
    fig.supylabel("čas")
    plt.savefig("./validation_time.pdf", dpi=600, bbox_inches="tight", pad_inches=0.1, transparent=True)
    plt.show()

    # Extraction trajectory comparison
    columns = df_true.columns[1:5]
    titles = ["p_1", "p_2", "p_3", "p_4"]
    center_koeffs_nn = [(df_nn[c] + df_nn[f"{c[:-1]}r"]) / 2 for c in columns]
    center_koeffs_alg = [(df_alg[c] + df_alg[f"{c[:-1]}r"]) / 2 for c in columns]
    center_koeffs_true = [(df_true[c] + df_true[f"{c[:-1]}r"]) / 2 for c in columns]
    fig, ax = plt.subplots(len(columns), 1, layout="tight", figsize=(8,8))
    for i, c in enumerate(columns):
        ax[i].plot(df_alg.index, abs(center_koeffs_alg[i]-center_koeffs_true[i]), label=f"alg")
        ax[i].plot(df_nn.index, abs(center_koeffs_nn[i]-center_koeffs_true[i]), label=f"nn")
        print(f"Alg-{c}-mean: {abs(center_koeffs_alg[i]-center_koeffs_true[i]).mean()}")
        print(f"Nn-{c}-mean: {abs(center_koeffs_nn[i]-center_koeffs_true[i]).mean()}")
        # ax[i].plot(df_true.index, center_koeffs_true[i], label=f"true-{c}")
        # ax[i].plot(df_nn.index, df_nn[c], label=f"nn-{c}")
        # ax[i].plot(df_true.index, df_true[c], label=f"true-{c}")
        # ax[i].set_xlabel("snímek")
        # ax[i].set_ylabel("absolutní odchylka")
        # ax[i].set_title(f"{c[0].upper()}{c[1:]}")
        ax[i].set_title(f"{titles[i]}")
        ax[i].legend(loc="upper left")
    fig.suptitle("Chyba extrakce středové trajektorie", fontsize=16)
    fig.supxlabel("číslo snímku")
    fig.supylabel("absolutní odchylka")
    plt.savefig("./validation_trajectory.pdf", dpi=600, bbox_inches="tight", pad_inches=0.1, transparent=True)
    plt.show()
