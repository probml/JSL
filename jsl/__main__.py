import os
import fire
from jsl.scripts import ekf_vs_ukf as ekf_vs_ukf_exp

class Experiments:
    def ekf_vs_ukf(self, savefig_path=None):
        figures = ekf_vs_ukf_exp.main()
        if savefig_path is not None:
            fig_names = ["nlds2d_data", "nlds2d_ekf", "nlds2d_ukf"]
            fig_names = [fig_name + ".pdf" for fig_name in fig_names]
            for fig, fig_name in zip(figures, fig_names):
                filename = os.path.join(savefig_path, fig_name)
                fig.savefig(filename)


if __name__ == "__main__":
    fire.Fire(Experiments)