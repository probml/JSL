import os
import fire
import matplotlib.pyplot as plt
from . import demos

class Experiments:
    @staticmethod
    def _save_plots(figures, savefig_path, fmt):
        if savefig_path is None:
            return False
        for name, figure in figures.items():
            figpath = os.path.join(savefig_path, name)
            figpath = f"{figpath}.{fmt}"
            figure.savefig(figpath)
        return True

    def list_demos(self):
        list_demos = dir(demos)
        print("*" * 20)
        for name in list_demos:
            if "__" not in name and "util" not in name:
                print(name)

    def run_demo(self, demo_name, savefig_path=None, fmt="pdf"):
        """
        Run one of the demos specified in the `list_demos` function

        Parameters
        ----------
        demo_name: str
            An element of `list_demos`

        savig_path: [None, str]
            The relative or absolute path to save
            the resulting figures

        fmt: str
            The format to store the optional path
        """
        figures = getattr(demos, demo_name).main()
        self._save_plots(figures, savefig_path, fmt)
        plt.show()


if __name__ == "__main__":
    fire.Fire(Experiments)
