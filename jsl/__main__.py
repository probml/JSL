import fire
from scripts import ekf_vs_ukf as ekf_vs_ukf_exp

class Experiments:
    def ekf_vs_ukf(self):
        ekf_vs_ukf_exp.main()

if __name__ == "__main__":
    fire.Fire(Experiments)