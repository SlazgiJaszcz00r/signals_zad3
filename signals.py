import numpy as np
import matplotlib.pyplot as plt
import re

SIGNAL_DESCRIPTIONS = {
    "S1": "Szum o rozkładzie jednostajnym",
    "S2": "Szum gaussowski",
    "S3": "Sygnał sinusoidalny",
    "S4": "Sygnał sinusoidalny wyprostowany jednopołówkowo",
    "S5": "Sygnał sinusoidalny wyprostowany dwupołówkowo",
    "S6": "Sygnał prostokątny",
    "S7": "Sygnał prostokątny symetryczny",
    "S8": "Sygnał trójkątny",
    "S9": "Skok jednostkowy",
    "S10": "Impuls jednostkowy (delta Kroneckera)",
    "S11": "Szum impulsowy"
}


class Signal:
    def __init__(self, type_name):
        self.signal_type = type_name
        self.samples = None
        self.t = None

    def draw_plot(self, reconstructed_samples=None, reconstructed_t=None):
        plt.figure(figsize=(10, 4))
        plt.axhline(color='black', lw=0.5, ls='--')

        if reconstructed_samples is not None:
            plt.plot(reconstructed_t, reconstructed_samples, 'b-', label="Zrekonstruowany", alpha=0.8)
            plt.plot(self.t, self.samples, 'r.', label="Próbki", markersize=4)
            plt.legend()
        else:
            plt.plot(self.t, self.samples, 'r-' if len(self.samples) > 200 else 'ro-', markersize=2)

        plt.title(f"Sygnał: {translate(self.signal_type)}")
        plt.xlabel("t [s]")
        plt.ylabel("Amplituda")
        plt.grid(True)
        plt.show()

    def draw_hist(self, bins):
        if self.samples is None: return
        plt.figure(figsize=(10, 4))
        plt.hist(self.samples, bins, rwidth=0.9)
        plt.show()


class ContinousSignal(Signal):
    def __init__(self, type_name, A, t1, d, T=None, kw=None, ts=None):
        super().__init__(type_name)
        self.A, self.t1, self.d, self.T, self.kw, self.ts = A, t1, d, T, kw, ts

    def to_discrete(self, fs):
        n_samples = int(np.ceil(self.d * fs))
        t = self.t1 + np.arange(n_samples) / fs
        if self.signal_type == "S1":
            samples = np.random.uniform(-self.A, self.A, len(t))
        elif self.signal_type == "S2":
            samples = np.random.normal(0, self.A / 3, len(t))
        elif self.signal_type == "S3":
            samples = self.A * np.sin(2 * np.pi * (1 / self.T) * (t - self.t1))
        elif self.signal_type == "S4":
            s = np.sin((2 * np.pi / self.T) * (t - self.t1))
            samples = 0.5 * self.A * (s + np.abs(s))
        elif self.signal_type == "S5":
            samples = self.A * np.abs(np.sin((2 * np.pi / self.T) * (t - self.t1)))
        elif self.signal_type == "S6":
            samples = np.where(((t - self.t1) % self.T) < (self.kw * self.T), self.A, 0.0)
        elif self.signal_type == "S7":
            samples = np.where(((t - self.t1) % self.T) < (self.kw * self.T), self.A, -self.A)
        elif self.signal_type == "S8":
            term = (t - self.t1) % self.T
            samples = np.where(term < self.kw * self.T, (self.A / (self.kw * self.T)) * term,
                               (-self.A / (self.T * (1 - self.kw))) * (term - self.T))
        elif self.signal_type == "S9":
            samples = np.where(t >= self.ts, self.A, 0.0)
        else:
            samples = np.zeros_like(t)

        return DiscreteSignal(self.signal_type, self.A, self.t1, self.d, fs, samples=samples, T=self.T)


class DiscreteSignal(Signal):
    def __init__(self, type_name, A, t1, d, fs, p=None, samples=None, T=None):
        super().__init__(type_name)
        self.t1, self.fs, self.T = t1, fs, T
        if samples is not None:
            self.samples = np.array(samples)
        elif type_name == "S10":
            n = int(d * fs)
            self.samples = np.zeros(n)
            if 0 <= p < n: self.samples[p] = A
        elif type_name == "S11":
            self.samples = np.where(np.random.rand(int(d * fs)) < p, A, 0.0)
        self.t = t1 + np.arange(len(self.samples)) / fs

    def quantize(self, bits, method='round'):
        v_min, v_max = np.min(self.samples), np.max(self.samples)
        levels = 2 ** bits
        if v_min == v_max: return self.samples.copy()
        step = (v_max - v_min) / (levels - 1)

        if method == 'truncate':
            q_indices = np.floor((self.samples - v_min) / step)
            q_indices = np.clip(q_indices, 0, levels - 1)
            return v_min + q_indices * step
        else:  # round
            return v_min + np.round((self.samples - v_min) / step) * step

    def reconstruct(self, t_target, method='R1', kernel_size=10):
        ts = 1.0 / self.fs
        if method == 'R1':
            idx = np.floor((t_target - self.t1) * self.fs).astype(int)
            idx = np.clip(idx, 0, len(self.samples) - 1)
            return self.samples[idx]
        elif method == 'R2':  # FOH
            return np.interp(t_target, self.t, self.samples)
        elif method == 'R3':  # Sinc
            recon = np.zeros_like(t_target)
            for i, t_val in enumerate(t_target):
                n_mid = int((t_val - self.t1) * self.fs)
                n_start = max(0, n_mid - kernel_size)
                n_end = min(len(self.samples), n_mid + kernel_size)
                for n in range(n_start, n_end):
                    recon[i] += self.samples[n] * np.sinc((t_val - self.t1) / ts - n)
            return recon

    def calculate_parameters(self):
        x = self.samples
        if self.T and self.T > 0:
            spp = int(round(self.T * self.fs))
            if spp > 0: x = x[:(len(x) // spp) * spp]
        p = np.mean(x ** 2)
        return {"Średnia": np.mean(x), "Śr. Bezwzgl.": np.mean(np.abs(x)), "Wariancja": np.var(x), "Moc": p,
                "RMS": np.sqrt(p)}


def calculate_metrics(original, processed):
    mse = np.mean((original - processed) ** 2)
    snr = 10 * np.log10(np.sum(original ** 2) / np.sum((original - processed) ** 2)) if mse > 0 else 100
    psnr = 10 * np.log10(np.max(original) / mse) if mse > 0 else 100
    md = np.max(np.abs(original - processed))
    enob = (snr - 1.76) / 6.02
    return {"MSE": mse, "SNR": snr, "PSNR": psnr, "MD": md, "ENOB": enob}


def translate(text: str) -> str:
    sorted_keys = sorted(SIGNAL_DESCRIPTIONS.keys(), key=len, reverse=True)
    pattern = re.compile('|'.join(re.escape(k) for k in sorted_keys))
    return pattern.sub(lambda m: SIGNAL_DESCRIPTIONS[m.group(0)], text)