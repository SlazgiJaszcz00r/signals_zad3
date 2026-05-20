from __future__ import annotations
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
        self.t1, self.fs, self.T,self.A,self.d = t1, fs, T,A,d
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
    
    @staticmethod
    def convolve_discrete(sig1: DiscreteSignal, sig2: DiscreteSignal) -> DiscreteSignal:
        if sig1.fs != sig2.fs:
            raise ValueError("Oba sygnały muszą mieć tę samą częstotliwość próbkowania.")

        # Pobranie próbek jako list (dla czytelności – można też użyć np.ndarray)
        x = sig1.samples  # długość N
        h = sig2.samples  # długość M
        N = len(x)
        M = len(h)
        L = N + M - 1      # długość splotu

        # Inicjalizacja listy wynikowej zerami
        y = [0.0] * L

        # Ręczne obliczenie splotu liniowego
        for n in range(L):
            suma = 0.0
            # Indeks dla sygnału x: k
            # Indeks dla sygnału h: n - k
            for k in range(max(0, n - M + 1), min(N, n + 1)):
                suma += x[k] * h[n - k]
            y[n] = suma

        y_arr = np.array(y)

        new_t1 = sig1.t1 + sig2.t1          # czas początkowy
        new_fs = sig1.fs                    # częstotliwość próbkowania
        new_d = L / new_fs                  # czas trwania

        # Utworzenie nowego sygnału dyskretnego
        return DiscreteSignal(
            type_name="Splot_ręczny",
            A=1,              
            t1=new_t1,
            d=new_d,
            fs=new_fs,
            samples=y_arr
        )
    @staticmethod
    def design_lowpass_filter(num_taps: int, cutoff_norm: float, window='hamming'):
        if num_taps <= 0:
            raise ValueError("Liczba współczynników musi być dodatnia.")
        if not (0 < cutoff_norm < 0.5):
            raise ValueError("Znormalizowana częstotliwość odcięcia musi być w (0, 0.5).")

        M = num_taps - 1          # rząd filtru
        n = np.arange(num_taps)   # indeksy współczynników
        # Idealna odpowiedź impulsowa filtru dolnoprzepustowego (przesunięta, by była przyczynowa)
        # fc_norm = cutoff_norm, idealne współczynniki: h_ideal[n] = 2*fc_norm * sinc(2*fc_norm * (n - M/2))
        center = M / 2.0
        h_ideal = np.zeros(num_taps)
        for i in range(num_taps):
            if i == center:
                # Sinc(0) = 1
                h_ideal[i] = 2 * cutoff_norm
            else:
                arg = 2 * cutoff_norm * (i - center)
                h_ideal[i] = 2 * cutoff_norm * np.sin(np.pi * arg) / (np.pi * arg)

        # Zastosowanie okna
        if window == 'rectangular':
            win = np.ones(num_taps)
        elif window == 'hamming':
            win = 0.54 - 0.46 * np.cos(2 * np.pi * n / M)
        else:
            raise ValueError("Dostępne okna: 'rectangular', 'hamming'.")

        h = h_ideal * win
        return h
    @staticmethod
    def design_bandpass_filter(num_taps: int, low_cut_norm: float, high_cut_norm: float, window='hamming'):
        if num_taps <= 0:
            raise ValueError("Liczba współczynników musi być dodatnia.")
        if not (0 < low_cut_norm < high_cut_norm < 0.5):
            raise ValueError("Częstotliwości muszą spełniać: 0 < low_cut_norm < high_cut_norm < 0.5")

        M = num_taps - 1          # rząd filtru
        n = np.arange(num_taps)   # indeksy współczynników
        center = M / 2.0

        # Idealna odpowiedź impulsowa filtru pasmowego (różnica dwóch dolnoprzepustowych)
        h_ideal = np.zeros(num_taps)
        for i in range(num_taps):
            if i == center:
                # Sinc(0) = 1
                h_ideal[i] = 2 * high_cut_norm - 2 * low_cut_norm
            else:
                arg_high = 2 * high_cut_norm * (i - center)
                arg_low  = 2 * low_cut_norm  * (i - center)
                sinc_high = np.sin(np.pi * arg_high) / (np.pi * arg_high)
                sinc_low  = np.sin(np.pi * arg_low)  / (np.pi * arg_low)
                h_ideal[i] = 2 * high_cut_norm * sinc_high - 2 * low_cut_norm * sinc_low

        # Zastosowanie okna
        if window == 'rectangular':
            win = np.ones(num_taps)
        elif window == 'hamming':
            win = 0.54 - 0.46 * np.cos(2 * np.pi * n / M)
        else:
            raise ValueError("Dostępne okna: 'rectangular', 'hamming'.")

        h = h_ideal * win
        return h
    
    def cross_correlation(self, other: 'DiscreteSignal', method: str = 'direct') -> 'DiscreteSignal':
        if self.fs != other.fs:
            raise ValueError("Obie sekwencje muszą mieć tę samą częstotliwość próbkowania")
        
        x = self.samples          # długość N
        h = other.samples         # długość M
        N, M = len(x), len(h)
        L = N + M - 1             # długość korelacji

        if method == 'direct':
            # ---------- WARIANT BEZPOŚREDNI ----------
            R = np.zeros(L)
            # Przesunięcie k (w próbkach) przechodzi od -(M-1) do (N-1)
            for k in range(L):
                shift = k - (M - 1)       # aktualne przesunięcie
                suma = 0.0
                for n in range(N):
                    idx_h = n + shift
                    if 0 <= idx_h < M:
                        suma += x[n] * h[idx_h]
                R[k] = suma
        elif method == 'conv':
            # ---------- WARIANT Z WYKORZYSTANIEM SPLOTU ----------
            # Korelacja: R[k] = (x * h_rev)[k], gdzie h_rev[n] = h[-n]
            h_rev = h[::-1]               # odwrócenie kolejności próbek
            # Tworzymy tymczasowy sygnał dla odwróconego h (parametry nieistotne dla próbek)
            h_rev_sig = DiscreteSignal(
                type_name=other.signal_type,
                A=other.A,
                t1=other.t1,
                d=other.d,
                fs=other.fs,
                samples=h_rev
            )
            # Splot liniowy x z h_rev
            conv_sig = DiscreteSignal.convolve_discrete(self, h_rev_sig)
            R = conv_sig.samples          # wynik splotu to właśnie korelacja (kolejność zgodna z L)
        else:
            raise ValueError("method musi być 'direct' lub 'conv'")

        # Określenie czasu początkowego nowego sygnału:
        # Dla przesunięcia k = -(M-1) czas = self.t1 - (M-1)/fs
        new_t1 = self.t1 - (M - 1) / self.fs
        new_d = L / self.fs

        return DiscreteSignal(
            type_name=f"Korelacja({self.signal_type},{other.signal_type})",
            A=1,                          # wartość symboliczna
            t1=new_t1,
            d=new_d,
            fs=self.fs,
            samples=R
        )

class RadarSensor:
    """
    Symulacja korelacyjnego czujnika odległości (np. radar impulsowy).
 
    Zasada działania (wg instrukcji do ćw. 3):
      Krok 1 – generuj ciągły, okresowy sygnał sondujący (złożony z co najmniej
               dwóch podstawowych sygnałów okresowych).
      Krok 2 – próbkuj oba sygnały (sondujący i zwrotny) z częstotliwością fs
               do buforów o tej samej długości buf_size.
      Krok 3 – co pewien okres dokonuj analizy korelacyjnej obu buforów.
      Krok 4 – wyznacz dyskretną funkcję korelacji wzajemnej R_yx, przejrzyj
               prawą połowę w poszukiwaniu maksimum.
      Krok 5 – na podstawie pozycji maksimum i okresu próbkowania Δt wyznacz
               opóźnienie czasowe t.
      Krok 6 – oblicz drogę S = V · t (sygnał przebył drogę tam i z powrotem).
      Krok 7 – podziel S przez 2 → chwilowa odległość czujnika od obiektu.
    """
 
    def __init__(
        self,
        signal_speed: float = 500.0,
        fs: float = 400.0,
        buf_size: int = 256,
        probe_period: float = None,
        noise_level: float = 0.1,
        report_interval: float = 0.5,
        max_distance: float = None,
    ):
        """
        Parametry
        ---------
        signal_speed    : prędkość rozchodzenia się sygnału [j.d./s]
        fs              : częstotliwość próbkowania [Hz]
        buf_size        : długość bufora w próbkach
        probe_period    : okres sygnału sondującego [s]; None = dobierany automatycznie
        noise_level     : poziom szumu odbicia jako ułamek amplitudy
        report_interval : co ile sekund symulowanego czasu raportować pomiar
        max_distance    : maksymalna mierzona odległość [j.d.]; None = auto z buf_size
 
        Ważne ograniczenie jednoznaczności pomiaru
        ------------------------------------------
        Opóźnienie sygnału zwrotnego (delay_samples = 2·d/v·fs) musi być krótsze
        niż jeden okres sygnału sondującego (period_samples = probe_period·fs).
        W przeciwnym razie korelacja wzajemna może zwrócić fałszywe maksimum
        w miejscu aliasu zamiast właściwego opóźnienia.
        Warunek:  probe_period >= 2 · max_distance / signal_speed
        """
        self.signal_speed = signal_speed
        self.fs = fs
        self.buf_size = buf_size
        self.noise_level = noise_level
        self.report_interval = report_interval
 
        self._dt = 1.0 / fs
        self._t_buf = buf_size / fs
 
        # Automatyczne dobranie max_distance i probe_period
        if max_distance is None:
            max_distance = (buf_size // 2) * signal_speed / (2.0 * fs)
        self.max_distance = max_distance
 
        min_period = 2.0 * max_distance / signal_speed
        if probe_period is None:
            probe_period = min_period * 1.2
        elif probe_period < min_period:
            print(f"[RadarSensor] UWAGA: probe_period={probe_period:.4f}s < "
                  f"min wymagany {min_period:.4f}s dla max_distance={max_distance:.1f}. "
                  f"Pomiar moze byc niejednoznaczny.")
 
        self.probe_period = probe_period
        self._period_samples = int(round(probe_period * fs))
        self._probe = self._generate_probe()
 
    # ------------------------------------------------------------------
    # Krok 1 – generowanie sygnału sondującego
    # Sygnał jest złożony z dwóch sinusoid (f1 i 2·f1), zgodnie z wymaganiem
    # użycia co najmniej dwóch podstawowych sygnałów okresowych.
    # ------------------------------------------------------------------
    def _generate_probe(self) -> np.ndarray:
        n = np.arange(self.buf_size)
        f1 = 1.0 / self.probe_period               # częstotliwość podstawowa [Hz]
        f2 = 2.0 * f1                              # druga harmoniczna
        # Normalizacja do zakresu [-1, 1]
        sig = np.sin(2 * np.pi * f1 * n * self._dt) + 0.5 * np.sin(2 * np.pi * f2 * n * self._dt)
        return sig / np.max(np.abs(sig))
 
    # ------------------------------------------------------------------
    # Krok 2 – symulacja sygnału zwrotnego
    # Sygnał sondujący jest przesunięty o opóźnienie wynikające z odległości,
    # a następnie dodawany jest szum gaussowski.
    # ------------------------------------------------------------------
    def _simulate_return(self, true_distance: float) -> np.ndarray:
        travel_time = 2.0 * true_distance / self.signal_speed   # czas tam i z powrotem
        delay_samples = int(round(travel_time * self.fs))
 
        # Cykliczne przesunięcie bufora (sygnał jest ciągły i okresowy)
        n = self.buf_size
        d = delay_samples % n
        returned = np.concatenate([self._probe[n - d:], self._probe[:n - d]])
 
        # Dodanie szumu gaussowskiego
        noise = np.random.normal(0, self.noise_level, self.buf_size)
        return returned + noise
 
    # ------------------------------------------------------------------
    # Kroki 3–5 – analiza korelacyjna
    # Używa DiscreteSignal.cross_correlation (metoda 'direct') z signals.py.
    # ------------------------------------------------------------------
    def _measure_once(self, true_distance: float):
        return_sig = self._simulate_return(true_distance)
 
        probe_ds = DiscreteSignal(
            type_name="probe", A=1, t1=0.0,
            d=self._t_buf, fs=self.fs, samples=self._probe.copy()
        )
        return_ds = DiscreteSignal(
            type_name="return", A=1, t1=0.0,
            d=self._t_buf, fs=self.fs, samples=return_sig
        )
 
        # Korelacja wzajemna R_yx (sygnał zwrotny względem sondującego)
        # Zgodnie z instrukcją (wzór 9): przesuwamy x względem y i szukamy max.
        corr_sig = probe_ds.cross_correlation(return_ds, method='direct')
        R = corr_sig.samples
        L = len(R)
 
        # Krok 4 – prawa połowa korelacji (opóźnienia >= 0)
        # Szukamy maksimum w oknie [1, period_samples], bo:
        #   • lag=0 to zawsze mocna korelacja sygnału ze sobą samym (nie opóźnienie),
        #   • lagi > period_samples to aliasy okresu (niejednoznaczne).
        half = L // 2
        right_half = R[half:]
 
        # Okno jednoznacznego pomiaru: od próbki 1 do period_samples
        window_end = min(self._period_samples, len(right_half) - 1)
        search_window = right_half[1:window_end + 1]
        peak_idx = int(np.argmax(search_window)) + 1   # +1 bo pomijamy lag=0
 
        # Krok 5 – czas opóźnienia
        delay_time = peak_idx * self._dt
 
        # Krok 6 & 7 – odległość = V · t / 2
        measured_distance = (delay_time * self.signal_speed) / 2.0
 
        return measured_distance, peak_idx, R, right_half
 
    # ------------------------------------------------------------------
    # Główna metoda symulacji
    # ------------------------------------------------------------------
    def run_simulation(
        self,
        object_distance,        # float lub callable(t) -> float
        sim_duration: float = 5.0,
        object_speed: float = 0.0,  # prędkość obiektu [j.d./s], gdy distance=float
        verbose: bool = True,
        plot: bool = True,
    ):
        """
        Symuluje działanie czujnika przez sim_duration sekund.
 
        Parametry
        ---------
        object_distance : float lub callable
            Stała odległość lub funkcja odległości od czasu t.
        sim_duration    : float
            Czas trwania symulacji [s].
        object_speed    : float
            Prędkość obiektu (używana gdy object_distance jest float).
        verbose         : bool
            Wypisuj wyniki kolejnych pomiarów na ekran.
        plot            : bool
            Rysuj wykres na końcu symulacji.
 
        Zwraca
        ------
        list of dict z kluczami: t, true_dist, meas_dist, error, lag_samples
        """
        results = []
        t = 0.0
        d0 = object_distance if callable(object_distance) else float(object_distance)
 
        if verbose:
            print(f"{'Czas [s]':>10} {'Rzeczywista [j.d.]':>20} "
                  f"{'Zmierzona [j.d.]':>18} {'Błąd [j.d.]':>13} {'Lag [próbki]':>14}")
            print("-" * 80)
 
        while t <= sim_duration:
            # Aktualna prawdziwa odległość
            if callable(object_distance):
                true_dist = object_distance(t)
            else:
                true_dist = d0 + object_speed * t
 
            true_dist = max(1e-3, true_dist)
 
            meas_dist, lag, R_full, R_right = self._measure_once(true_dist)
            error = abs(meas_dist - true_dist)
 
            results.append({
                "t": t,
                "true_dist": true_dist,
                "meas_dist": meas_dist,
                "error": error,
                "lag_samples": lag,
                "R_full": R_full,
                "R_right": R_right,
            })
 
            if verbose:
                print(f"{t:>10.3f} {true_dist:>20.3f} {meas_dist:>18.3f} "
                      f"{error:>13.4f} {lag:>14}")
 
            t += self.report_interval
 
        if plot:
            self._plot_results(results)
 
        return results
 
    # ------------------------------------------------------------------
    # Wizualizacja – ostatni pomiar + historia pomiarów
    # ------------------------------------------------------------------
    def _plot_results(self, results: list):
        last = results[-1]
        R = last["R_full"]
        L = len(R)
        half = L // 2
        lags = np.arange(L) - half
 
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        fig.suptitle("Korelacyjny czujnik odległości", fontsize=14)
 
        # 1) Sygnał sondujący
        axes[0].plot(self._probe, 'b-', lw=1)
        axes[0].set_title("Sygnał sondujący x(n)")
        axes[0].set_xlabel("Próbka")
        axes[0].set_ylabel("Amplituda")
        axes[0].grid(True, alpha=0.4)
 
        # 2) Sygnał zwrotny (ostatni pomiar)
        return_sig = self._simulate_return(last["true_dist"])
        axes[1].plot(return_sig, 'r-', lw=1)
        axes[1].set_title(f"Sygnał zwrotny y(n)  [szum={self.noise_level*100:.0f}%]")
        axes[1].set_xlabel("Próbka")
        axes[1].set_ylabel("Amplituda")
        axes[1].grid(True, alpha=0.4)
 
        # 3) Korelacja wzajemna – ostatni pomiar
        peak_abs = half + last["lag_samples"]
        axes[2].plot(lags, R, 'k-', lw=1, label="R_yx")
        axes[2].axvline(x=last["lag_samples"], color='g', lw=1.5, ls='--',
                        label=f"max @ lag={last['lag_samples']} → d≈{last['meas_dist']:.2f}")
        axes[2].axvline(x=0, color='gray', lw=0.8, ls=':')
        axes[2].scatter([last["lag_samples"]], [R[peak_abs]], color='g', zorder=5, s=60)
        axes[2].set_title("Korelacja wzajemna R_yx (ostatni pomiar)")
        axes[2].set_xlabel("Przesunięcie [próbki]")
        axes[2].set_ylabel("Korelacja")
        axes[2].legend(fontsize=9)
        axes[2].grid(True, alpha=0.4)
 
        # 4) Historia pomiarów
        ts = [r["t"] for r in results]
        true_ds = [r["true_dist"] for r in results]
        meas_ds = [r["meas_dist"] for r in results]
        axes[3].plot(ts, true_ds, 'b-o', ms=4, label="Rzeczywista odległość")
        axes[3].plot(ts, meas_ds, 'r--s', ms=4, label="Zmierzona odległość")
        axes[3].set_title("Historia pomiarów")
        axes[3].set_xlabel("Czas [s]")
        axes[3].set_ylabel("Odległość [j.d.]")
        axes[3].legend(fontsize=9)
        axes[3].grid(True, alpha=0.4)
 
        plt.tight_layout()
        plt.show()


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