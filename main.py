import struct
import numpy as np
import matplotlib.pyplot as plt
from signals import ContinousSignal, DiscreteSignal,RadarSensor, translate, SIGNAL_DESCRIPTIONS, calculate_metrics


def validate_float(prompt, min_val=None, max_val=None, default=None):
    while True:
        try:
            val = input(prompt).strip()
            if not val and default is not None:
                return default
            f = float(val)
            if min_val is not None and f < min_val:
                print(f"Wartość nie może być mniejsza niż {min_val}.")
                continue
            if max_val is not None and f > max_val:
                print(f"Wartość nie może być większa niż {max_val}.")
                continue
            return f
        except ValueError:
            print("Proszę podać liczbę.")


def validate_int(prompt, min_val=None, max_val=None, default=None):
    while True:
        try:
            val = input(prompt).strip()
            if not val and default is not None:
                return default
            i = int(val)
            if min_val is not None and i < min_val:
                print(f"Wartość nie może być mniejsza niż {min_val}.")
                continue
            if max_val is not None and i > max_val:
                print(f"Wartość nie może być większa niż {max_val}.")
                continue
            return i
        except ValueError:
            print("Proszę podać liczbę całkowitą.")


def show_signal_list():
    print("\nDostępne kody sygnałów:")
    for code, desc in SIGNAL_DESCRIPTIONS.items():
        print(f"  {code}: {desc}")


class SignalApp:
    def __init__(self):
        self.active_signal = None
        self.fs = 1000.0  #domyślna częstotliwość

    def create_signal(self):
        show_signal_list()
        stype = input("Kod sygnału: ").upper()
        if stype not in SIGNAL_DESCRIPTIONS:
            print("Nieznany kod sygnału.")
            return None

        A = validate_float("Amplituda (A): ",min_val=0.0001)
        t1 = validate_float("Czas początkowy (t1): ")
        d = validate_float("Czas trwania (d): ", min_val=0.0001)

        if stype in ("S10", "S11"):
            fs = validate_float("Częstotliwość próbkowania (fs) [Hz]: ", min_val=0.1)
            if stype == "S11":
                p = validate_float("Prawdopodobieństwo (p) [0..1]: ", min_val=0, max_val=1)
                signal = DiscreteSignal(stype, A, t1, d, fs, p=p)
            else:
                # liczba próbek
                n_samples = int(d * fs)
                max_index = n_samples - 1
                if max_index < 0:
                    print("Błąd: czas trwania za krótki dla podanej fs.")
                    return None
                p = validate_int(f"Numer próbki skoku (0..{max_index}): ", min_val=0, max_val=max_index)
                signal = DiscreteSignal(stype, A, t1, d, fs, p=p)
            return signal

        T = None
        kw = None
        ts = None
        if stype in ("S3", "S4", "S5", "S6", "S7", "S8"):
            T = validate_float("Okres (T) [s]: ", min_val=0.0001)
        if stype in ("S6", "S7", "S8"):
            kw = validate_float("Wypełnienie (kw) [0..1]: ", min_val=0, max_val=1)
        if stype == "S9":
            ts = validate_float("Czas skoku (ts) [s]: ")
        signal = ContinousSignal(stype, A, t1, d, T, kw, ts)
        print(f"Utworzono sygnał {stype}: {SIGNAL_DESCRIPTIONS[stype]}")
        return signal

    def discretize_current(self):
        if self.active_signal is None:
            print("Brak aktywnego sygnału.")
            return
        if isinstance(self.active_signal, DiscreteSignal):
            print("Sygnał jest już dyskretny.")
            return
        if isinstance(self.active_signal, ContinousSignal):
            self.active_signal = self.active_signal.to_discrete(self.fs)
            print(f"Przekonwertowano na sygnał dyskretny z fs={self.fs} Hz.")
        else:
            print("Nieznany typ sygnału.")

    def save_binary(self):
        if not self.active_signal:
            print("Brak aktywnego sygnału.")
            return
        if isinstance(self.active_signal, ContinousSignal):
            print("Sygnał ciągły – najpierw dokonaj dyskretyzacji (opcja 3 w menu dla ciągłego).")
            return
        fname = input("Nazwa pliku (.bin): ")
        fs = validate_float("Częstotliwość próbkowania do zapisu [Hz]: ", min_val=0.1)
        samples = self.active_signal.samples
        t1 = self.active_signal.t1
        with open(fname, 'wb') as f:
            f.write(struct.pack('ddii', t1, fs, 0, len(samples)))
            for s in samples:
                f.write(struct.pack('d', s))
        print("Zapisano.")

    def load_binary(self):
        fname = input("Plik do odczytu: ")
        try:
            with open(fname, 'rb') as f:
                h = f.read(24)
                t1, fs, _, n = struct.unpack('ddii', h)
                data = [struct.unpack('d', f.read(8))[0] for _ in range(n)]
                self.active_signal = DiscreteSignal(f"Plik:{fname}", 0, t1, 0, fs, samples=data)
                print(f"Wczytano {n} próbek.")
        except Exception as e:
            print(f"Błąd: {e}")

    def change_fs(self):
        new_fs = validate_float(f"Podaj nową częstotliwość próbkowania [Hz] (obecna: {self.fs}): ", min_val=0.1)
        self.fs = new_fs
        print(f"Ustawiono fs = {self.fs} Hz.")

    def operation(self):
        if not self.active_signal:
            print("Brak aktywnego sygnału.")
            return
        if isinstance(self.active_signal, ContinousSignal):
            print("Sygnał ciągły – najpierw dokonaj dyskretyzacji.")
            return
        print("Dostępne operacje: +, -, *, /")
        o = input("Wybierz operację: ")
        if o not in ('+', '-', '*', '/'):
            print("Nieprawidłowa operacja.")
            return
        print("Tworzenie drugiego sygnału:")
        other = self.create_signal()
        if other is None:
            return
        # dyskretyzacja jesli któryś sygnał nie jest dyskretny
        if isinstance(other, ContinousSignal):
            other_disc = other.to_discrete(self.fs)
        else:
            other_disc = other
        current = self.active_signal
        if isinstance(current, ContinousSignal):
            current = current.to_discrete(self.fs)
        try:
            if o == '+':
                self.active_signal = current + other_disc
            elif o == '-':
                self.active_signal = current - other_disc
            elif o == '*':
                self.active_signal = current * other_disc
            elif o == '/':
                self.active_signal = current / other_disc
            print("Operacja wykonana.")
        except Exception as e:
            print(f"Błąd podczas operacji: {e}")

    def quantization_task(self):
        if not self.active_signal or not isinstance(self.active_signal, DiscreteSignal):
            print("Błąd: Musisz mieć aktywny sygnał dyskretny.")
            return

        bits = validate_int("Liczba bitów: ", 1, 16)
        print("Metoda: 1. Obcięcie (Q1), 2. Zaokrąglanie (Q2)")
        m = validate_int("Wybór: ", 1, 2)
        method = 'truncate' if m == 1 else 'round'

        q_samples = self.active_signal.quantize(bits, method)
        metrics = calculate_metrics(self.active_signal.samples, q_samples)

        print("\n--- Wyniki Kwantyzacji ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(self.active_signal.t, self.active_signal.samples, 'r--', label="Oryginał", alpha=0.6)
        plt.step(self.active_signal.t, q_samples, 'b', label=f"Skwantowany ({bits} bit)", where='post')
        plt.legend()
        plt.grid(True)
        plt.show()

    def sampling_reconstruction_task(self):
        if not self.active_signal or not isinstance(self.active_signal, ContinousSignal):
            print("Błąd: Ta operacja wymaga sygnału ciągłego jako wzorca.")
            return

        fs_high = self.fs * 10
        original_dense = self.active_signal.to_discrete(fs_high)
        sampled_discrete = self.active_signal.to_discrete(self.fs)

        print("Metoda rekonstrukcji: 1. ZOH (R1), 2. FOH (R2), 3. Sinc (R3)")
        m = validate_int("Wybór: ", 1, 3)
        method = {1: 'R1', 2: 'R2', 3: 'R3'}[m]

        recon = sampled_discrete.reconstruct(original_dense.t, method)
        metrics = calculate_metrics(original_dense.samples, recon)

        print(f"\n--- Wyniki Rekonstrukcji ({method}) ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        sampled_discrete.draw_plot(recon, original_dense.t)

    def aliasing_demo(self):
        print("\nWybierz wariant testowy:")
        print("1. f0=100Hz, fs=1000Hz")
        print("2. f0=440Hz, fs=22050Hz")
        print("3. f0=220Hz, fs=44100Hz")
        opt = validate_int("Wybór: ", 1, 3)

        f0, fs = {1: (100, 1000), 2: (440, 22050), 3: (220, 44100)}[opt]
        fd = validate_float(f"Podaj częstotliwość zakłócającą fd [Hz] (np. {fs + f0}): ")
        amp_d = validate_float("Amplituda sygnału zakłócającego (fd): ", min_val=0)

        t_end = 0.02
        t_analog = np.linspace(0, t_end, 1000)
        sig_analog = np.sin(2 * np.pi * f0 * t_analog) + amp_d * np.sin(2 * np.pi * fd * t_analog)

        t_sampled = np.arange(0, t_end, 1 / fs)
        sig_sampled = np.sin(2 * np.pi * f0 * t_sampled) + amp_d * np.sin(2 * np.pi * fd * t_sampled)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(t_analog, sig_analog, 'g', alpha=0.3, label="Sygnał wejściowy (f0 + fd)")
        plt.stem(t_sampled, sig_sampled, 'r', label=f"Próbki (fs={fs}Hz)")
        plt.title(f"Demonstracja Aliasingu (f0={f0}Hz, fd={fd}Hz)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def convolution_task(self):

        """Splot dwóch dowolnych sygnałów dyskretnych."""

        if not self.active_signal or not isinstance(self.active_signal, DiscreteSignal):

            print("Błąd: wymagany aktywny sygnał dyskretny.")

            return



        print("\nUtwórz drugi sygnał (h) do splotu z aktywnym sygnałem (x):")

        other = self.create_signal()

        if other is None:

            return

        if isinstance(other, ContinousSignal):

            other = other.to_discrete(self.fs)



        result = DiscreteSignal.convolve_discrete(self.active_signal, other)



        print(f"\nSplot wykonany. Wynikowy sygnał ma {len(result.samples)} próbek.")

        print("Co zrobić z wynikiem?")

        print("1. Ustaw jako aktywny sygnał")

        print("2. Tylko wyświetl wykres")

        c = input("Wybór [1/2]: ").strip()



        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=False)

        axes[0].plot(self.active_signal.t, self.active_signal.samples, 'b-o', ms=3, lw=1)

        axes[0].set_title(f"x – sygnał wejściowy ({self.active_signal.signal_type})")

        axes[0].grid(True, alpha=0.4)

        axes[1].plot(other.t, other.samples, 'g-o', ms=3, lw=1)

        axes[1].set_title(f"h – odpowiedź impulsowa ({other.signal_type})")

        axes[1].grid(True, alpha=0.4)

        axes[2].plot(result.t, result.samples, 'r-', lw=1)

        axes[2].set_title("y = h * x – sygnał po splocie")

        axes[2].grid(True, alpha=0.4)

        plt.tight_layout()

        plt.show()



        if c == '1':

            self.active_signal = result

            print("Wynik splotu ustawiony jako aktywny sygnał.")



    # ------------------------------------------------------------------ #

    #  ZADANIE 3 – Filtracja                                               #

    # ------------------------------------------------------------------ #



    def _pick_window(self) -> str:

        print("Okno: 1. Prostokątne  2. Hamminga  3. Hanninga  4. Blackmana")

        w = validate_int("Wybór: ", 1, 4)

        return {1: 'rectangular', 2: 'hamming', 3: 'hanning', 4: 'blackman'}[w]



    def _apply_window(self, h: np.ndarray, window: str, M: int) -> np.ndarray:

        """Ręczne okno Hanninga i Blackmana (nie ma ich w signals.py)."""

        n = np.arange(M)

        if window == 'rectangular':

            return h

        elif window == 'hamming':

            return h  # design_lowpass_filter stosuje hamming wewnętrznie

        elif window == 'hanning':

            win = 0.5 - 0.5 * np.cos(2 * np.pi * n / (M - 1))

            # przelicz h_ideal i zastosuj okno Hanninga ręcznie

            return self._recompute_with_window(M, h, win)

        elif window == 'blackman':

            win = 0.42 - 0.5 * np.cos(2 * np.pi * n / (M - 1)) + 0.08 * np.cos(4 * np.pi * n / (M - 1))

            return self._recompute_with_window(M, h, win)

        return h



    @staticmethod

    def _recompute_with_window(M: int, h_hamming: np.ndarray, new_win: np.ndarray) -> np.ndarray:

        """Odtwarza h_ideal z h_hamming (przez podział przez okno Hamminga), potem mnoży przez new_win."""

        n = np.arange(M)

        ham = 0.54 - 0.46 * np.cos(2 * np.pi * n / (M - 1))

        ham = np.where(np.abs(ham) < 1e-10, 1e-10, ham)

        h_ideal = h_hamming / ham

        return h_ideal * new_win



    def _design_filter_coeffs(self, ftype: str) -> tuple:

        """Zwraca (h, fs_ref) – współczynniki filtru i fs używaną do normalizacji."""

        M = validate_int("Liczba współczynników filtru (M, nieparzysta): ", min_val=3)

        if M % 2 == 0:

            M += 1

            print(f"  Zaokrąglono do M={M} (wymagana nieparzysta).")



        fs_ref = self.active_signal.fs if isinstance(self.active_signal, DiscreteSignal) else self.fs

        print(f"  Częstotliwość próbkowania sygnału: {fs_ref} Hz")



        window = self._pick_window()



        if ftype == 'lowpass':

            fc = validate_float(f"Częstotliwość odcięcia fc [Hz] (0 < fc < {fs_ref/2:.1f}): ",

                                min_val=0.001, max_val=fs_ref / 2 - 0.001)

            fc_norm = fc / fs_ref

            h = DiscreteSignal.design_lowpass_filter(M, fc_norm, window='hamming')

            h = self._apply_window(h, window, M)



        elif ftype == 'highpass':

            fc = validate_float(f"Częstotliwość odcięcia fc [Hz] (0 < fc < {fs_ref/2:.1f}): ",

                                min_val=0.001, max_val=fs_ref / 2 - 0.001)

            fc_norm = fc / fs_ref

            h_lp = DiscreteSignal.design_lowpass_filter(M, fc_norm, window='hamming')

            h_lp = self._apply_window(h_lp, window, M)

            # Przekształcenie do górnoprzepustowego: h_hp[n] = (-1)^n * h_lp[n]

            n = np.arange(M)

            h = h_lp * ((-1) ** n)



        elif ftype == 'bandpass':

            print(f"  Podaj dwie częstotliwości odcięcia (0 < fd < fg < {fs_ref/2:.1f} Hz):")

            fd = validate_float("  fd (dolna) [Hz]: ", min_val=0.001, max_val=fs_ref / 2 - 0.001)

            fg = validate_float(f"  fg (górna) [Hz] (> {fd}): ", min_val=fd + 0.001, max_val=fs_ref / 2 - 0.001)

            fd_norm = fd / fs_ref

            fg_norm = fg / fs_ref

            h = DiscreteSignal.design_bandpass_filter(M, fd_norm, fg_norm, window='hamming')

            h = self._apply_window(h, window, M)



        else:

            raise ValueError(f"Nieznany typ filtru: {ftype}")



        return h, fs_ref, M



    def _plot_filter_and_result(self, h: np.ndarray, filtered: 'DiscreteSignal',

                                original: 'DiscreteSignal', fs_ref: float, title: str):

        N_fft = 1024

        H = np.fft.rfft(h, n=N_fft)

        freqs = np.fft.rfftfreq(N_fft, d=1.0 / fs_ref)

        H_mag = np.abs(H)

        H_dB = 20 * np.log10(np.where(H_mag < 1e-12, 1e-12, H_mag))



        fig, axes = plt.subplots(4, 1, figsize=(12, 11))

        fig.suptitle(title, fontsize=13)



        axes[0].stem(np.arange(len(h)), h, markerfmt='C0.', linefmt='C0-', basefmt='k-')

        axes[0].set_title("Odpowiedź impulsowa filtru h(n)")

        axes[0].set_xlabel("Próbka")

        axes[0].grid(True, alpha=0.4)



        axes[1].plot(freqs, H_mag, 'b-', lw=1.2)

        axes[1].set_title("|H(f)| – moduł transmitancji")

        axes[1].set_xlabel("Częstotliwość [Hz]")

        axes[1].set_ylabel("|H|")

        axes[1].grid(True, alpha=0.4)



        axes[2].plot(freqs, H_dB, 'b-', lw=1.2)

        axes[2].set_title("|H(f)| [dB]")

        axes[2].set_xlabel("Częstotliwość [Hz]")

        axes[2].set_ylabel("dB")

        axes[2].set_ylim(bottom=-100)

        axes[2].grid(True, alpha=0.4)



        axes[3].plot(original.t, original.samples, 'r-', lw=0.8, alpha=0.6, label="Oryginalny")

        axes[3].plot(filtered.t[:len(original.samples)], filtered.samples[:len(original.samples)],

                     'b-', lw=1, label="Po filtracji")

        axes[3].set_title("Sygnał przed i po filtracji")

        axes[3].set_xlabel("t [s]")

        axes[3].legend()

        axes[3].grid(True, alpha=0.4)



        plt.tight_layout()

        plt.show()



    def filtration_task(self):

        """Projektowanie filtru SOI i filtracja aktywnego sygnału dyskretnego."""

        if not self.active_signal or not isinstance(self.active_signal, DiscreteSignal):

            print("Błąd: wymagany aktywny sygnał dyskretny.")

            return



        print("\nTyp filtru:")

        print("1. Dolnoprzepustowy (LP)")

        print("2. Górnoprzepustowy (HP)")

        print("3. Pasmowy (BP)")

        ft = validate_int("Wybór: ", 1, 3)

        ftype = {1: 'lowpass', 2: 'highpass', 3: 'bandpass'}[ft]



        h, fs_ref, M = self._design_filter_coeffs(ftype)



        # Filtracja = splot sygnału z odpowiedzią impulsową

        h_sig = DiscreteSignal(

            type_name=f"filtr_{ftype}",

            A=1, t1=0.0, d=M / fs_ref, fs=fs_ref, samples=h

        )

        filtered = DiscreteSignal.convolve_discrete(self.active_signal, h_sig)



        type_names = {1: "Dolnoprzepustowy", 2: "Górnoprzepustowy", 3: "Pasmowy"}

        self._plot_filter_and_result(h, filtered, self.active_signal, fs_ref,

                                     f"Filtr {type_names[ft]} (M={M})")



        c = input("\nUstawić przefiltrowany sygnał jako aktywny? [t/n]: ").lower()

        if c == 't':

            self.active_signal = filtered

            print("Przefiltrowany sygnał ustawiony jako aktywny.")



    # ------------------------------------------------------------------ #

    #  ZADANIE 3 – Korelacja                                               #

    # ------------------------------------------------------------------ #



    def correlation_task(self):

        """Korelacja wzajemna aktywnego sygnału z wybranym drugim sygnałem."""

        if not self.active_signal or not isinstance(self.active_signal, DiscreteSignal):

            print("Błąd: wymagany aktywny sygnał dyskretny.")

            return



        print("\nUtwórz drugi sygnał do korelacji wzajemnej:")

        other = self.create_signal()

        if other is None:

            return

        if isinstance(other, ContinousSignal):

            other = other.to_discrete(self.fs)



        print("\nMetoda obliczania korelacji:")

        print("1. Bezpośrednia (wg wzoru z instrukcji)")

        print("2. Z użyciem splotu")

        m = validate_int("Wybór: ", 1, 2)

        method = 'direct' if m == 1 else 'conv'



        corr = self.active_signal.cross_correlation(other, method=method)

        L = len(corr.samples)

        half = L // 2

        peak_abs = int(np.argmax(np.abs(corr.samples)))

        peak_lag = peak_abs - half

        print(f"\nKorelacja obliczona ({method}). Długość: {L} próbek.")

        print(f"Maksimum |R| przy przesunięciu: {peak_lag} próbek "

              f"({peak_lag / self.active_signal.fs:.4f} s)")



        lags = np.arange(L) - half

        fig, axes = plt.subplots(3, 1, figsize=(11, 9))

        axes[0].plot(self.active_signal.t, self.active_signal.samples, 'b-', lw=1)

        axes[0].set_title(f"Sygnał x – {self.active_signal.signal_type}")

        axes[0].grid(True, alpha=0.4)

        axes[1].plot(other.t, other.samples, 'g-', lw=1)

        axes[1].set_title(f"Sygnał h – {other.signal_type}")

        axes[1].grid(True, alpha=0.4)

        axes[2].plot(lags, corr.samples, 'r-', lw=1, label="R_hx")

        axes[2].axvline(x=peak_lag, color='navy', ls='--', lw=1.2,

                        label=f"max @ lag={peak_lag}")

        axes[2].axvline(x=0, color='gray', ls=':', lw=0.8)

        axes[2].scatter([peak_lag], [corr.samples[peak_abs]], color='navy', zorder=5, s=50)

        axes[2].set_title("Korelacja wzajemna R_hx")

        axes[2].set_xlabel("Przesunięcie [próbki]")

        axes[2].legend()

        axes[2].grid(True, alpha=0.4)

        plt.tight_layout()

        plt.show()



        c = input("Ustawić wynik korelacji jako aktywny sygnał? [t/n]: ").lower()

        if c == 't':

            self.active_signal = corr

            print("Korelacja ustawiona jako aktywny sygnał.")



    # ------------------------------------------------------------------ #

    #  ZADANIE 3 – Czujnik odległości                                      #

    # ------------------------------------------------------------------ #



    def radar_sensor_task(self):

        """Interaktywna konfiguracja i uruchomienie korelacyjnego czujnika odległości."""

        print("\n--- Konfiguracja korelacyjnego czujnika odległości ---")

        print("Parametry ośrodka i obiektu:")

        signal_speed = validate_float("  Prędkość sygnału [j.d./s]: ", min_val=0.001, default=500.0)

        max_distance = validate_float("  Maksymalna mierzona odległość [j.d.]: ", min_val=0.001, default=120.0)



        print("\nParametry czujnika:")

        fs_r = validate_float("  Częstotliwość próbkowania [Hz]: ", min_val=1.0, default=400.0)

        buf_size = validate_int("  Długość bufora [próbki]: ", min_val=32, default=256)

        noise = validate_float("  Poziom szumu odbicia [0..1]: ", min_val=0.0, max_val=1.0, default=0.05)

        rep = validate_float("  Okres raportowania [s]: ", min_val=0.001, default=0.5)



        print("\nRuch obiektu:")

        print("  1. Stała odległość")

        print("  2. Ruch jednostajny (stała prędkość)")

        print("  3. Ruch sinusoidalny")

        mo = validate_int("  Wybór: ", 1, 3)



        if mo == 1:

            d0 = validate_float("  Odległość obiektu [j.d.]: ", min_val=0.001, default=80.0)

            dist_fn = d0

            obj_speed = 0.0

        elif mo == 2:

            d0 = validate_float("  Odległość początkowa [j.d.]: ", min_val=0.001, default=100.0)

            obj_speed = validate_float("  Prędkość obiektu [j.d./s] (ujemna = zbliżanie): ", default=-10.0)

            dist_fn = d0

        else:

            d0 = validate_float("  Środkowa odległość [j.d.]: ", min_val=0.001, default=80.0)

            amp = validate_float("  Amplituda oscylacji [j.d.]: ", min_val=0.0, default=30.0)

            freq = validate_float("  Częstotliwość oscylacji [Hz]: ", min_val=0.001, default=0.2)

            dist_fn = lambda t, _d=d0, _a=amp, _f=freq: _d + _a * np.sin(2 * np.pi * _f * t)

            obj_speed = 0.0



        sim_dur = validate_float("Czas trwania symulacji [s]: ", min_val=0.001, default=5.0)



        sensor = RadarSensor(

            signal_speed=signal_speed,

            fs=fs_r,

            buf_size=buf_size,

            max_distance=max_distance,

            noise_level=noise,

            report_interval=rep,

        )

        print(f"\n  probe_period = {sensor.probe_period:.4f} s  "

              f"({sensor._period_samples} próbek)\n")



        if mo == 2:

            results = sensor.run_simulation(dist_fn, sim_duration=sim_dur,

                                            object_speed=obj_speed, verbose=True, plot=True)

        else:

            results = sensor.run_simulation(dist_fn, sim_duration=sim_dur,

                                            verbose=True, plot=True)



        avg_err = sum(r["error"] for r in results) / len(results) if results else 0

        print(f"\nŚredni błąd pomiaru: {avg_err:.4f} j.d.")



    def run(self):

        while True:

            if self.active_signal is None:

                act_desc = "Brak"

                is_discrete = False

            else:

                code = self.active_signal.signal_type

                desc = translate(code)

                act_desc = f"{code} ({desc})"

                is_discrete = isinstance(self.active_signal, DiscreteSignal)



            print(f"\n--- AKTYWNY SYGNAŁ: {act_desc} (fs dyskretyzacji = {self.fs} Hz) ---")



            if is_discrete:

                print("1.  Generuj nowy sygnał")

                print("2.  Pokaż parametry statystyczne")

                print("3.  Wykres / Histogram")

                print("4.  Zapisz do pliku binarnego")

                print("5.  Odczytaj z pliku binarnego")

                print("6.  Operacje arytmetyczne na sygnale")

                print("7.  Zmień częstotliwość próbkowania (fs)")

                print("8.  Wykonaj Kwantyzację (Zad 2: Q1/Q2 + C1-C4)")

                print("9.  Demo Aliasingu (Zad 2: A1-A3)")

                print("--- Zadanie 3 ---")

                print("10. Splot dyskretny (Zad 3)")

                print("11. Filtracja SOI – LP/HP/BP (Zad 3)")

                print("12. Korelacja wzajemna (Zad 3)")

                print("13. Czujnik odległości – symulacja (Zad 3)")

                print("0.  Wyjście")

                c = input("Wybór: ").strip()

                if c == '1':

                    new_sig = self.create_signal()

                    if new_sig:

                        self.active_signal = new_sig

                elif c == '2':

                    params = self.active_signal.calculate_parameters()

                    for k, v in params.items():

                        print(f"{k}: {v:.4f}")

                elif c == '3':

                    p = input("Typ wykresu: w (funkcja) czy h (histogram): ").lower()

                    if p == 'w':

                        self.active_signal.draw_plot()

                    elif p == 'h':

                        bins = validate_int("Liczba przedziałów histogramu: ", min_val=1)

                        self.active_signal.draw_hist(bins)

                    else:

                        print("Nieprawidłowy wybór.")

                elif c == '4':

                    self.save_binary()

                elif c == '5':

                    self.load_binary()

                elif c == '6':

                    self.operation()

                elif c == '7':

                    self.change_fs()

                elif c == '8':

                    self.quantization_task()

                elif c == '9':

                    self.aliasing_demo()

                elif c == '10':

                    self.convolution_task()

                elif c == '11':

                    self.filtration_task()

                elif c == '12':

                    self.correlation_task()

                elif c == '13':

                    self.radar_sensor_task()

                elif c == '0':

                    break

                else:

                    print("Nieprawidłowy wybór.")

            else:

                print("1. Generuj nowy sygnał")

                print("2. Odczytaj sygnał z pliku binarnego")

                if self.active_signal is not None and isinstance(self.active_signal, ContinousSignal):

                    print("3. Dyskretyzuj aktywny sygnał ciągły (użyj bieżącej fs)")

                    print("4. Test Rekonstrukcji (Zad 2: R1-R3 + C1-C4)")

                print("7. Zmień częstotliwość próbkowania (fs)")

                print("0. Wyjście")

                c = input("Wybór: ").strip()

                if c == '1':

                    new_sig = self.create_signal()

                    if new_sig:

                        self.active_signal = new_sig

                elif c == '2':

                    self.load_binary()

                elif c == '3' and self.active_signal is not None:

                    self.discretize_current()

                elif c == '4' and self.active_signal is not None:

                    self.sampling_reconstruction_task()

                elif c == '7':

                    self.change_fs()

                elif c == '0':

                    break

                else:

                    print("Nieprawidłowy wybór.")






if __name__ == "__main__":
    SignalApp().run()