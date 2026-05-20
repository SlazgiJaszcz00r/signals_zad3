import struct
import numpy as np
from signals import ContinousSignal, DiscreteSignal, translate, SIGNAL_DESCRIPTIONS, calculate_metrics


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
                print("1. Generuj nowy sygnał")
                print("2. Pokaż parametry statystyczne")
                print("3. Wykres / Histogram")
                print("4. Zapisz do pliku binarnego")
                print("5. Odczytaj z pliku binarnego")
                print("6. Operacje arytmetyczne na sygnale")
                print("7. Zmień częstotliwość próbkowania (fs)")
                print("8. Wykonaj Kwantyzację (Zad 2: Q1/Q2 + C1-C4)")
                print("9. Demo Aliasingu (Zad 2: A1-A3)")
                print("0. Wyjście")
                c = input("Wybór: ")
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
                c = input("Wybór: ")
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