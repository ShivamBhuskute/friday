#!/usr/bin/env python3
"""
Analyse audio windows dumped by friday_kws.cpp (or any wav).

  python analyse_dump.py /dev/ttyACM0 --dataset dataset          # capture from ESP (close idf.py monitor first)
  python analyse_dump.py capture.log                              # re-parse a saved log
  python analyse_dump.py clip.wav [--sweep] [--hpf 80]            # score one wav (e.g. a training clip)

Options:
  --hpf HZ   also score after a zero-phase Butterworth high-pass at HZ (tests whether the <100 Hz rumble hurts)
  --sweep    (wav mode) score every 120 ms hop of the whole clip, like live streaming

Model scoring runs in child processes so a native TensorFlow crash cannot kill the capture.
"""
import os, re, sys, glob, random, argparse, subprocess, threading, queue
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt

SR, N_SAMPLES = 16000, 16000
WINDOW_SIZE, STRIDE, FFT_SIZE, MEL_BINS = 480, 320, 512, 40
FMIN, FMAX = 60.0, 7800.0
BANDS = [(0, 100), (100, 500), (500, 2000), (2000, 4000), (4000, 8000)]

_mel = None
def mel_weights():
    global _mel
    if _mel is None:
        import librosa
        _mel = librosa.filters.mel(sr=SR, n_fft=FFT_SIZE, n_mels=MEL_BINS, fmin=FMIN, fmax=FMAX)
    return _mel

def extract_features(a):
    """Identical to train.py extract_features(). Returns (features, rms, std_val, gate)."""
    a = a.astype(np.float32)
    a = a - np.mean(a)
    rms = float(np.sqrt(np.mean(a ** 2)))
    zeros = np.zeros((49, MEL_BINS, 1), np.float32)
    if rms < 0.008:
        return zeros, rms, None, "RMS gate"
    if len(a) < N_SAMPLES:
        a = np.pad(a, (0, N_SAMPLES - len(a)))
    a = a[:N_SAMPLES]
    frames = np.zeros((49, WINDOW_SIZE), np.float32)
    for i in range(49):
        frames[i] = a[i * STRIDE: i * STRIDE + WINDOW_SIZE]
    frames = frames * np.hanning(WINDOW_SIZE + 1)[:-1]
    spec = np.abs(np.fft.rfft(frames, n=FFT_SIZE))
    lm = np.log10(np.maximum(np.dot(spec, mel_weights().T), 1e-5))
    sd = float(np.std(lm))
    if sd < 0.15:
        return zeros, rms, sd, "STD gate"
    lm = (lm - np.mean(lm)) / sd
    return np.expand_dims(lm.astype(np.float32), -1), rms, sd, "ok"

# ---------- audio statistics ----------
def band_stats(a):
    a = a.astype(np.float64)
    a = a - a.mean()
    P = np.abs(np.fft.rfft(a * np.hanning(len(a)))) ** 2
    f = np.fft.rfftfreq(len(a), 1.0 / SR)
    tot = P.sum() + 1e-12
    fr = [100.0 * P[(f >= lo) & (f < hi)].sum() / tot for lo, hi in BANDS]
    k = int(P.argmax())
    return fr, float(f[k]), 100.0 * P[max(k - 2, 0):k + 3].sum() / tot

def low_profile(x):
    sos = butter(4, 100, btype="low", fs=SR, output="sos")
    y = sosfiltfilt(sos, x.astype(np.float64) - x.mean())
    return np.array([1000 * np.sqrt(np.mean(y[i * 1600:(i + 1) * 1600] ** 2)) for i in range(10)])

def edge_dc(x):
    return 1000 * float(np.mean(x[:1600])), 1000 * float(np.mean(x[-1600:]))

def fmt_bands(fr):
    return " | ".join(f"{lo}-{hi}Hz {v:4.1f}%" for (lo, hi), v in zip(BANDS, fr))

def fmt_prof(p):
    return " ".join(f"{v:5.1f}" for v in p)

def highpass(x, hz):
    sos = butter(4, hz, btype="high", fs=SR, output="sos")
    return sosfiltfilt(sos, x.astype(np.float64) - np.mean(x))

# ---------- model scoring (child process) ----------
def _load_tflite(model_path):
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=model_path)
    it.allocate_tensors()
    return it

def _run_tflite(it, q):
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    it.set_tensor(inp["index"], q.reshape(1, 49, MEL_BINS, 1).astype(np.int8))
    it.invoke()
    o = int(it.get_tensor(out["index"])[0][0])
    s2, zp2 = out["quantization"]
    return (o - zp2) * s2

def _quantize(it, feat):
    s, zp = it.get_input_details()[0]["quantization"]
    return np.clip(np.round(feat / s + zp), -128, 127).astype(np.int8)

def child_main(kind, model_path, wav_path, mode="p", extra=None):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    sr, a = wavfile.read(wav_path)
    x = a.astype(np.float32) / 32768.0
    if mode == "sweep":
        it = _load_tflite(model_path)
        ps = []
        for st in range(0, max(len(x) - N_SAMPLES, 0) + 1, 1920):
            feat, _, _, gate = extract_features(x[st:st + N_SAMPLES])
            ps.append(0.0 if gate != "ok" else _run_tflite(it, _quantize(it, feat)))
        print("SWEEP=" + " ".join(f"{p:.2f}" for p in ps))
        return
    feat, rms, sd, gate = extract_features(x)
    if gate != "ok":
        print("P=0.000000")
        return
    import tensorflow as tf
    if kind == "tflite":
        it = _load_tflite(model_path)
        q = _quantize(it, feat)
        print(f"P={_run_tflite(it, q):.6f}")
        if extra and os.path.exists(extra):
            e = np.load(extra).astype(np.int8).reshape(q.shape)
            d = np.abs(e.astype(int) - q.astype(int))
            print(f"PE={_run_tflite(it, e):.6f}")
            print(f"QDIFF={d.mean():.3f},{d.max()},{100.0 * (d > 0).mean():.1f}")
    else:
        @classmethod
        def g(cls, c): c.pop("input_axes", None); c.pop("output_axes", None); return cls(**c)
        @classmethod
        def b(cls, c):
            for k in ("renorm", "renorm_clipping", "renorm_momentum"): c.pop(k, None)
            return cls(**c)
        @classmethod
        def d(cls, c): c.pop("quantization_config", None); return cls(**c)
        tf.keras.initializers.GlorotUniform.from_config = g
        tf.keras.layers.BatchNormalization.from_config = b
        tf.keras.layers.Dense.from_config = d
        m = tf.keras.models.load_model(model_path)
        print(f"P={float(m(feat[None], training=False).numpy()[0][0]):.6f}")

def run_child(kind, model_path, wav_path, mode="p", extra=None):
    """Returns dict of parsed KEY=value lines, or {'err': ...}."""
    if not model_path or not os.path.exists(model_path):
        return {"err": "n/a"}
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3", TF_ENABLE_ONEDNN_OPTS="0")
    cmd = [sys.executable, os.path.abspath(__file__), "--child", kind, model_path, wav_path, mode] + ([extra] if extra else [])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    except subprocess.TimeoutExpired:
        return {"err": "timeout"}
    out = dict(re.findall(r"^([A-Z]+)=(.*)$", r.stdout, flags=re.M))
    return out if out else {"err": f"CRASHED(exit {r.returncode})"}

def pstr(d):
    return d["err"] if "err" in d else f"{float(d['P']):.3f}" if "P" in d else "?"

# ---------- reports ----------
def analyse(a16, tag, esp_p, idx, args, esp_in=None, chk=None):
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{idx:02d}_{tag}.wav")
    wavfile.write(path, SR, a16)
    x = a16.astype(np.float32) / 32768.0
    feat, rms, sd, gate = extract_features(x)
    fr, pk, pk_share = band_stats(x)

    msg = f"[{idx}] {tag} samples={len(a16)}"
    if chk is not None:
        msg += " checksum OK" if chk else " CHECKSUM MISMATCH (serial data lost - rerun this event)"
    if esp_p is not None:
        msg += f" | ESP p={esp_p:.3f}"
    npy = None
    if esp_in is not None:
        npy = os.path.join(args.out, f"{idx:02d}_{tag}_input.npy")
        np.save(npy, esp_in)
    if gate == "ok":
        t = run_child("tflite", args.tflite, path, "p", npy)
        k = run_child("keras", args.keras, path)
        msg += f" | python tflite-int8 p={pstr(t)} | python keras p={pstr(k)}"
    else:
        t = {}
        msg += f" | {gate} -> zero tensor"
    print(msg)
    if "PE" in t:
        md, mx, pct = t["QDIFF"].split(",")
        print(f"     ESP int8 input fed to python TFLite: p={float(t['PE']):.3f} | ESP vs python quantized input: "
              f"mean |diff| {md}, max {mx}, {pct}% of values differ")
    print(f"     rms {rms:.4f} | peak {np.abs(x).max():.3f} | log-mel std {sd if sd is None else round(sd, 3)}")
    print(f"     {fmt_bands(fr)} | strongest {pk:.0f} Hz ({pk_share:.0f}%)")
    d0, d1 = edge_dc(x)
    print(f"     <100Hz rms x1000 per 100ms: {fmt_prof(low_profile(x))} | raw mean x1000 first/last: {d0:.1f}/{d1:.1f}")
    if args.hpf and gate == "ok":
        hp = np.clip(highpass(x, args.hpf) * 32768, -32768, 32767).astype(np.int16)
        hpath = os.path.join(args.out, f"{idx:02d}_{tag}_hpf{int(args.hpf)}.wav")
        wavfile.write(hpath, SR, hp)
        print(f"     after {args.hpf:.0f} Hz high-pass: python tflite p={pstr(run_child('tflite', args.tflite, hpath))}"
              f" | keras p={pstr(run_child('keras', args.keras, hpath))}")
    print(f"     saved {path}")

def dataset_report(root, n=60):
    import librosa
    for cls in ("friday", "background"):
        files = glob.glob(os.path.join(root, cls, "*.wav"))
        random.seed(0)
        files = random.sample(files, min(n, len(files)))
        if not files:
            continue
        rms, bands, peaks, profs, d0s, d1s = [], [], [], [], [], []
        for p in files:
            w, _ = librosa.load(p, sr=SR)
            w = np.pad(w[:N_SAMPLES], (0, max(0, N_SAMPLES - len(w))))
            rms.append(np.sqrt(np.mean((w - w.mean()) ** 2)))
            bands.append(band_stats(w)[0])
            peaks.append(np.abs(w).max())
            profs.append(low_profile(w))
            a, b = edge_dc(w)
            d0s.append(a); d1s.append(b)
        r = np.array(rms)
        print(f"[dataset/{cls}] n={len(files)} rms p10/50/90 = {np.percentile(r,10):.4f}/{np.median(r):.4f}/{np.percentile(r,90):.4f}"
              f" | peak median {np.median(peaks):.3f}")
        print(f"      median band power: {fmt_bands(np.median(np.array(bands), axis=0))}")
        print(f"      median <100Hz rms x1000 per 100ms: {fmt_prof(np.median(np.array(profs), axis=0))}")
        print(f"      median raw mean x1000 first/last 100ms: {np.median(d0s):.1f}/{np.median(d1s):.1f}")

def lines_from(src, log_path):
    if src.startswith("/dev/") or src.upper().startswith("COM"):
        import serial
        ser = serial.Serial(src, 115200, timeout=1)
        print(f"Capturing from {src} -> {log_path} (Ctrl+C to stop)", flush=True)
        with open(log_path, "w") as log:
            while True:
                raw = ser.readline()
                if not raw:
                    continue
                l = raw.decode(errors="ignore")
                log.write(l); log.flush()
                yield l
    else:
        with open(src, errors="ignore") as f:
            for l in f:
                yield l

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child_main(*sys.argv[2:8])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--dataset")
    ap.add_argument("--tflite", default="models/friday_model.tflite")
    ap.add_argument("--keras", default="models/friday.keras")
    ap.add_argument("--log", default="capture.log")
    ap.add_argument("--out", default="dumps")
    ap.add_argument("--hpf", type=float, default=0.0)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    if args.dataset:
        dataset_report(args.dataset)

    if args.source.lower().endswith(".wav"):
        import librosa
        w, _ = librosa.load(args.source, sr=SR)
        os.makedirs(args.out, exist_ok=True)
        if args.sweep:
            full = w if not args.hpf else highpass(w, args.hpf)
            fpath = os.path.join(args.out, "00_file_full.wav")
            wavfile.write(fpath, SR, np.clip(full * 32768, -32768, 32767).astype(np.int16))
            r = run_child("tflite", args.tflite, fpath, "sweep")
            print(f"tflite p per 120 ms hop over the whole clip ({len(w)/SR:.1f} s"
                  f"{', high-passed' if args.hpf else ''}):\n  {r.get('SWEEP', r.get('err'))}")
        w1 = np.pad(w[:N_SAMPLES], (0, max(0, N_SAMPLES - len(w))))
        analyse(np.clip(w1 * 32768, -32768, 32767).astype(np.int16), "file", None, 0, args)
        return

    jobs = queue.Queue()

    def worker():
        while True:
            item = jobs.get()
            try:
                analyse(*item)
            except Exception as e:
                print(f"(analysis failed: {e})", flush=True)
            jobs.task_done()

    threading.Thread(target=worker, daemon=True).start()

    cur, cur_in, esp_in, idx = None, None, None, 0
    try:
        for line in lines_from(args.source, args.log):
            line = line.strip()
            if line.startswith("INPUT_BEGIN"):
                cur_in = []
            elif line.startswith("INPUT_END") and cur_in is not None:
                s_in = "".join(cur_in)
                esp_in = None
                if len(s_in) == 49 * MEL_BINS * 2:
                    esp_in = np.frombuffer(bytes.fromhex(s_in), dtype=np.int8).copy()
                cur_in = None
            elif line.startswith("DUMP_BEGIN"):
                m = re.search(r"tag=(\w+) p=([\d.]+) n=\d+ sum=(-?\d+)", line)
                if m:
                    cur = {"tag": m.group(1), "p": float(m.group(2)), "sum": int(m.group(3)), "hex": []}
            elif line.startswith("DUMP_END") and cur:
                s_a = "".join(cur["hex"])
                if len(s_a) != N_SAMPLES * 4:
                    print(f"(dump '{cur['tag']}' lost data: {len(s_a)} of {N_SAMPLES * 4} hex chars received - skipped, repeat the event)",
                          flush=True)
                else:
                    a = np.frombuffer(bytes.fromhex(s_a), dtype=">i2").astype(np.int16)
                    ok = int(a.astype(np.int32).sum()) == cur["sum"]
                    idx += 1
                    jobs.put((a, cur["tag"], cur["p"], idx, args, esp_in if ok else None, ok))
                cur, esp_in = None, None
            elif cur_in is not None and re.fullmatch(r"[0-9a-f]+", line):
                cur_in.append(line)
            elif cur is not None and re.fullmatch(r"[0-9a-f]+", line):
                cur["hex"].append(line)
            elif "DETECTED" in line:
                print(line, flush=True)
    except KeyboardInterrupt:
        print("\nStopping - waiting for pending analysis (Ctrl+C again to quit)...", flush=True)
    try:
        jobs.join()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
