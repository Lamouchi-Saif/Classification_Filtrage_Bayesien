#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Partie 2.d - Test d'un modele DiffWave pre-entraine (SpeechBrain, Hugging Face :
speechbrain/tts-diffwave-ljspeech, entraine sur LJSpeech, 22.05 kHz)

Version adaptee du script d'origine (qui utilisait le depot lmnt-com/diffwave dont les poids
ne sont plus telechargeables). Les poids sont telecharges automatiquement par SpeechBrain.

NON TESTE par l'auteur de cette adaptation (pas d'acces reseau/GPU) : a lancer sur Colab/GPU.

Scenario de test (DiffWave = vocodeur neuronal conditionne par un mel-spectrogramme) :
  1. Resynthese ("copy-synthesis") : wav propre -> mel -> DiffWave -> wav reconstruit.
     Indicateurs : STOI, PESQ (wb), SI-SDR, distance log-spectrale (LSD).
  2. Echantillonnage standard (50 pas) vs rapide (6 pas) : qualite, latence, RTF, memoire.
  3. Limite : wav bruite (bruit blanc a differents SNR) -> mel bruite -> DiffWave.
     Le vocodeur NE debruite PAS (il reproduit le mel qu'on lui donne).
  4. Illustration numerique du processus de diffusion directe q(x_t | x_0).
  5. Spectrogrammes compares (propre / bruite / sortie DiffWave).

Installation :
    pip install speechbrain torch torchaudio soundfile numpy scipy matplotlib pystoi pesq psutil
Exemple :
    python diffwave_test_speechbrain.py --wav ma_voix.wav --out resultats

Reference : SpeechBrain (Ravanelli et al., 2021, arXiv:2106.04624) ;
            DiffWave (Kong et al., arXiv:2009.09761).
"""
import argparse
import json
import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import time

import numpy as np
import torch
import torchaudio                                    # uniquement pour resample (pas de load/save)
import soundfile as sf                              # lecture/ecriture wav sans torchcodec/FFmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from speechbrain.inference.vocoders import DiffWaveVocoder
try:                                                 # SpeechBrain recent
    from speechbrain.lobes.models.HifiGAN import mel_spectrogram as mel_spectogram
except ImportError:                                  # anciennes versions (nom avec faute)
    from speechbrain.lobes.models.HifiGAN import mel_spectogram

try:
    import resource                                  # Linux / macOS uniquement
except ImportError:                                  # Windows
    resource = None

SR = 22050                                           # frequence d'echantillonnage du modele
HOP = 256                                            # doit etre identique au calcul du mel
FAST_SCHEDULE = [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5]   # 6 pas (fiche du modele HF)


# ----------------------------------------------------------------- audio / mel
def load_wav(path, sr=SR):
    x, fs = sf.read(path, dtype="float32", always_2d=True)     # [T, canaux]
    wav = torch.from_numpy(x.mean(axis=1))                      # mono
    if fs != sr:
        wav = torchaudio.functional.resample(wav, fs, sr)
    return torch.clamp(wav, -1.0, 1.0)


def save_wav(path, audio, sr=SR):
    audio = audio.detach().cpu().numpy() if torch.is_tensor(audio) else np.asarray(audio)
    sf.write(path, np.clip(audio.astype(np.float32).squeeze(), -1.0, 1.0), sr, subtype="PCM_16")


def mel_diffwave(wav):
    """Parametres du mel identiques a ceux de la fiche du modele SpeechBrain. Sortie [1, 80, T]."""
    return mel_spectogram(
        sample_rate=SR, hop_length=HOP, win_length=1024, n_fft=1024, n_mels=80,
        f_min=0, f_max=8000, power=1.0, normalized=False,
        norm="slaney", mel_scale="slaney", compression=True,
        audio=wav.unsqueeze(0))


def add_noise(wav, snr_db, gen):
    p = wav.pow(2).mean()
    noise = torch.randn(wav.shape, generator=gen)
    noise = noise * torch.sqrt(p / (10 ** (snr_db / 10)) / noise.pow(2).mean())
    return torch.clamp(wav + noise, -1.0, 1.0)


# --------------------------------------------------------------------- DiffWave
def load_vocoder(device):
    # Windows : pas de liens symboliques sans droits admin -> on COPIE les fichiers
    extra = {}
    try:
        from speechbrain.utils.fetching import LocalStrategy
        extra["local_strategy"] = LocalStrategy.COPY
    except ImportError:                              # ancienne version de SpeechBrain
        pass
    return DiffWaveVocoder.from_hparams(
        source="speechbrain/tts-diffwave-ljspeech",
        savedir="pretrained_models/tts-diffwave-ljspeech",
        run_opts={"device": str(device)}, **extra)


def vocode(vocoder, mel, device, fast):
    mel = mel.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    kw = dict(hop_len=HOP, fast_sampling=fast)
    if fast:
        kw["fast_sampling_noise_schedule"] = FAST_SCHEDULE
    waveforms = vocoder.decode_batch(mel, **kw)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    audio = waveforms.squeeze().detach().cpu().numpy()
    if device.type == "cuda":
        mem = torch.cuda.max_memory_allocated() / 2**20
    else:   # memoire du PROCESSUS entier : non comparable d'un run a l'autre
        mem = cpu_process_mem_mib()
    return audio, SR, dt, mem


def cpu_process_mem_mib():
    """Linux/macOS : pic RSS via resource ; Windows : RSS courant via psutil (pip install psutil)."""
    if resource is not None:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024   # ko sous Linux
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except Exception:                                # noqa
        return None


def count_params(vocoder):
    """Nombre de parametres du reseau (essaie plusieurs chemins d'acces, selon la version)."""
    for getter in (lambda: vocoder.hparams.diffwave.parameters(),
                   lambda: vocoder.mods.parameters()):
        try:
            return sum(p.numel() for p in getter())
        except Exception:                            # noqa
            continue
    return None


# -------------------------------------------------------------------- metriques
def align(a, b):
    n = min(len(a), len(b))
    return a[:n].astype(np.float64), b[:n].astype(np.float64)


def si_sdr(ref, est):
    ref, est = align(ref, est)
    ref, est = ref - ref.mean(), est - est.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    target = alpha * ref
    return 10 * np.log10(np.sum(target ** 2) / (np.sum((est - target) ** 2) + 1e-12) + 1e-12)


def lsd(ref, est, n_fft=1024, hop=256):
    ref, est = align(ref, est)
    w = np.hanning(n_fft)

    def spec(x):
        fr = np.stack([x[i:i + n_fft] * w for i in range(0, len(x) - n_fft, hop)])
        return np.abs(np.fft.rfft(fr, axis=1)) ** 2 + 1e-10
    d = 10 * np.log10(spec(ref)) - 10 * np.log10(spec(est))
    return float(np.mean(np.sqrt(np.mean(d ** 2, axis=1))))


def stoi_pesq(ref, est, sr):
    out = {}
    ref, est = align(ref, est)
    try:
        from pystoi import stoi
        out["STOI"] = float(stoi(ref, est, sr, extended=False))
    except Exception as e:                           # noqa
        print("STOI indisponible :", e); out["STOI"] = None
    try:
        from pesq import pesq
        r16 = torchaudio.functional.resample(torch.tensor(ref).float(), sr, 16000).numpy()
        e16 = torchaudio.functional.resample(torch.tensor(est).float(), sr, 16000).numpy()
        out["PESQ_wb"] = float(pesq(16000, r16, e16, "wb"))
    except Exception as e:                           # noqa
        print("PESQ indisponible :", e); out["PESQ_wb"] = None
    return out


def all_metrics(ref, est, sr):
    # NB : SI-SDR est souvent bas pour un vocodeur (phase regeneree, pas d'alignement
    # echantillon par echantillon) : s'appuyer surtout sur STOI / PESQ / LSD.
    m = {"SI-SDR_dB": float(si_sdr(ref, est)), "LSD_dB": lsd(ref, est)}
    m.update(stoi_pesq(ref, est, sr))
    return m


# ------------------------------------------------- illustration : diffusion directe
def demo_forward_diffusion(wav, outdir, T=50, beta_min=1e-4, beta_max=0.05):
    """q(x_t|x_0) = N(sqrt(abar_t) x_0, (1-abar_t) I), planning lineaire de DiffWave BASE
    (a verifier dans le hparams.yaml du modele utilise)."""
    beta = np.linspace(beta_min, beta_max, T)
    abar = np.cumprod(1 - beta)
    x0 = wav[: 2 * 2048].numpy()
    x0 = x0 / (np.abs(x0).max() + 1e-9)
    rng = np.random.default_rng(0)
    ts = [0, 4, 14, 29, T - 1]
    fig, axs = plt.subplots(len(ts), 1, figsize=(8, 8), sharex=True)
    for ax, t in zip(axs, ts):
        xt = np.sqrt(abar[t]) * x0 + np.sqrt(1 - abar[t]) * rng.standard_normal(x0.shape)
        snr = 10 * np.log10(abar[t] / (1 - abar[t]))
        ax.plot(xt, lw=0.6)
        ax.set_ylabel(f"t={t + 1}")
        ax.set_title(f"SNR(t) = {snr:.1f} dB,  alpha_bar = {abar[t]:.4f}", fontsize=8, loc="right")
    axs[-1].set_xlabel("echantillon")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "diffusion_forward.png"), dpi=140)
    plt.close(fig)
    print(f"[diffusion] alpha_bar_T = {abar[-1]:.3f}  ->  SNR(T) = "
          f"{10 * np.log10(abar[-1] / (1 - abar[-1])):.1f} dB")


def plot_spectrograms(signals, outdir, name="spectrogrammes.png"):
    """signals : liste de (titre, signal numpy) ; spectrogrammes en dB cote a cote."""
    fig, axs = plt.subplots(1, len(signals), figsize=(4.5 * len(signals), 3.6), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, (title, x) in zip(axs, signals):
        ax.specgram(x, NFFT=1024, Fs=SR, noverlap=768, cmap="magma", vmin=-120, vmax=-20)
        ax.set_title(title, fontsize=9); ax.set_xlabel("temps (s)")
    axs[0].set_ylabel("frequence (Hz)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, name), dpi=140)
    plt.close(fig)


def print_table(results):
    print("\n=== Tableau recapitulatif (a recopier dans le rapport) ===")
    print(f"{'Scenario':44s} {'STOI':>6s} {'PESQ':>6s} {'LSD':>6s} {'SI-SDR':>7s} {'RTF':>6s}")
    f = lambda v: "  n/a " if v is None else f"{v:6.3f}"
    for r in results["runs"]:
        if "sortie_vs_propre" in r:
            for lab, m, rtf in (("  entree bruitee vs propre", r["entree_bruitee_vs_propre"], None),
                                ("  sortie DiffWave vs propre", r["sortie_vs_propre"], r["RTF"])):
                print(f"{r['scenario'][:20] + lab:44s} {f(m['STOI'])} {f(m['PESQ_wb'])} "
                      f"{m['LSD_dB']:6.2f} {m['SI-SDR_dB']:7.2f} {f(rtf)}")
        else:
            print(f"{r['scenario']:44s} {f(r['STOI'])} {f(r['PESQ_wb'])} {r['LSD_dB']:6.2f} "
                  f"{r['SI-SDR_dB']:7.2f} {r['RTF']:6.3f}")


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="fichier audio de test (parole propre)")
    ap.add_argument("--out", default="resultats")
    ap.add_argument("--snr", type=float, nargs="*", default=[20, 10, 0])
    ap.add_argument("--max_seconds", type=float, default=8.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)                             # echantillonnage reproductible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    vocoder = load_vocoder(device)
    n_par = count_params(vocoder)
    if n_par:
        print(f"Parametres du modele : {n_par / 1e6:.2f} M  ({n_par * 4 / 2**20:.1f} Mo en float32)")

    wav = load_wav(args.wav)[: int(args.max_seconds * SR)]
    dur = len(wav) / SR
    print(f"Signal de test : {dur:.2f} s a {SR} Hz")
    demo_forward_diffusion(wav, args.out)

    results = {"device": str(device), "params_M": (n_par or 0) / 1e6, "duree_s": dur, "runs": []}
    ref = wav.numpy()
    mel_clean = mel_diffwave(wav)                    # [1, 80, T]

    # ---- 1 & 2 : copy-synthesis, 50 pas vs 6 pas
    outputs = {}
    for fast in (False, True):
        name = "rapide (6 pas)" if fast else "standard (50 pas)"
        vocode(vocoder, mel_clean[..., :50], device, fast)         # echauffement
        audio, sr, dt, mem = vocode(vocoder, mel_clean, device, fast)
        outputs[fast] = audio
        m = all_metrics(ref, audio, sr)
        m.update({"scenario": f"copy-synthesis, {name}", "latence_s": dt, "RTF": dt / dur,
                  "memoire_MiB": mem})
        results["runs"].append(m)
        save_wav(os.path.join(args.out, f"copy_{'fast' if fast else 'std'}.wav"), audio, sr)
        print(json.dumps(m, indent=1))

    # ---- 3 : mel bruite -> DiffWave (limite : pas de debruitage)
    gen = torch.Generator().manual_seed(0)
    spec_signals = [("propre", ref), ("DiffWave(mel propre), 50 pas", outputs[False])]
    for snr in args.snr:
        noisy = add_noise(wav, snr, gen)
        audio, sr, dt, mem = vocode(vocoder, mel_diffwave(noisy), device, False)
        m_out = all_metrics(ref, audio, sr)                         # sortie vs reference propre
        m_in = all_metrics(ref, noisy.numpy(), sr)                  # entree bruitee vs propre
        results["runs"].append({"scenario": f"mel bruite SNR={snr:g} dB -> DiffWave",
                                "sortie_vs_propre": m_out, "entree_bruitee_vs_propre": m_in,
                                "RTF": dt / dur})
        save_wav(os.path.join(args.out, f"noisy_{int(snr)}dB_in.wav"), noisy, SR)
        save_wav(os.path.join(args.out, f"noisy_{int(snr)}dB_out.wav"), audio, sr)
        print(f"SNR {snr:>4} dB  entree: {m_in}\n             sortie: {m_out}")
        if snr == args.snr[-1] or snr == 10:
            spec_signals += [(f"bruite {snr:g} dB", noisy.numpy()),
                             (f"DiffWave(mel bruite {snr:g} dB)", audio)]

    plot_spectrograms(spec_signals, args.out)
    with open(os.path.join(args.out, "resultats.json"), "w") as f:
        json.dump(results, f, indent=2)
    print_table(results)
    print(f"\nResultats ecrits dans {args.out}/ (resultats.json, figures, wav)")


if __name__ == "__main__":
    main()