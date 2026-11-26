"""Gera arquivos de teste para os leitores da página (docs/index.html).

Os arquivos são escritos pelas bibliotecas de referência (wfdb e pyedflib), não por código
nosso, e relidos por elas para obter o sinal esperado em mV. O teste em JavaScript
(testar_leitores.mjs) lê os mesmos arquivos com os leitores da página e compara amostra a amostra.

Uso:  pip install wfdb pyedflib numpy
      python testes/gerar_casos.py          (escreve em testes/saida/)
"""
import json
import os

import numpy as np
import pyedflib
import wfdb

AQUI = os.path.dirname(os.path.abspath(__file__))
SAIDA = os.path.join(AQUI, "saida")
os.makedirs(SAIDA, exist_ok=True)

FS = 250
DUR = 20


def ecg_sintetico(fs=FS, dur=DUR, bpm=72):
    """Duas derivações em mV: P, QRS, T gaussianos, com linha de base oscilando."""
    t = np.arange(int(fs * dur)) / fs
    rr = 60 / bpm
    fase = (t + rr / 2) % rr - rr / 2
    g = lambda a, c, w: a * np.exp(-0.5 * ((fase - c) / w) ** 2)
    d1 = g(0.12, -0.17, 0.02) + g(1.1, 0, 0.011) + g(-0.28, 0.032, 0.011) + g(0.32, 0.27, 0.045)
    d2 = g(0.07, -0.17, 0.02) - 0.55 * g(1, 0, 0.011) + g(-0.22, 0.27, 0.045)
    base = 0.2 * np.sin(2 * np.pi * 0.2 * t)
    return np.column_stack([d1 + base, d2 + 0.6 * base])


SINAL = ecg_sintetico()
NOMES = ["ECG1", "ECG2"]
casos = []


def esperado(nome, arquivos, fs, sinal_mv, nomes, extra=None):
    c = {"nome": nome, "arquivos": arquivos, "fs": fs, "n": int(sinal_mv.shape[0]),
         "nomes": nomes, "sinal": np.round(sinal_mv, 9).T.tolist()}
    c.update(extra or {})
    casos.append(c)


# ---------- WFDB: formatos aceitos pela página ----------
# A biblioteca wfdb escreve 16, 24, 32 e 80.
for fmt in ["16", "24", "32", "80"]:
    rec = f"wfdb_{fmt}"
    wfdb.wrsamp(rec, fs=FS, units=["mV", "mV"], sig_name=NOMES, p_signal=SINAL, fmt=[fmt, fmt], write_dir=SAIDA)
    lido = wfdb.rdrecord(os.path.join(SAIDA, rec))
    esperado(rec, [rec + ".hea", rec + ".dat"], FS, lido.p_signal, NOMES)

# 212 (o formato da SDDB e da QTDB) e 61 ela só lê: empacotamos os valores digitais aqui
# e conferimos que a própria wfdb relê o arquivo com o sinal original.
# ganho de 200 unidades/mV, como na SDDB: o sinal cabe nos 12 bits do formato 212
wfdb.wrsamp("wfdb_base", fs=FS, units=["mV", "mV"], sig_name=NOMES, p_signal=SINAL, fmt=["16", "16"],
            adc_gain=[200.0, 200.0], baseline=[0, 0], write_dir=SAIDA)
base = wfdb.rdrecord(os.path.join(SAIDA, "wfdb_base"), physical=False)
ref = wfdb.rdrecord(os.path.join(SAIDA, "wfdb_base")).p_signal
dig = base.d_signal.astype(np.int64)
assert np.abs(dig).max() < 2047
for fmt in ["212", "61"]:
    rec = f"wfdb_{fmt}"
    d = dig
    if fmt == "212":
        v = (d.reshape(-1) & 0xFFF).astype(np.uint32)
        if len(v) % 2:
            v = np.append(v, 0)
        a, b = v[0::2], v[1::2]
        dat = np.column_stack([a & 0xFF, ((a >> 8) & 0x0F) | ((b >> 4) & 0xF0), b & 0xFF]).astype(np.uint8).tobytes()
    else:
        dat = d.astype(">i2").tobytes()
    with open(os.path.join(SAIDA, rec + ".dat"), "wb") as fh:
        fh.write(dat)
    with open(os.path.join(SAIDA, "wfdb_base.hea"), encoding="utf-8") as fh:
        hea = fh.read().replace("wfdb_base.dat 16", f"{rec}.dat {fmt}").replace("wfdb_base ", f"{rec} ", 1)
    with open(os.path.join(SAIDA, rec + ".hea"), "w", encoding="utf-8") as fh:
        fh.write(hea)
    lido = wfdb.rdrecord(os.path.join(SAIDA, rec))
    assert np.allclose(lido.p_signal, ref), f"a wfdb não releu o {fmt} empacotado aqui"
    esperado(rec, [rec + ".hea", rec + ".dat"], FS, lido.p_signal, NOMES)

# Unidade em µV no cabeçalho: a página deve converter para mV
rec = "wfdb_16_uV"
wfdb.wrsamp(rec, fs=FS, units=["uV", "uV"], sig_name=NOMES, p_signal=SINAL * 1000, fmt=["16", "16"], write_dir=SAIDA)
lido = wfdb.rdrecord(os.path.join(SAIDA, rec))
esperado(rec, [rec + ".hea", rec + ".dat"], FS, lido.p_signal / 1000, NOMES)

# Formato não aceito: a página deve recusar com mensagem clara
rec = "wfdb_310"
with open(os.path.join(SAIDA, "wfdb_16.hea"), encoding="utf-8") as fh:
    hea = fh.read().replace("wfdb_16.dat 16", f"{rec}.dat 310").replace("wfdb_16 ", f"{rec} ", 1)
with open(os.path.join(SAIDA, rec + ".hea"), "w", encoding="utf-8") as fh:
    fh.write(hea)
with open(os.path.join(SAIDA, rec + ".dat"), "wb") as fh:
    fh.write(bytes(1200))
casos.append({"nome": rec, "arquivos": [rec + ".hea", rec + ".dat"], "erro": "Formato 310"})

# .hea sem o .dat: a página deve pedir o arquivo que falta
casos.append({"nome": "wfdb_sem_dat", "arquivos": ["wfdb_16.hea"], "erro": "Falta o arquivo"})


# ---------- EDF+, BDF+ e EDF simples (pyedflib) ----------
def escrever_edf(nome, tipo, canais):
    """canais: lista de (rótulo, dimensão, fs, dados físicos na dimensão)."""
    caminho = os.path.join(SAIDA, nome)
    f = pyedflib.EdfWriter(caminho, len(canais), file_type=tipo)
    cab = []
    for rot, dim, fs, x in canais:
        cab.append({"label": rot, "dimension": dim, "sample_frequency": fs,
                    "physical_max": float(np.ceil(x.max()) + 1), "physical_min": float(np.floor(x.min()) - 1),
                    "digital_max": 8388607 if tipo in (pyedflib.FILETYPE_BDF, pyedflib.FILETYPE_BDFPLUS) else 32767,
                    "digital_min": -8388608 if tipo in (pyedflib.FILETYPE_BDF, pyedflib.FILETYPE_BDFPLUS) else -32768,
                    "transducer": "", "prefilter": ""})
    f.setSignalHeaders(cab)
    if tipo in (pyedflib.FILETYPE_EDFPLUS, pyedflib.FILETYPE_BDFPLUS):
        f.writeAnnotation(1.0, -1, "evento de teste")
    # um registro de dados por segundo, canal após canal (canais com taxas diferentes)
    for s in range(DUR):
        for _, _, fs, x in canais:
            f.writePhysicalSamples(np.ascontiguousarray(x[s * fs:(s + 1) * fs], dtype=np.float64))
    f.close()
    r = pyedflib.EdfReader(caminho)
    lidos = [r.readSignal(k) for k in range(r.signals_in_file)]
    r.close()
    return caminho, lidos


spo2 = np.full(DUR, 97.0)
for nome, tipo in [("edfplus.edf", pyedflib.FILETYPE_EDFPLUS), ("bdfplus.bdf", pyedflib.FILETYPE_BDFPLUS), ("edf_simples.edf", pyedflib.FILETYPE_EDF)]:
    # ECG I em µV, ECG II em mV e um canal lento (SpO2, 1 Hz) que deve ser ignorado
    canais = [("ECG I", "uV", FS, SINAL[:, 0] * 1000), ("ECG II", "mV", FS, SINAL[:, 1]), ("SpO2", "%", 1, spo2)]
    _, lidos = escrever_edf(nome, tipo, canais)
    mv = np.column_stack([lidos[0] / 1000, lidos[1]])
    esperado(nome, [nome], FS, mv, ["ECG I", "ECG II"])


# ---------- CSV ----------
def csv(nome, texto, fs, sinal_mv, nomes, opcoes):
    with open(os.path.join(SAIDA, nome), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(texto)
    esperado(nome, [nome], fs, sinal_mv, nomes, {"opcoes": opcoes})


S = np.round(SINAL, 5)
t = np.arange(len(S)) / FS
csv("virgula_cabecalho.csv", "D1,D2\n" + "\n".join(f"{a:.5f},{b:.5f}" for a, b in S) + "\n",
    FS, S, ["D1", "D2"], {"fs": FS, "unid": "mV", "tempo": False})
csv("pontovirgula_decimal_tempo.csv", "tempo;D1;D2\n" + "\n".join(f"{x:.3f};{a:.5f};{b:.5f}".replace(".", ",") for x, (a, b) in zip(t, S)) + "\n",
    FS, S, ["D1", "D2"], {"fs": 100, "unid": "mV", "tempo": True})
# tempo com 3 casas a 360 Hz: os passos alternam entre 0,002 e 0,003 s; a amostragem tem de sair 360
S360 = np.round(ecg_sintetico(fs=360, dur=10), 5)
t360 = np.arange(len(S360)) / 360
csv("tempo_arredondado_360.csv", "\n".join(f"{x:.3f},{a:.5f},{b:.5f}" for x, (a, b) in zip(t360, S360)) + "\n",
    360, S360, ["Coluna 1", "Coluna 2"], {"fs": 250, "unid": "mV", "tempo": True})
csv("tab_sem_cabecalho.tsv", "\n".join(f"{a:.5f}\t{b:.5f}" for a, b in S) + "\n",
    FS, S, ["Coluna 1", "Coluna 2"], {"fs": FS, "unid": "mV", "tempo": False})
csv("espacos.txt", "\n".join(f"  {a:.5f}   {b:.5f}" for a, b in S) + "\n",
    FS, S, ["Coluna 1", "Coluna 2"], {"fs": FS, "unid": "mV", "tempo": False})
csv("aspas_crlf.csv", '"D1","D2"\r\n' + "\r\n".join(f'"{a:.5f}","{b:.5f}"' for a, b in S) + "\r\n",
    FS, S, ["D1", "D2"], {"fs": FS, "unid": "mV", "tempo": False})
# metadados antes dos dados, uma derivação em µV a 512 Hz (formato comum de exportação de relógios)
S512 = np.round(ecg_sintetico(fs=512, dur=10)[:, 0] * 1000, 3)
meta = "Name,Teste\nRecorded Date,2026-01-01 10:00:00\nSample Rate,512 hertz\n\nLead,Lead I\nUnit,µV\n\n"
csv("metadados_uV.csv", meta + "\n".join(f"{v:.3f}" for v in S512) + "\n",
    512, (S512 / 1000)[:, None], ["Coluna 1"], {"fs": 250, "unid": "auto", "tempo": False, "meta_fs": 512, "meta_unid": "uV"})

# Arquivos inválidos
with open(os.path.join(SAIDA, "nao_e_edf.edf"), "wb") as fh:
    fh.write(b"isto nao e um EDF" * 40)
casos.append({"nome": "nao_e_edf.edf", "arquivos": ["nao_e_edf.edf"], "erro": "EDF inválido"})
with open(os.path.join(SAIDA, "so_texto.csv"), "w", encoding="utf-8") as fh:
    fh.write("nome,valor\nabc,def\n")
casos.append({"nome": "so_texto.csv", "arquivos": ["so_texto.csv"], "opcoes": {"fs": 250, "unid": "mV", "tempo": False}, "erro": "Não encontrei linhas"})

with open(os.path.join(SAIDA, "casos.json"), "w", encoding="utf-8") as fh:
    json.dump(casos, fh)
print(f"{len(casos)} casos em {SAIDA}")
