"""Referência em Python para conferir a delineação da página (docs/index.html) com a do TCC (codigo/tcc_lib.py).

Para cada sinal: prepara com preparar_para_delinear, delineia com delinear_batimentos nas versões 1 e 2 a partir
dos mesmos picos R e grava sinal, picos e resultado em testes/saida/delineacao.json. O teste em JavaScript
(testar_delineacao.mjs) refaz tudo com o código da página e compara ponto a ponto.

Uso:  python testes/gerar_delineacao.py [registro WFDB ...]     (ex.: caminho/para/sel30, opcional)
"""
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "codigo"))
import tcc_lib as T  # noqa: E402

SAIDA = os.path.join(AQUI, "saida")
os.makedirs(SAIDA, exist_ok=True)
rng = np.random.default_rng(0)


def sintetico(fs, dur, bpm, var_rr, amp_t, desvio_st, ruido):
    """ECG de duas derivações com RR variável, T de amplitude escolhida (inclusive achatada) e ST deslocado."""
    n = int(fs * dur)
    t = np.arange(n) / fs
    x = np.zeros((n, 2))
    picos = []
    tb = 0.6
    while tb < dur - 1:
        rr = 60 / bpm * (1 + var_rr * rng.standard_normal())
        g = lambda a, c, w: a * np.exp(-0.5 * ((t - tb - c) / w) ** 2)
        x[:, 0] += g(0.12, -0.17, 0.02) + g(1.1, 0, 0.011) + g(-0.28, 0.032, 0.011) + g(desvio_st, 0.12, 0.05) + g(amp_t, 0.27, 0.045)
        x[:, 1] += g(0.07, -0.17, 0.02) - 0.55 * g(1, 0, 0.011) + g(-0.7 * amp_t, 0.27, 0.045)
        picos.append(int(round(tb * fs)))
        tb += rr
    x += 0.15 * np.sin(2 * np.pi * 0.25 * t)[:, None] + ruido * rng.standard_normal((n, 2))
    return x, np.array(picos)


casos = []


def adicionar(nome, sinal, fs, picos):
    prep = T.preparar_para_delinear(sinal, fs)
    res = {}
    for v in (1, 2):
        d = T.delinear_batimentos(prep, fs, picos, versao=v)
        cols = ["R", "QRSon", "QRSfim", "Ton", "Tpico", "Tfim", "Ton_faixa", "Tfim_faixa", "amp_QRS", "amp_T", "T_prom", "T_confiavel"]
        res[v] = {c: [None if (isinstance(val, float) and np.isnan(val)) else (bool(val) if c == "T_confiavel" else float(val))
                      for val in d[c]] if c in d else [] for c in cols}
    casos.append({"nome": nome, "fs": fs, "sinal": sinal.T.round(9).tolist(), "picos": [int(p) for p in picos],
                  "prep": prep.T.round(12).tolist(), "v1": res[1], "v2": res[2]})
    print(f"{nome}: {len(picos)} picos, v1 {len(res[1]['R'])} batimentos, v2 {len(res[2]['R'])} batimentos")


for nome, kw in [("normal 70 bpm", dict(bpm=70, var_rr=0.03, amp_t=0.32, desvio_st=0.0, ruido=0.01)),
                 ("taquicardia 130 bpm", dict(bpm=130, var_rr=0.05, amp_t=0.30, desvio_st=0.05, ruido=0.015)),
                 ("T achatada", dict(bpm=75, var_rr=0.04, amp_t=0.04, desvio_st=0.0, ruido=0.01)),
                 ("ST elevado, ruído", dict(bpm=85, var_rr=0.08, amp_t=0.25, desvio_st=0.15, ruido=0.04))]:
    s, p = sintetico(250, 40, **kw)
    adicionar(nome, s, 250, p)

# registros reais (opcional): picos do xqrs, como no TCC
for caminho in sys.argv[1:]:
    import wfdb
    import wfdb.processing as P
    r = wfdb.rdrecord(caminho)
    sinal = np.nan_to_num(r.p_signal)
    picos = P.xqrs_detect(sinal[:, 0], fs=r.fs, verbose=False)
    adicionar(os.path.basename(caminho), sinal, r.fs, picos)

with open(os.path.join(SAIDA, "delineacao.json"), "w", encoding="utf-8") as fh:
    json.dump(casos, fh)
