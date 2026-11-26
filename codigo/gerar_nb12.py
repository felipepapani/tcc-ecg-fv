import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md(r"""# 12 — Onda P e segmento ST a partir da matriz de distância — validação na QT Database

**Onde estamos:** o notebook 11 delineou QRS e onda T. O ponto J (fim do QRS) é consistente dentro de cada registro (DP ≤ 12 ms na
maioria); o fim da T melhorou com a v2, mas ainda não atinge a tolerância CSE. Para o **ST isso não importa**: a medida clássica do ST
é o nível em **J + 60 ms** (ST60) em relação à linha isoelétrica — usa só o ponto J e o PR, não as bordas da T.

**Este notebook faz duas coisas:**
1. **Onda P** — mesma ideia da T: `d(n) = ‖x(n) − b‖`, a distância de cada amostra ao repouso (o nível isoelétrico do PR), agora numa
   janela **antes** do QRS (PR de até 300 ms, sem invadir a T do batimento anterior). Pico = corcova mais proeminente; início e fim pelo
   trapézio. Validada contra o início, pico e fim da P marcados pelo especialista (tolerância CSE: início 10,2 ms, fim 12,7 ms).
   Esta parte **não foi ajustada olhando estes dados**: a regra foi escrita antes (a mesma da T v2), então todos os registros servem de
   teste.
   *Controle natural:* quatro trechos são de **fibrilação atrial** (sel35, sel36, sel37, sel50), em que não existe onda P — o especialista
   não marcou P neles. O método deve marcar "P não confiável" ali.
2. **ST60** — o nível em J + 60 ms por derivação (com sinal: + supra, − infra) e como distância (sem sinal). Não há ST anotado na QT
   Database, então a validação é indireta e direta ao ponto: **quanto o erro do nosso ponto J muda o ST60**, comparando o ST60 medido no J
   do método com o ST60 medido no J do especialista, no mesmo sinal. A referência clínica é 0,1 mV (1 mm): é o desvio do ST considerado
   significativo no ECG; um erro bem menor que isso não muda a leitura.

**Saídas:** `p_st_erros.csv`, `p_st_resumo.csv`, `p_st_por_registro.csv`; figuras `fig_p_metodo.png`, `fig_p_erros.png`,
`fig_st_erro.png`.""")

md("## 0. Configuração")
code(r"""import os, sys, importlib, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

for pacote in ('wfdb', 'neurokit2'):
    try:
        __import__(pacote)
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pacote], check=True)
import wfdb
from wfdb import processing

EM_COLAB = os.path.isdir('/content')
if EM_COLAB and not os.path.isdir('/content/drive/MyDrive'):
    from google.colab import drive
    drive.mount('/content/drive')
BASE = os.environ.get('TCC_BASE') or ('/content/drive/MyDrive/Pesquisa/TCC' if EM_COLAB else os.path.abspath('TCC'))
PASTA_CODIGO = os.path.join(BASE, 'codigo')
PASTA_QTDB   = os.path.join(BASE, 'QTDB')
PASTA_CSV    = os.path.join(BASE, 'tcc_p_st', 'csvs')
PASTA_FIG    = os.path.join(BASE, 'tcc_p_st', 'figs')
for p in (PASTA_QTDB, PASTA_CSV, PASTA_FIG):
    os.makedirs(p, exist_ok=True)
sys.path.insert(0, PASTA_CODIGO)
import tcc_lib as T
importlib.reload(T)
T.estilo_figuras()

COMPARAR_NK = True
FA = {'sel35', 'sel36', 'sel37', 'sel50'}          # fibrilação atrial: sem onda P
ST_MS = 60
REF_ST_MV = 0.10                                     # 1 mm: desvio do ST considerado significativo
NOMES_P = {'Pon': 'Início da P', 'Ppico': 'Pico da P', 'Pfim': 'Fim da P', 'QRSfim': 'Ponto J'}

def ancoras_r(sig, fs):
    return np.asarray(processing.xqrs_detect(sig[:, 0], fs=fs, verbose=False))""")

md("## 1. QT Database (os mesmos 24 trechos da SDDB do notebook 11)")
code(r"""QT_SDDB = [f'sel{n}' for n in range(30, 53)] + ['sel17152']
ok = T.baixar_wfdb('qtdb', QT_SDDB, PASTA_QTDB, exts=('hea', 'dat', 'q1c'))
print(f'{len(ok)} de {len(QT_SDDB)} registros')
MAN, INFO = {}, {}
for r in ok:
    h = wfdb.rdheader(os.path.join(PASTA_QTDB, r))
    INFO[r] = {'fs': float(h.fs), 'nomes': h.sig_name}
    MAN[r] = T.ler_marcas_qtdb(os.path.join(PASTA_QTDB, r), 'q1c')
print('Batimentos com P anotada:', int(sum(m.Pon.notna().sum() for m in MAN.values())),
      '| em fibrilação atrial (não deve haver P):', [r for r in ok if r in FA])""")

md(r"""## 2. O método num batimento real
As duas derivações, a curva `d(n)` antes e depois do QRS, a P (início, pico, fim), o ponto J e o ponto onde o ST60 é medido.""")
code(r"""r = next(x for x in ok if x not in FA and MAN[x].Pon.notna().sum() > 0)
fs = INFO[r]['fs']
sig = np.nan_to_num(wfdb.rdrecord(os.path.join(PASTA_QTDB, r)).p_signal)
xp = T.preparar_para_delinear(sig, fs)
de = T.delinear_p_st(xp, fs, T.delinear_batimentos(xp, fs, ancoras_r(sig, fs), versao=2), st_ms=ST_MS)
m = MAN[r]; m = m[m.Pon.notna()]
alvo = m.iloc[len(m) // 2]
b = de.iloc[int(np.argmin(np.abs(de.R - alvo.R)))]
a0, a1 = int(b.R - 0.35 * fs), int(b.R + 0.25 * fs)
tt = (np.arange(a0, a1) - b.R) / fs * 1000
nivel, _ = T.ponto_isoeletrico(xp, int(b.R), fs)
d = np.sqrt(((xp[a0:a1] - nivel) ** 2).sum(axis=1))
fig, axs = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
for c, cor in zip(range(xp.shape[1]), ['tab:blue', 'tab:gray']):
    axs[0].plot(tt, xp[a0:a1, c], color=cor, lw=1.6, label=f'Derivação {c + 1}')
    axs[0].axhline(nivel[c], color=cor, ls=':', lw=1)
axs[0].set_ylabel('Amplitude (mV)'); axs[0].set_title(f'(a) Batimento do registro {r}: linha isoelétrica pontilhada')
axs[1].plot(tt, d, color='k', lw=1.8, label='d(n) = distância ao repouso')
for p, cor, ls, rot in [('Pon', 'tab:orange', '-', 'Bordas da P (trapézio)'), ('Ppico', 'tab:orange', ':', 'Pico da P'),
                        ('Pfim', 'tab:orange', '-', None), ('QRSfim', 'tab:purple', '-', 'Ponto J')]:
    if np.isfinite(b[p]):
        for a in axs:
            a.axvline((b[p] - b.R) / fs * 1000, color=cor, ls=ls, lw=2, label=rot if a is axs[1] else None)
j60 = (b.QRSfim - b.R) / fs * 1000 + ST_MS
for a in axs:
    a.axvline(j60, color='tab:red', ls='--', lw=2, label=f'J + {ST_MS} ms (ST{ST_MS})' if a is axs[1] else None)
for p in ['Pon', 'Ppico', 'Pfim', 'QRSfim']:
    if np.isfinite(alvo[p]):
        axs[1].plot((alvo[p] - b.R) / fs * 1000, 0, '^', ms=13, color='tab:green', clip_on=False,
                    label='Especialista (QT Database)' if p == 'Pon' else None)
axs[1].set_xlabel('Tempo em relação ao R (ms)'); axs[1].set_ylabel('Distância (mV)')
axs[1].set_title('(b) Distância ao repouso: a P e o ST são afastamentos do nível isoelétrico')
axs[0].legend(loc='upper left', fontsize=11)
axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=11)
for a in axs: T.negrito_ticks(a)
plt.tight_layout(); plt.savefig(os.path.join(PASTA_FIG, 'fig_p_metodo.png')); plt.show()
print(f"ST{ST_MS} neste batimento: " + ', '.join(f"derivação {k + 1} = {T.num(b[f'ST_d{k + 1}'], 3)} mV" for k in range(xp.shape[1])))""")

md(r"""## 3. Todos os registros
P pelo método da matriz e pelo `neurokit2` (wavelet, uma derivação por vez), nos mesmos R (xqrs). Para o ST: em cada batimento com o
J anotado, o ST60 no J do método menos o ST60 no J do especialista (mesmo sinal, mesma linha isoelétrica) — **o erro no ST causado só
pelo erro do ponto J**.""")
code(r"""def p_nk(sig, fs, r_amostras, canal):
    import neurokit2 as nk
    try:
        _, w = nk.ecg_delineate(nk.ecg_clean(sig[:, canal], sampling_rate=int(fs)), rpeaks=np.asarray(r_amostras),
                                sampling_rate=int(fs), method='dwt')
    except Exception as e:
        print(f'   neurokit2 falhou ({type(e).__name__})'); return pd.DataFrame()
    out = pd.DataFrame({'R': np.asarray(r_amostras)})
    for p, k in {'Pon': 'ECG_P_Onsets', 'Ppico': 'ECG_P_Peaks', 'Pfim': 'ECG_P_Offsets'}.items():
        v = np.asarray(w.get(k, [np.nan] * len(out)), float)
        out[p] = v[:len(out)] if len(v) >= len(out) else np.r_[v, [np.nan] * (len(out) - len(v))]
    return out

erros, st_err, cobertura = [], [], []
for r in ok:
    fs = INFO[r]['fs']
    sig = np.nan_to_num(wfdb.rdrecord(os.path.join(PASTA_QTDB, r)).p_signal)
    rq = ancoras_r(sig, fs)
    xp = T.preparar_para_delinear(sig, fs)
    de = T.delinear_p_st(xp, fs, T.delinear_batimentos(xp, fs, rq, versao=2), st_ms=ST_MS)
    m = MAN[r]
    cobertura.append({'registro': r, 'FA': r in FA, '% P confiável': 100 * de.P_confiavel.mean(), 'batimentos': len(de)})
    metodos = {'Matriz': de}
    if COMPARAR_NK and r not in FA:
        for c in range(sig.shape[1]):
            metodos[f'neurokit2 wavelet (derivação {c + 1})'] = p_nk(sig, fs, rq, c)
    for nome, d_ in metodos.items():
        e = T.comparar_delineacao(d_, m, fs, pontos=('Pon', 'Ppico', 'Pfim'), extras=('P_confiavel',))
        if e.empty:
            continue
        e.insert(0, 'metodo', nome); e.insert(0, 'registro', r)
        erros.append(e)
    # erro no ST60 causado pelo erro do J
    rd = de.R.to_numpy()
    for _, mm in m[m.QRSfim.notna()].iterrows():
        k = int(np.argmin(np.abs(rd - mm.R)))
        if abs(rd[k] - mm.R) > 0.15 * fs:
            continue
        b = de.iloc[k]
        st_esp = T.st_no_ponto_j(xp, fs, int(b.R), int(mm.QRSfim), ST_MS)
        st_met = np.array([b[f'ST_d{c + 1}'] for c in range(xp.shape[1])])
        st_err.append({'registro': r, 'erro_J_ms': (b.QRSfim - mm.QRSfim) / fs * 1000,
                       **{f'ST_d{c + 1}_especialista': st_esp[c] for c in range(xp.shape[1])},
                       **{f'erro_ST_d{c + 1}_mV': st_met[c] - st_esp[c] for c in range(xp.shape[1])}})
    print(f'  {r}: {len(de)} batimentos, P confiável em {100 * de.P_confiavel.mean():.0f} %')
err = pd.concat(erros, ignore_index=True)
err.to_csv(os.path.join(PASTA_CSV, 'p_st_erros.csv'), index=False)
st_err = pd.DataFrame(st_err)
st_err.to_csv(os.path.join(PASTA_CSV, 'st_erro_por_j.csv'), index=False)
cobertura = pd.DataFrame(cobertura).set_index('registro')""")

md(r"""## 4. Resumo da onda P
`média` = viés; `DP` = dispersão; `mediana |erro|` = batimento típico; `passa` = DP ≤ tolerância CSE (início 10,2 ms; fim 12,7 ms).
**Matriz confiável** = só batimentos com `P_confiavel` (proeminência da P ≥ 4 % do QRS). Os trechos em fibrilação atrial não entram
aqui (não há P anotada neles); eles aparecem na tabela de cobertura.""")
code(r"""def resumo_p(g, rotulo):
    linhas = []
    for p in ['Pon', 'Ppico', 'Pfim']:
        ref = g[f'{p}_tem_ref'].astype('boolean').fillna(False).astype(bool)
        v = g[p][ref].dropna(); n_ref = int(ref.sum())
        if n_ref == 0:
            continue
        tol = T.TOL_CSE_MS.get(p)
        linhas.append({'metodo': rotulo, 'ponto': p, 'n_ref': n_ref, '% marcado': 100 * len(v) / n_ref, 'média (ms)': v.mean(),
                       'DP (ms)': v.std(ddof=1), 'mediana |erro| (ms)': v.abs().median(), 'tol. CSE (ms)': tol,
                       'passa': (v.std(ddof=1) <= tol) if tol else None})
    return linhas
ok_ = err[err.casou]
linhas = []
for nome, g in ok_.groupby('metodo'):
    linhas += resumo_p(g, nome)
    if nome == 'Matriz':
        gc = g[g.P_confiavel.astype('boolean').fillna(False).astype(bool)]
        for l in resumo_p(gc, 'Matriz confiável'):
            l['% batimentos confiáveis'] = 100 * len(gc) / max(len(g), 1); linhas.append(l)
res_p = pd.DataFrame(linhas)
res_p.to_csv(os.path.join(PASTA_CSV, 'p_st_resumo.csv'), index=False)
display(res_p.round(1))
print('\nCobertura da P confiável: ritmo sinusal × fibrilação atrial (na FA deve ser baixa)')
display(cobertura.groupby('FA')['% P confiável'].describe().round(1))""")

code(r"""fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
cores = {'Matriz': 'tab:orange', 'neurokit2 wavelet (derivação 1)': 'tab:gray', 'neurokit2 wavelet (derivação 2)': 'darkgray'}
mc = ok_[(ok_.metodo == 'Matriz') & ok_.P_confiavel.astype('boolean').fillna(False).astype(bool)]
for a, p in zip(axs, ['Pon', 'Ppico', 'Pfim']):
    for nome, g in ok_.groupby('metodo'):
        v = g[p].dropna()
        if len(v):
            a.hist(v, bins=np.arange(-100, 101, 4), histtype='step', lw=2.2, color=cores.get(nome, 'k'),
                   ls='-' if nome == 'Matriz' else ':', label=f'{nome} (n = {len(v)})')
    v = mc[p].dropna()
    a.hist(v, bins=np.arange(-100, 101, 4), histtype='step', lw=2.2, color='tab:red', label=f'Matriz, P confiável (n = {len(v)})')
    tol = T.TOL_CSE_MS.get(p)
    if tol:
        a.axvspan(-tol, tol, color='tab:green', alpha=0.12, label=f'± tolerância CSE ({T.num(tol, 1)} ms)')
    a.axvline(0, color='k', lw=1)
    a.set_title(NOMES_P[p]); a.set_xlabel('Erro: detectado − especialista (ms)'); T.negrito_ticks(a)
axs[0].set_ylabel('Batimentos')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.09), fontsize=11)
plt.tight_layout(rect=(0, 0.09, 1, 1)); plt.savefig(os.path.join(PASTA_FIG, 'fig_p_erros.png')); plt.show()""")

md(r"""## 5. ST60: quanto o erro do ponto J muda o ST
Para cada derivação: o erro no ST60 (mV) entre medir no J do método e no J do especialista. A faixa verde é ±0,05 mV (metade do
critério clínico de 1 mm). **Isto mede só o efeito do J**: o nível isoelétrico e o sinal são os mesmos nos dois lados.""")
code(r"""nc = len([c for c in st_err.columns if c.startswith('erro_ST_d')])
linhas = []
for c in range(nc):
    e = st_err[f'erro_ST_d{c + 1}_mV'].dropna()
    linhas.append({'derivação': c + 1, 'n': len(e), 'viés (mV)': e.mean(), 'mediana |erro| (mV)': e.abs().median(),
                   'p95 |erro| (mV)': np.percentile(e.abs(), 95), f'% com |erro| < {T.num(REF_ST_MV / 2, 2)} mV': 100 * np.mean(e.abs() < REF_ST_MV / 2),
                   f'% com |erro| < {T.num(REF_ST_MV, 2)} mV': 100 * np.mean(e.abs() < REF_ST_MV)})
res_st = pd.DataFrame(linhas)
res_st.to_csv(os.path.join(PASTA_CSV, 'st_resumo.csv'), index=False)
display(res_st.round(3))
fig, axs = plt.subplots(1, nc + 1, figsize=(6 * (nc + 1), 5))
for c in range(nc):
    a = axs[c]; e = st_err[f'erro_ST_d{c + 1}_mV'].dropna()
    a.hist(e, bins=np.arange(-0.3, 0.31, 0.01), color='tab:red', alpha=0.75)
    a.axvspan(-REF_ST_MV / 2, REF_ST_MV / 2, color='tab:green', alpha=0.15, label='± 0,05 mV')
    a.axvline(0, color='k', lw=1)
    a.set_title(f'Erro no ST{ST_MS}, derivação {c + 1}'); a.set_xlabel('mV'); T.negrito_ticks(a)
axs[0].set_ylabel('Batimentos'); axs[0].legend(loc='upper left')
a = axs[-1]
for c, cor in zip(range(nc), ['tab:blue', 'tab:gray']):
    a.scatter(st_err.erro_J_ms, st_err[f'erro_ST_d{c + 1}_mV'], s=10, alpha=0.5, color=cor, label=f'Derivação {c + 1}')
a.axhspan(-REF_ST_MV / 2, REF_ST_MV / 2, color='tab:green', alpha=0.12); a.axhline(0, color='k', lw=1); a.axvline(0, color='k', lw=1)
a.set_xlabel('Erro do ponto J (ms)'); a.set_ylabel('Erro no ST (mV)'); a.set_title('Erro do J × erro no ST')
a.legend(loc='upper left'); T.negrito_ticks(a)
plt.tight_layout(); plt.savefig(os.path.join(PASTA_FIG, 'fig_st_erro.png')); plt.show()""")

md("## 6. Por registro")
code(r"""mat = ok_[ok_.metodo == 'Matriz']
por_reg = mat.groupby('registro').agg(n_P=('Pon', lambda x: x.notna().sum()), vies_Pon=('Pon', 'mean'), dp_Pon=('Pon', 'std'),
                                      vies_Pfim=('Pfim', 'mean'), dp_Pfim=('Pfim', 'std'))
por_reg = por_reg.join(cobertura[['FA', '% P confiável']], how='outer')
por_reg = por_reg.join(st_err.groupby('registro').agg(**{'mediana |erro ST d1| (mV)': ('erro_ST_d1_mV', lambda x: x.abs().median())}), how='left')
por_reg.round(2).to_csv(os.path.join(PASTA_CSV, 'p_st_por_registro.csv'))
display(por_reg.round(2))""")

md(r"""## O que me mandar
1. A figura `fig_p_metodo.png` e a linha do ST60 abaixo dela.
2. A tabela `res_p` e a cobertura ritmo sinusal × FA (seção 4) e `fig_p_erros.png`.
3. A tabela `res_st` e `fig_st_erro.png` (seção 5) — **isto diz se o ST60 do método é utilizável**.
4. `por_reg` (seção 6).

**Próximo passo (notebook 13):** aplicar na SDDB inteira — séries batimento a batimento de ST60 e da forma da P — e a **recorrência
ST × P** (a P como controle interno): última hora antes da FV contra o período basal, com a NSRDB como controle.""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}, "colab": {"provenance": []}}
nbf.write(nb, "/home/claude/tcc/12_onda_p_st_qtdb.ipynb")
print("ok")
