import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md(r"""# 11 — Início e fim da onda T (e do QRS) a partir da matriz de distância — validação na QT Database

**O que a Nicole fez:** localizou o QRS (≈ 99 % de acurácia) e o *pico* da onda T (≈ 76 % na PTB), pela matriz de distância.
**O que falta e este notebook faz:** as **bordas** — início e fim do QRS (o fim é o **ponto J**) e **início, pico e fim da T**.
Sem elas não existe segmento ST (de J até o início da T) nem intervalo QT.

**A ideia (continuação direta da matriz da Nicole):**
1. Em cada batimento, acha-se a **linha isoelétrica** `b`: o trecho de 20 ms mais "parado" do segmento PR (o coração em repouso).
2. Calcula-se `d(n) = ‖x(n) − b‖` para as amostras do batimento. Isso é **uma coluna da matriz de distância** — a do instante
   isoelétrico: a distância de cada amostra ao estado de repouso. Em termos de recorrência: as ondas são os trechos **não recorrentes
   com o repouso**.
3. Por ser uma norma, `d(n)` é sempre positivo: **uma T invertida e uma T positiva viram a mesma "corcova"**, e as duas derivações
   entram juntas (uma T positiva numa derivação e negativa na outra não se cancelam).
4. Sobre essa corcova, as bordas saem por dois métodos, comparados:
   - **trapézio** (Vázquez-Seisdedos et al., 2011): o fim da onda é o "joelho" onde a descida termina — o ponto que maximiza a área
     de um trapézio entre o ponto de descida mais íngreme e um ponto depois da onda;
   - **faixa** (o jeito da Nicole, com limiar): a borda é onde a corcova cruza 20 % da sua altura — equivale à largura da faixa da T na
     matriz binarizada.

**Por que a QT Database:** ela contém **24 trechos de 15 min da própria SDDB** (`sel30`–`sel52`, `sel17152`), mesmo Holter, mesmas
derivações, 250 Hz, com 30–50 batimentos por registro marcados **à mão** por especialistas (início, pico e fim de P, QRS e T).
É a validação nos mesmos pacientes em que o TCC vai aplicar o método.

**Critério de aceitação (definido antes de ver os resultados):** as tolerâncias do *CSE Working Party* (1985), usadas como referência
em todos os delineadores publicados: desvio-padrão do erro ≤ 2·s_CSE — QRSon 6,5 ms; QRSfim 11,6 ms; Tfim 30,6 ms (não há tolerância
CSE para início da T, que é mal definido até para especialistas — por isso a base quase não o anota).

**Comparação:** o delineador por wavelet do `neurokit2` (método de Martínez et al., 2004), um padrão da literatura, nos mesmos R.

**Saídas:** `delin_erros.csv`, `delin_resumo.csv`, `delin_por_registro.csv`; figuras `fig_delin_metodo.png`, `fig_delin_erros.png`,
`fig_delin_registros.png`.""")

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

EM_COLAB = os.path.isdir('/content')
if EM_COLAB and not os.path.isdir('/content/drive/MyDrive'):
    from google.colab import drive
    drive.mount('/content/drive')
BASE = os.environ.get('TCC_BASE') or ('/content/drive/MyDrive/Pesquisa/TCC' if EM_COLAB else os.path.abspath('TCC'))
PASTA_CODIGO = os.path.join(BASE, 'codigo')
PASTA_QTDB   = os.path.join(BASE, 'QTDB')            # ~0,7 MB por registro (15 min)
PASTA_CSV    = os.path.join(BASE, 'tcc_delin', 'csvs')
PASTA_FIG    = os.path.join(BASE, 'tcc_delin', 'figs')
for p in (PASTA_QTDB, PASTA_CSV, PASTA_FIG):
    os.makedirs(p, exist_ok=True)
sys.path.insert(0, PASTA_CODIGO)
import tcc_lib as T
importlib.reload(T)
T.estilo_figuras()

PARAM_NICOLE = dict(q=0.95, frac=0.75, eps_s=0.100, min_s=0.002)
ALFA_FAIXA = 0.20             # limiar da variante "faixa": 20 % da altura da corcova
COMPARAR_NK = True            # delineador wavelet do neurokit2 nos mesmos R
ANCORA = 'xqrs'               # R usados como âncora da delineação: 'xqrs' (padrão; o QRS da Nicole perde batimentos no Holter, notebook 10) ou 'nicole'
PONTOS = ['QRSon', 'QRSfim', 'Ton', 'Tpico', 'Tfim']
NOMES_P = {'QRSon': 'Início do QRS', 'QRSfim': 'Fim do QRS (ponto J)', 'Ton': 'Início da T', 'Tpico': 'Pico da T',
           'Tfim': 'Fim da T', 'QT': 'Intervalo QT'}""")

md(r"""## 1. QT Database — os 24 trechos da SDDB
Baixa `.hea`, `.dat` e `.q1c` (anotador 1, revisado). Conta quantos batimentos têm cada ponto anotado: o anotador nem sempre marca
todas as bordas (o início da T, em especial, é raro).""")
code(r"""QT_SDDB = [f'sel{n}' for n in range(30, 53)] + ['sel17152']
ok = T.baixar_wfdb('qtdb', QT_SDDB, PASTA_QTDB, exts=('hea', 'dat', 'q1c'))
print(f'{len(ok)} de {len(QT_SDDB)} registros baixados')
MAN, INFO = {}, {}
for r in ok:
    h = wfdb.rdheader(os.path.join(PASTA_QTDB, r))
    INFO[r] = {'fs': float(h.fs), 'derivacoes': ', '.join(h.sig_name)}
    MAN[r] = T.ler_marcas_qtdb(os.path.join(PASTA_QTDB, r), 'q1c')
cont = pd.DataFrame({r: {'derivações': INFO[r]['derivacoes'], 'batimentos': len(m),
                         **{f'{p} anotado': int(m[p].notna().sum()) for p in ['QRSon', 'QRSfim', 'Ton', 'Tpico', 'Tfim', 'Pon', 'Pfim']}}
                     for r, m in MAN.items()}).T
display(cont)
print('Total de batimentos anotados:', int(cont['batimentos'].sum()))""")

md(r"""## 2. O método passo a passo num batimento real
Figura para o capítulo de método: as duas derivações, a curva `d(n)` (distância ao repouso) e as marcas — trapézio, faixa e o
especialista.""")
code(r"""r = ok[0]
fs = INFO[r]['fs']
rec = wfdb.rdrecord(os.path.join(PASTA_QTDB, r))
sig = np.nan_to_num(rec.p_signal)
from wfdb import processing
def ancoras_r(sig, fs):
    if ANCORA == 'xqrs':
        return np.asarray(processing.xqrs_detect(sig[:, 0], fs=fs, verbose=False))
    return T.detectar_qrs_matriz(sig, fs, **PARAM_NICOLE)['centroide']
r_anc = ancoras_r(sig, fs)
xprep = T.preparar_para_delinear(sig, fs)
de = T.delinear_batimentos(xprep, fs, r_anc, alfa_faixa=ALFA_FAIXA)
m = MAN[r]
m_ok = m[m.Tfim.notna()]
alvo = m_ok.iloc[len(m_ok) // 2] if len(m_ok) else m.iloc[len(m) // 2]
j = int(np.argmin(np.abs(de.R - alvo.R))); b = de.iloc[j]
a0, a1 = int(b.R - 0.3 * fs), int(b.R + 0.75 * fs)
tt = (np.arange(a0, a1) - b.R) / fs * 1000
nivel, _ = T.ponto_isoeletrico(xprep, int(b.R), fs)
d = np.sqrt(((xprep[a0:a1] - nivel) ** 2).sum(axis=1))

fig, axs = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
for c, cor in zip(range(xprep.shape[1]), ['tab:blue', 'tab:gray']):
    axs[0].plot(tt, xprep[a0:a1, c], color=cor, lw=1.6, label=f'Derivação {c + 1} ({rec.sig_name[c]})')
axs[0].axvspan((b.iso - b.R) / fs * 1000 - 10, (b.iso - b.R) / fs * 1000 + 10, color='tab:green', alpha=0.25,
               label='Linha isoelétrica (PR)')
axs[0].set_ylabel('Amplitude (mV)'); axs[0].set_title(f'(a) Batimento do registro {r}')
axs[1].plot(tt, d, color='k', lw=1.8, label='d(n) = distância ao repouso')
marcas = [('QRSon', 'tab:purple', '-'), ('QRSfim', 'tab:purple', '-'), ('Ton', 'tab:red', '-'), ('Tpico', 'tab:red', ':'),
          ('Tfim', 'tab:red', '-'), ('Ton_faixa', 'darkorange', '--'), ('Tfim_faixa', 'darkorange', '--')]
for k, (p, cor, ls) in enumerate(marcas):
    if np.isfinite(b[p]):
        for a in axs:
            a.axvline((b[p] - b.R) / fs * 1000, color=cor, ls=ls, lw=1.8,
                      label=({'QRSon': 'Bordas do QRS (trapézio)', 'Ton': 'Bordas da T (trapézio)', 'Tpico': 'Pico da T',
                              'Ton_faixa': f'Bordas da T (faixa, {T.num(100 * ALFA_FAIXA, 0)} %)'}.get(p) if a is axs[1] else None))
for p in ['QRSon', 'QRSfim', 'Ton', 'Tpico', 'Tfim']:
    if np.isfinite(alvo[p]):
        axs[1].plot((alvo[p] - b.R) / fs * 1000, 0, marker='^', ms=13, color='tab:green', clip_on=False,
                    label='Especialista (QT Database)' if p == 'QRSon' else None)
axs[1].set_xlabel('Tempo em relação ao R (ms)'); axs[1].set_ylabel('Distância (mV)')
axs[1].set_title('(b) Coluna da matriz de distância no instante isoelétrico')
for a in axs:
    a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13 if a is axs[0] else -0.2), ncol=4, fontsize=11); T.negrito_ticks(a)
plt.tight_layout(); plt.savefig(os.path.join(PASTA_FIG, 'fig_delin_metodo.png')); plt.show()""")

md(r"""## 3. Todos os registros: QRS da Nicole → delineação → comparação com o especialista
Para cada registro: detecção dos QRS, delineação (trapézio e faixa) e, se `COMPARAR_NK`, o delineador wavelet do `neurokit2`
em cada derivação separadamente (ele trabalha com uma derivação por vez), usando os **mesmos R**.""")
code(r"""def delinear_nk(sig, fs, r_amostras, canal):
    import neurokit2 as nk
    limpo = nk.ecg_clean(sig[:, canal], sampling_rate=int(fs))
    try:
        _, w = nk.ecg_delineate(limpo, rpeaks=np.asarray(r_amostras), sampling_rate=int(fs), method='dwt')
    except Exception as e:
        print(f'   neurokit2 falhou ({type(e).__name__})'); return pd.DataFrame()
    mapa = {'QRSon': 'ECG_R_Onsets', 'QRSfim': 'ECG_R_Offsets', 'Ton': 'ECG_T_Onsets', 'Tpico': 'ECG_T_Peaks',
            'Tfim': 'ECG_T_Offsets'}
    out = pd.DataFrame({'R': np.asarray(r_amostras)})
    for p, k in mapa.items():
        v = np.asarray(w.get(k, [np.nan] * len(out)), float)
        out[p] = v[:len(out)] if len(v) >= len(out) else np.r_[v, [np.nan] * (len(out) - len(v))]
    return out

# A v2 foi desenhada olhando um batimento destes 6 registros (figura de diagnóstico); os outros 18 nunca foram
# olhados batimento a batimento: o resultado neles é o teste mais limpo possível nesta base.
VISTOS = {'sel30', 'sel41', 'sel45', 'sel46', 'sel48', 'sel50'}
erros, qrs_ok = [], []
for r in ok:
    fs = INFO[r]['fs']
    sig = np.nan_to_num(wfdb.rdrecord(os.path.join(PASTA_QTDB, r)).p_signal)
    rq_nicole = T.detectar_qrs_matriz(sig, fs, **PARAM_NICOLE)['centroide']
    rq = ancoras_r(sig, fs) if ANCORA == 'xqrs' else rq_nicole
    m = MAN[r]
    cmp_r = T.comparar_com_referencia(rq_nicole, m.R.to_numpy(), fs, 0.150)   # QRS da Nicole nos batimentos anotados
    qrs_ok.append({'registro': r, 'anotados': len(m), 'encontrados': cmp_r['VP']})
    xp = T.preparar_para_delinear(sig, fs)
    metodos = {'Matriz v1': T.delinear_batimentos(xp, fs, rq, alfa_faixa=ALFA_FAIXA),
               'Matriz v2': T.delinear_batimentos(xp, fs, rq, versao=2)}
    if COMPARAR_NK:
        for c in range(sig.shape[1]):
            metodos[f'neurokit2 wavelet (derivação {c + 1})'] = delinear_nk(sig, fs, rq, c)
    for nome, de in metodos.items():
        pts = PONTOS + (['Ton_faixa', 'Tfim_faixa'] if nome == 'Matriz v1' else [])
        e = T.comparar_delineacao(de, m, fs, pontos=pts, extras=('T_confiavel',))
        if e.empty:
            continue
        if 'QRSon' in de and 'Tfim' in de:            # intervalo QT derivado
            de = de.assign(QT=de.Tfim - de.QRSon)
            mq = m.assign(QT=m.Tfim - m.QRSon)
            e['QT'] = T.comparar_delineacao(de[['R', 'QT']], mq[['R', 'QT']], fs, pontos=('QT',))['QT'].to_numpy()
        e.insert(0, 'metodo', nome); e.insert(0, 'registro', r)
        e.insert(2, 'conjunto', 'vistos' if r in VISTOS else 'não vistos')
        erros.append(e)
    print(f'  {r}: {cmp_r["VP"]}/{len(m)} QRS anotados encontrados')
err = pd.concat(erros, ignore_index=True)
err.to_csv(os.path.join(PASTA_CSV, 'delin_erros.csv'), index=False)
qrs_ok = pd.DataFrame(qrs_ok)
print(f"\nQRS da Nicole nos batimentos anotados: {qrs_ok.encontrados.sum()}/{qrs_ok.anotados.sum()} "
      f"({100 * qrs_ok.encontrados.sum() / qrs_ok.anotados.sum():.1f} %)")""")

md(r"""## 4. Resumo: erro (detectado − especialista) por ponto e método
- `n_ref` = batimentos com aquele ponto anotado; `% marcado` = em quantos deles o método também marcou;
- `média` = viés; `DP` = dispersão; `mediana |erro|` = o batimento típico (o DP é puxado pelos erros grosseiros);
  `tol. CSE` = 2·s_CSE; `passa` = DP ≤ tolerância;
- **v1** = versão original (trapézio; `*_faixa` = variante com limiar); **v2** = janela fisiológica + pico mais proeminente
  (ver `delinear_batimentos`); **v2 confiável** = só os batimentos com `T_confiavel` (T com proeminência ≥ 10 % do QRS);
- `conjunto`: **não vistos** = os 18 registros que não foram olhados para desenhar a v2 (o teste honesto); **todos** = os 24.""")
code(r"""def resumir(g, rotulo):
    linhas = []
    for p in PONTOS + ['Ton_faixa', 'Tfim_faixa', 'QT']:
        col = f'{p}_tem_ref'
        if p not in g or g[p].isna().all() or (p != 'QT' and (col not in g or g[col].isna().all())):
            continue
        ref = (g[col] if p != 'QT' else (g['QRSon_tem_ref'] & g['Tfim_tem_ref'])).astype('boolean').fillna(False).astype(bool)
        v = g[p][ref].dropna()
        n_ref = int(ref.sum())
        if n_ref == 0:
            continue
        tol = T.TOL_CSE_MS.get(p.replace('_faixa', ''))
        linhas.append({'metodo': rotulo, 'ponto': p, 'n_ref': n_ref, '% marcado': 100 * len(v) / n_ref,
                       'média (ms)': v.mean(), 'DP (ms)': v.std(ddof=1), 'mediana |erro| (ms)': v.abs().median(),
                       'tol. CSE (ms)': tol, 'passa': (v.std(ddof=1) <= tol) if tol else None})
    return linhas

ok_ = err[err.casou]
linhas = []
for conj, sub in [('todos', ok_), ('não vistos', ok_[ok_.conjunto == 'não vistos'])]:
    for nome, g in sub.groupby('metodo'):
        for l in resumir(g, nome):
            linhas.append({'conjunto': conj, **l})
        if nome == 'Matriz v2':
            gc = g[g['T_confiavel'].astype('boolean').fillna(False).astype(bool)]
            for l in resumir(gc, 'Matriz v2 confiável'):
                l['% batimentos confiáveis'] = 100 * len(gc) / max(len(g), 1)
                linhas.append({'conjunto': conj, **l})
res = pd.DataFrame(linhas)
res.to_csv(os.path.join(PASTA_CSV, 'delin_resumo.csv'), index=False)
pd.set_option('display.max_rows', 200)
display(res[res.ponto.isin(['QRSon', 'QRSfim', 'Tpico', 'Tfim', 'QT'])].round(1))""")

code(r"""fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
cores = {'Matriz v1': 'tab:blue', 'Matriz v2': 'tab:red', 'neurokit2 wavelet (derivação 1)': 'tab:gray',
         'neurokit2 wavelet (derivação 2)': 'darkgray'}
for a, p in zip(axs, ['QRSfim', 'Tpico', 'Tfim']):
    for nome, g in ok_.groupby('metodo'):
        v = g[p].dropna()
        if len(v):
            a.hist(v, bins=np.arange(-150, 151, 6), histtype='step', lw=2.2, color=cores.get(nome, 'k'),
                   ls='-' if 'Matriz' in nome else ':', label=f'{nome} (n = {len(v)})')
    tol = T.TOL_CSE_MS.get(p)
    if tol:
        a.axvspan(-tol, tol, color='tab:green', alpha=0.12, label=f'± tolerância CSE ({T.num(tol, 1)} ms)')
    a.axvline(0, color='k', lw=1)
    a.set_title(NOMES_P[p]); a.set_xlabel('Erro: detectado − especialista (ms)'); T.negrito_ticks(a)
axs[0].set_ylabel('Batimentos')
h, l = axs[2].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.08), fontsize=11)
plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig(os.path.join(PASTA_FIG, 'fig_delin_erros.png')); plt.show()""")

md(r"""## 5. Onde o método erra: fim da T por registro
Se o erro se concentra em poucos registros (T achatada, ruído, extrassístoles), isso aparece aqui — e vira critério de qualidade
para a aplicação na SDDB inteira.""")
code(r"""def por_registro(nome):
    d = ok_[ok_.metodo == nome]
    return d.groupby('registro').agg(n=('Tfim', lambda x: x.notna().sum()), vies_Tfim=('Tfim', 'mean'),
                                     dp_Tfim=('Tfim', 'std'), vies_J=('QRSfim', 'mean'), dp_J=('QRSfim', 'std'))
por_reg = pd.concat({'v1': por_registro('Matriz v1'), 'v2': por_registro('Matriz v2')}, axis=1).round(1)
conf = ok_[ok_.metodo == 'Matriz v2'].groupby('registro')['T_confiavel'].apply(lambda x: 100 * x.astype(bool).mean()).round(0)
por_reg[('v2', '% T confiável')] = conf
por_reg[('', 'conjunto')] = ['vistos' if r in VISTOS else 'não vistos' for r in por_reg.index]
por_reg.to_csv(os.path.join(PASTA_CSV, 'delin_por_registro.csv'))
display(por_reg)
fig, a = plt.subplots(figsize=(16, 5.5))
regs = list(por_reg.index)
for k, rg in enumerate(regs):
    for nome, cor, dx in [('Matriz v1', 'tab:blue', -0.18), ('Matriz v2', 'tab:red', 0.18)]:
        d = ok_[(ok_.metodo == nome) & (ok_.registro == rg)]
        v = d.Tfim.dropna()
        a.scatter(np.full(len(v), k + dx) + np.random.default_rng(k).uniform(-0.08, 0.08, len(v)), v, s=12, alpha=0.6,
                  color=cor, label=nome if k == 0 else None)
a.axhspan(-T.TOL_CSE_MS['Tfim'], T.TOL_CSE_MS['Tfim'], color='tab:green', alpha=0.12)
a.axhline(0, color='k', lw=1)
a.set_xticks(range(len(regs)))
a.set_xticklabels([f'{r}*' if r in VISTOS else r for r in regs], rotation=60, fontsize=11)
a.set_ylabel('Erro no fim da T (ms)'); a.set_title('Erro do fim da T por registro (* = olhado para desenhar a v2)')
a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2); T.negrito_ticks(a)
plt.tight_layout(); plt.savefig(os.path.join(PASTA_FIG, 'fig_delin_registros.png')); plt.show()""")

md(r"""## 6. Um batimento de cada tipo, v1 × v2
Linhas finas = v1, grossas = v2; triângulos = especialista. Mostra em que corcova cada versão marcou a T.""")
code(r"""regs = [r for r in ['sel30', 'sel45', 'sel41', 'sel48', 'sel46', 'sel50'] if r in INFO]
fig, axs = plt.subplots(len(regs), 1, figsize=(14, 3.2 * len(regs)))
for a, r in zip(np.atleast_1d(axs), regs):
    fs = INFO[r]['fs']
    sig = np.nan_to_num(wfdb.rdrecord(os.path.join(PASTA_QTDB, r)).p_signal)
    rq = ancoras_r(sig, fs)
    xp = T.preparar_para_delinear(sig, fs)
    d1, d2 = T.delinear_batimentos(xp, fs, rq), T.delinear_batimentos(xp, fs, rq, versao=2)
    m = MAN[r]; m = m[m.Tfim.notna()]
    alvo = m.iloc[len(m) // 2]
    j = int(np.argmin(np.abs(d1.R - alvo.R))); b1, b2 = d1.iloc[j], d2.iloc[int(np.argmin(np.abs(d2.R - alvo.R)))]
    a0, a1 = int(b1.R - 0.3 * fs), int(b1.R + 0.8 * fs)
    tt = (np.arange(a0, a1) - b1.R) / fs * 1000
    nivel, _ = T.ponto_isoeletrico(xp, int(b1.R), fs)
    dd = np.sqrt(((xp[a0:a1] - nivel) ** 2).sum(axis=1))
    for c, cor in zip(range(xp.shape[1]), ['tab:blue', 'tab:gray']):
        a.plot(tt, xp[a0:a1, c], color=cor, lw=1.3, label=f'Derivação {c + 1}')
    a.plot(tt, dd, color='k', lw=2, label='d(n)')
    for b, lw, rot in [(b1, 1.2, 'v1'), (b2, 3, 'v2')]:
        for p, ls in [('Tpico', ':'), ('Tfim', '-')]:
            if np.isfinite(b[p]):
                a.axvline((b[p] - b1.R) / fs * 1000, color='tab:red', ls=ls, lw=lw)
    y0 = a.get_ylim()[0]
    for p in ['Tpico', 'Tfim']:
        if np.isfinite(alvo[p]): a.plot((alvo[p] - b1.R) / fs * 1000, y0, '^', ms=12, color='tab:green', clip_on=False)
    e1 = (b1.Tfim - alvo.Tfim) / fs * 1000; e2 = (b2.Tfim - alvo.Tfim) / fs * 1000
    a.set_title(f"{r}: erro no fim da T  v1 = {T.num(e1, 0)} ms · v2 = {T.num(e2, 0)} ms"
                f"{'' if b2.T_confiavel else ' (v2: T não confiável)'}", fontsize=13)
    a.set_ylabel('mV')
np.atleast_1d(axs)[-1].set_xlabel('Tempo em relação ao R (ms)')
plt.tight_layout(); plt.savefig(os.path.join(PASTA_FIG, 'fig_delin_diagnostico.png')); plt.show()""")

md(r"""## O que me mandar
1. A tabela `cont` (seção 1) e a figura `fig_delin_metodo.png`.
2. A linha "QRS da Nicole nos batimentos anotados" (seção 3).
3. A tabela `res` (seção 4) e `fig_delin_erros.png` — **isto decide se a v2 segue para o ST** (olhe primeiro as linhas
   `não vistos`).
4. `por_reg`, `fig_delin_registros.png` e `fig_delin_diagnostico.png`.

**Próximo passo, se passar:** segmento ST = de J (`QRSfim`) até o início da T, com o desvio do ST medido como distância ao repouso
em J + 60 ms; depois a onda P (mesma ideia, numa janela antes do QRS, validada com `Pon`/`Pfim` da mesma base) e a recorrência
ST × P.""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}, "colab": {"provenance": []}}
nbf.write(nb, "/home/claude/tcc/11_delineacao_t_qtdb.ipynb")
print("ok")
