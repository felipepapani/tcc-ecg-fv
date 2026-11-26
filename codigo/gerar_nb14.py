import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md(r"""# 14 — Figuras e números finais do TCC (com a coerência corrigida)

**Por que este notebook existe.** Ele junta, num lugar só, as figuras e os números que vão para o texto, a partir dos notebooks
10, 11, 12, 13 e 15. E corrige um problema achado no fim do notebook 15:

> A **coerência** (correlação de forma entre batimentos) dava valor alto a trechos **sem onda nenhuma**. A spline zera a linha de base
> no ponto isoelétrico logo antes do QRS, então um trecho plano vira uma rampa que termina em zero, e rampas iguais se correlacionam
> bem. Na QT Database, um registro em FA (sel36) tinha coerência da P de 0,69. **Tirando a reta de cada derivação antes de
> correlacionar** ("sem tendência"), ele cai para 0,05, e os sinusais continuam altos (mediana 0,86). AUC sinusal × FA: 0,96 → 1,00.

**O que muda:** as duas coerências (do ST e da P) passam a ser calculadas sem tendência. A taxa de recorrência, o determinismo e a
recorrência conjunta usam distância euclidiana, que inclui a amplitude, então não têm esse artefato. Ficam iguais. O ST60 (achado
principal) não é afetado, porque é um nível num ponto.

**Seções**
1. Validação da coerência da P na QT Database: original × sem tendência (e o filtro de RR como sensibilidade).
2. Uma passada pelos 31 registros (SDDB + NSRDB) com a coerência nova (~1 h a 1 h 30; salva por registro, retoma de onde parou).
3. Efeitos finais (família de 11 medidas com Holm), testes dentro dos grupos, ajuste por ruído e FC, retirada de um registro.
4. Figuras finais em português (vírgula decimal), numa pasta só: `tcc_final/figs`.
5. Tabela única de números do TCC: `tcc_final/csvs/numeros_tcc.csv`.""")

md("## 0. Configuração")
code(r"""import os, sys, time, importlib, subprocess, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import wfdb
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'wfdb'], check=True)
    import wfdb
from wfdb import processing
from scipy.stats import wilcoxon, mannwhitneyu, spearmanr

EM_COLAB = os.path.isdir('/content')
if EM_COLAB and not os.path.isdir('/content/drive/MyDrive'):
    from google.colab import drive
    drive.mount('/content/drive')
BASE = os.environ.get('TCC_BASE') or ('/content/drive/MyDrive/Pesquisa/TCC' if EM_COLAB else os.path.abspath('TCC'))
PASTA_CODIGO = os.path.join(BASE, 'codigo')
PASTA_SDDB, PASTA_NSRDB, PASTA_QTDB = (os.path.join(BASE, x) for x in ('SDDB_atr', 'NSRDB_atr', 'QTDB'))
PASTA_DET = os.path.join(BASE, 'tcc_qrs', 'deteccoes')
CSV_QRS, CSV_DELIN, CSV_PST = (os.path.join(BASE, x, 'csvs') for x in ('tcc_qrs', 'tcc_delin', 'tcc_p_st'))
JAN_15 = os.path.join(BASE, 'tcc_sens', 'janelas', 'principal13')          # janelas do notebook 15 (para conferência)
PASTA_FINAL = os.path.join(BASE, 'tcc_final')
PASTA_JAN, PASTA_CSV, PASTA_FIG = (os.path.join(PASTA_FINAL, x) for x in ('janelas', 'csvs', 'figs'))
PASTA_SINAL = '/content/sinais' if EM_COLAB else os.path.join(BASE, 'sinais')
SINAL = {'sddb': os.path.join(PASTA_SINAL, 'sddb'), 'nsrdb': os.path.join(PASTA_SINAL, 'nsrdb')}
PASTAS_LOCAIS = {'sddb': ['/content/drive/MyDrive/Pesquisa/SDDB_hrv'], 'nsrdb': []}
for p in (PASTA_SDDB, PASTA_NSRDB, PASTA_QTDB, PASTA_DET, PASTA_JAN, PASTA_CSV, PASTA_FIG, *SINAL.values()):
    os.makedirs(p, exist_ok=True)
sys.path.insert(0, PASTA_CODIGO)
import tcc_lib as T
importlib.reload(T)
T.estilo_figuras()
assert hasattr(T, 'MEDIDAS_ST_P_FINAL'), 'Copie o tcc_lib.py novo para a pasta codigo/ antes de rodar.'

def pasta_do_sinal(base, r):
    for p in PASTAS_LOCAIS.get(base, []):
        if os.path.exists(os.path.join(p, f'{r}.dat')) and os.path.exists(os.path.join(p, f'{r}.hea')):
            return p
    return SINAL[base]

HORAS, TERMINAL, BASAL, BASAL_SENS, GATE, SEED = 24, (-60, 0), -120, -360, 0.80, 0
PAR = dict(win_s=300, min_bat=100, eps_q=0.10)          # configuração principal (notebooks 13 e 15)
MED = list(T.MEDIDAS_ST_P_FINAL)
ANTIGAS = ['COER_ST', 'COER_P']
ALTERN = ['FC', 'RR_dp', 'ruido', 'frac_valida', 'ALT_ST']
NOMES = {'ST60_d1': 'ST60, derivação 1', 'ST60_d2': 'ST60, derivação 2', 'ST_dist': 'ST60, módulo (2 derivações)',
         'COER_ST_semtend': 'Coerência do ST', 'COER_P_semtend': 'Coerência da P',
         'COER_ST': 'Coerência do ST (versão antiga)', 'COER_P': 'Coerência da P (versão antiga)',
         'RR_ST': 'Recorrência do ST', 'DET_ST': 'Determinismo do ST', 'RR_P': 'Recorrência da P', 'DET_P': 'Determinismo da P',
         'JRR': 'Recorrência conjunta ST × P', 'RAZAO_JRR': 'Razão conjunta ST × P',
         'FC': 'FC (bpm)', 'RR_dp': 'DP do RR (ms)', 'ruido': 'Ruído > 15 Hz (mV)', 'frac_valida': 'Fração de batimentos válidos',
         'ALT_ST': 'Alternância do ST'}
NUMEROS = []                                             # (bloco, item, valor) — vira a tabela da seção 5
def guarda(bloco, item, valor):
    NUMEROS.append({'bloco': bloco, 'item': item, 'valor': valor})
def p_txt(p):
    return 'p < 0,001' if p < 0.001 else f'p = {T.num(p, 3)}'
print('Pronto. Base:', BASE)""")

md(r"""## 1. Validação da coerência da P na QT Database
Mesma conta que você rodou: quatro variantes da coerência da P em cada registro (15 min), sinusal × FA.
- **original**: a do notebook 13;
- **RR ≥ 700 ms**: só batimentos com o RR anterior longo o bastante para a janela da P (que começa ~290 ms antes do R) não pegar a onda T
  do batimento anterior (que termina ~420 ms depois do R anterior);
- **sem tendência**: a versão adotada;
- **os dois**.

Mínimo de 30 batimentos; abaixo disso fica sem dado (registros taquicárdicos perdem todos os batimentos no filtro de RR).""")
code(r"""FA = {'sel35', 'sel36', 'sel37', 'sel50'}
N_MIN = 30
VARS = ['original', 'RR ≥ 700 ms', 'sem tendência', 'os dois']

def coer_var(Vp, nc, rr_prev, rr_min=None, sem_tend=False):
    V = Vp[rr_prev >= rr_min] if rr_min else Vp
    V = V[np.isfinite(V).all(axis=1)]
    if len(V) < N_MIN:
        return np.nan, len(V)
    return T.coerencia(V, nc=nc, sem_tendencia=sem_tend), len(V)

arq_val = os.path.join(PASTA_CSV, 'validacao_coerencia_p_qtdb.csv')
if os.path.exists(arq_val):
    vq = pd.read_csv(arq_val, index_col=0)
else:
    linhas = []
    for r in [f'sel{n}' for n in range(30, 53)] + ['sel17152']:
        if not os.path.exists(os.path.join(PASTA_QTDB, f'{r}.dat')):
            continue
        rec = wfdb.rdrecord(os.path.join(PASTA_QTDB, r)); fs = float(rec.fs); sig = np.nan_to_num(rec.p_signal)
        de, Vst, Vp = T.extrair_batimentos(sig, fs, processing.xqrs_detect(sig[:, 0], fs=fs, verbose=False))
        rr_prev = np.r_[np.nan, np.diff(de.R.to_numpy()) / fs * 1000]
        nc = sig.shape[1]
        c0, n0 = coer_var(Vp, nc, rr_prev); c1, n1 = coer_var(Vp, nc, rr_prev, 700)
        c2, _ = coer_var(Vp, nc, rr_prev, sem_tend=True); c3, _ = coer_var(Vp, nc, rr_prev, 700, True)
        linhas.append({'registro': r, 'ritmo': 'FA' if r in FA else 'sinusal', 'RR mediano (ms)': np.nanmedian(rr_prev),
                       'n total': n0, 'n RR≥700': n1, 'original': c0, 'RR ≥ 700 ms': c1, 'sem tendência': c2, 'os dois': c3})
        print(f'  {r}: ok')
    vq = pd.DataFrame(linhas).set_index('registro')
    vq.to_csv(arq_val)
display(vq.round(3))

auc = {}
for c in VARS:
    s, f = vq[vq.ritmo == 'sinusal'][c].dropna(), vq[vq.ritmo == 'FA'][c].dropna()
    if len(s) and len(f):
        u = mannwhitneyu(s, f)
        auc[c] = dict(s=s.median(), f=f.median(), ns=len(s), nf=len(f), auc=u.statistic / (len(s) * len(f)), p=u.pvalue)
        print(f"{c:15s} sinusal {T.num(s.median(), 2)} (n = {len(s)}) × FA {T.num(f.median(), 2)} (n = {len(f)}) | "
              f"AUC = {T.num(auc[c]['auc'], 2)} | {p_txt(u.pvalue)}")
for c in ('original', 'sem tendência'):
    if c in auc:
        a = auc[c]
        guarda('Validação da coerência da P (QTDB)', f'{c}: mediana sinusal × FA',
               f"{T.num(a['s'], 2)} (n = {a['ns']}) × {T.num(a['f'], 2)} (n = {a['nf']})")
        guarda('Validação da coerência da P (QTDB)', f'{c}: AUC', f"{T.num(a['auc'], 2)} ({p_txt(a['p'])})")
for r in ('sel36', 'sel38'):
    if r in vq.index:
        guarda('Validação da coerência da P (QTDB)', f'{r} ({vq.loc[r, "ritmo"]}): original → sem tendência',
               f"{T.num(vq.loc[r, 'original'], 2)} → {T.num(vq.loc[r, 'sem tendência'], 2)}")

fig, axs = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
for a, c in zip(axs, ['original', 'sem tendência']):
    for k, (rit, cor) in enumerate([('sinusal', 'tab:blue'), ('FA', 'tab:red')]):
        g = vq[vq.ritmo == rit][c].dropna()
        x = k + np.random.default_rng(k).uniform(-0.1, 0.1, len(g))
        a.scatter(x, g, s=60, color=cor, zorder=3, edgecolor='k', lw=0.5)
        if len(g): a.hlines(g.median(), k - 0.28, k + 0.28, color='k', lw=3)
        for xi, (r, v) in zip(x, g.items()):
            if r in ('sel36', 'sel38'):
                a.annotate(r, (xi, v), xytext=(10, 0), textcoords='offset points', va='center', fontsize=11, fontweight='bold')
    a.axhline(0, color='0.6', ls=':')
    tit = c.capitalize() + (f"\nAUC = {T.num(auc[c]['auc'], 2)}" if c in auc else '')
    a.set_title(tit); a.set_xticks([0, 1]); a.set_xticklabels(['Sinusal', 'Fibrilação atrial']); a.set_xlim(-0.6, 1.6)
    T.negrito_ticks(a)
axs[0].set_ylabel('Coerência da onda P')
plt.tight_layout(); plt.savefig(os.path.join(PASTA_FIG, 'fig_final_validacao_coerencia_p.png')); plt.show()""")

md(r"""## 2. Uma passada pelos 31 registros com a coerência corrigida
Mesmo processamento do notebook 15 (configuração principal), agora com as colunas `COER_ST_semtend` e `COER_P_semtend`. Usa os
batimentos do xqrs já salvos no notebook 10. Salva um CSV por registro em `tcc_final/janelas`; se cair, rode a célula de novo
que ela continua de onde parou.

No fim, **confere** que a coerência antiga sai igual à do notebook 15 (mesmo código, mesmos batimentos): a diferença deve ser ~0.""")
code(r"""meta = T.SDDB_META.copy()
REG_SDDB = meta[(meta.rhythm == 'sinus') & (~meta.paced) & meta.vf_onset_s.notna()].index.tolist()
ok_s = T.baixar_wfdb('sddb', REG_SDDB, PASTA_SDDB, exts=('hea',))
info = {r: T.info_header(os.path.join(PASTA_SDDB, r)) for r in ok_s}
NSRDB_LISTA = ['16265', '16272', '16273', '16420', '16483', '16539', '16773', '16786', '16795',
               '17052', '17453', '18177', '18184', '19088', '19090', '19093', '19140', '19830']
nsr_ok = T.baixar_wfdb('nsrdb', NSRDB_LISTA, PASTA_NSRDB)
info_nsr = {x: T.info_header(os.path.join(PASTA_NSRDB, x)) for x in nsr_ok}
pseudo = T.pseudo_eventos(nsr_ok, {x: h['dur_s'] for x, h in info_nsr.items()}, meta.vf_onset_s.dropna().tolist(), seed=SEED)
ALVOS = [('sddb', r, info[r]['vfon_s']) for r in ok_s] + [('nsrdb', r, pseudo[r]) for r in nsr_ok]
arq_jan = lambda b, r: os.path.join(PASTA_JAN, f'{b}_{r}.csv')
pend = [(b, r) for b, r, _ in ALVOS if not os.path.exists(arq_jan(b, r))]
for base in ('sddb', 'nsrdb'):
    baixar = [r for b, r in pend if b == base and pasta_do_sinal(base, r) == SINAL[base]]
    if baixar:
        print(f'Baixando o sinal de {len(baixar)} registros da {base.upper()}…'); T.baixar_wfdb(base, baixar, SINAL[base], exts=('hea', 'dat'))

def batimentos_xqrs(base, r, pasta, t_evento):
    arq = os.path.join(PASTA_DET, f'{base}_{r}_xqrswfdb.npz')
    if os.path.exists(arq):
        z = np.load(arq); return np.asarray(z['marcas'], int), float(z['fs'])
    marcas, fs = [], None
    for s in np.arange(max(0.0, t_evento - HORAS * 3600 - 60), t_evento, 3600):
        e = min(s + 3600, t_evento)
        sig, fs, a0 = T.ler_sinal(base, r, pasta, s - 5, e + 5)
        q = processing.xqrs_detect(sig[:, 0], fs=fs, verbose=False) + a0
        marcas.extend(q[(q >= s * fs) & (q < e * fs)].tolist())
    return np.array(sorted(marcas), int), fs

def processar(base, r, t_evento):
    if os.path.exists(arq_jan(base, r)):
        return
    pasta = pasta_do_sinal(base, r)
    marcas, fs = batimentos_xqrs(base, r, pasta, t_evento)
    tabs, VS, VP = [], [], []
    for s in np.arange(max(0.0, t_evento - HORAS * 3600 - 60), t_evento, 3600):
        e = min(s + 3600, t_evento)
        sig, fs, a0 = T.ler_sinal(base, r, pasta, s - 5, e + 5)
        loc = marcas[(marcas >= a0) & (marcas < a0 + len(sig))] - a0
        de, Vs, Vp = T.extrair_batimentos(sig, fs, loc)
        if de.empty:
            continue
        for c in ('R', 'QRSon', 'QRSfim', 'iso'):
            de[c] = de[c] + a0
        dentro = ((de.R >= s * fs) & (de.R < e * fs)).to_numpy()
        tabs.append(de[dentro]); VS.append(Vs[dentro]); VP.append(Vp[dentro])
    de = pd.concat(tabs, ignore_index=True); Vs = np.vstack(VS); Vp = np.vstack(VP)
    valido = T.validade_batimentos(de.R.to_numpy(), marcas, fs)
    tab = T.tabela_janelas(de, Vs, Vp, valido, t_evento, fs, base, r, horas=HORAS, gate=GATE, basal_min=BASAL, extras=True, **PAR)
    tab.to_csv(arq_jan(base, r), index=False)

t0 = time.time()
for k, (base, r, tev) in enumerate(ALVOS, 1):
    t1 = time.time()
    try:
        processar(base, r, tev)
        print(f'  [{k}/{len(ALVOS)}] {base} {r}: {time.time() - t1:.0f} s (total {(time.time() - t0) / 60:.1f} min)')
    except Exception as e:
        print(f'  [{k}/{len(ALVOS)}] {base} {r}: FALHOU ({type(e).__name__}: {e}) — rode de novo esta célula')
jan = pd.concat([pd.read_csv(arq_jan(b, r)) for b, r, _ in ALVOS if os.path.exists(arq_jan(b, r))], ignore_index=True)
jan.to_csv(os.path.join(PASTA_CSV, 'janelas_final.csv'), index=False)
print(f"{len(jan)} janelas, {jan.passed.mean() * 100:.0f} % aprovadas")

arqs15 = [os.path.join(JAN_15, f'{b}_{r}.csv') for b, r, _ in ALVOS]
if all(os.path.exists(a) for a in arqs15):
    j15 = pd.concat([pd.read_csv(a) for a in arqs15], ignore_index=True)
    m = jan.merge(j15, on=['base', 'registro', 't_inicio_s'], suffixes=('', '_15'))
    for c in ('COER_ST', 'ST60_d2', 'RR_ST', 'RAZAO_JRR'):
        dif = np.abs(m[c] - m[c + "_15"])
        print(f'Conferência com o notebook 15, {c}: diferença mediana {np.nanmedian(dif):.1e}, maior {np.nanmax(dif):.1e}')
    print('(esperado: ~0 em tudo. Se o ST60_d2 diferir, os batimentos mudaram entre os dois notebooks: me avise antes de seguir.)')
else:
    print('(janelas do notebook 15 não encontradas; conferência pulada)')

pa = jan[jan.passed]
print('\nQuanto a correção mudou a coerência, por janela aprovada (mediana [p25; p75]):')
for b in ('SDDB', 'NSRDB'):
    g = pa[pa.base == b]
    for c in ('COER_ST', 'COER_P'):
        d = g[f'{c}_semtend'] - g[c]
        print(f'  {b} {c}: antiga {T.num(g[c].median(), 2)} → sem tendência {T.num(g[c + "_semtend"].median(), 2)} '
              f'(diferença {T.num(d.median(), 3)} {T.ic(d.quantile(0.25), d.quantile(0.75), 3)})')""")

md(r"""## 3. Efeitos finais
Efeito por registro = mediana da última hora − mediana do basal (≥ 3 janelas de cada lado). SDDB × NSRDB por Mann–Whitney com Hedges g
(IC 95 % por bootstrap); **Holm sobre as 11 medidas da família final**. As coerências antigas aparecem só como referência (fora do Holm).""")
code(r"""efeitos, EF = [], {}
for rot, lim in [('basal antes de −2 h', BASAL), ('basal antes de −6 h', BASAL_SENS)]:
    e = T.efeito_ultima_hora(jan, lim, TERMINAL, MED + ANTIGAS + ALTERN)
    EF[rot] = e
    S, N = e[e.base == 'SDDB'], e[e.base == 'NSRDB']
    tab = pd.DataFrame([{'basal': rot, 'medida': mm, 'n_SDDB': S[f'{mm}_delta'].notna().sum(), 'n_NSRDB': N[f'{mm}_delta'].notna().sum(),
                         'mediana Δ SDDB': S[f'{mm}_delta'].median(), 'mediana Δ NSRDB': N[f'{mm}_delta'].median(),
                         **T.hedges_g_boot(S[f'{mm}_delta'], N[f'{mm}_delta'])} for mm in MED + ANTIGAS])
    fam = tab.medida.isin(MED)
    tab['p_holm'] = np.nan
    tab.loc[fam, 'p_holm'] = T.holm(tab.loc[fam, 'p_mw'].to_numpy())
    tab['família'] = np.where(fam, 'final', 'antiga (referência)')
    efeitos.append(tab)
efe = pd.concat(efeitos, ignore_index=True)
efe.to_csv(os.path.join(PASTA_CSV, 'efeitos_final.csv'), index=False)
pd.set_option('display.max_rows', 100)
display(efe[['basal', 'medida', 'família', 'n_SDDB', 'n_NSRDB', 'mediana Δ SDDB', 'mediana Δ NSRDB', 'g', 'ic_lo', 'ic_hi',
             'p_mw', 'p_holm']].round(4))

e2 = EF['basal antes de −2 h']
S, N = e2[e2.base == 'SDDB'], e2[e2.base == 'NSRDB']
linhas = []
for bse, g in (('SDDB', S), ('NSRDB', N)):
    for mm in MED + ANTIGAS + ALTERN:
        v = g[f'{mm}_delta'].dropna()
        p = wilcoxon(v).pvalue if len(v) >= 6 and (v != 0).any() else np.nan
        linhas.append({'grupo': bse, 'medida': mm, 'n': len(v), 'mediana do Δ': v.median(), 'Δ > 0 (registros)': int((v > 0).sum()),
                       'p Wilcoxon': p})
dentro = pd.DataFrame(linhas)
dentro.to_csv(os.path.join(PASTA_CSV, 'dentro_dos_grupos_final.csv'), index=False)
print('Dentro de cada grupo (última hora × basal, Wilcoxon pareado):')
display(dentro.pivot(index='medida', columns='grupo', values=['mediana do Δ', 'p Wilcoxon']).round(4))""")

md(r"""**Explicações alternativas e influência.** Para as coerências e a razão conjunta: a diferença entre grupos continua depois de
descontar a mudança de ruído e de FC (regressão do Δ no grupo + Δruído + ΔFC, IC por bootstrap)? E tirando um registro da SDDB por vez,
o g muda de sinal ou o IC passa a cruzar o zero?""")
code(r"""def ajuste(alvo, n_boot=2000):
    d = e2[['base', f'{alvo}_delta', 'ruido_delta', 'FC_delta']].dropna()
    if len(d) < 8:
        return dict(medida=alvo, n=len(d), coef=np.nan, ic_lo=np.nan, ic_hi=np.nan)
    X = np.c_[np.ones(len(d)), (d.base == 'SDDB').astype(float), d.ruido_delta, d.FC_delta]
    y = d[f'{alvo}_delta'].to_numpy()
    coef = np.linalg.lstsq(X, y, rcond=None)[0][1]
    rng = np.random.default_rng(SEED); bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(d), len(d))
        bs.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0][1])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(medida=alvo, n=len(d), coef=coef, ic_lo=lo, ic_hi=hi)
ajt = pd.DataFrame([ajuste(a) for a in ['ST60_d2', 'COER_ST_semtend', 'COER_P_semtend', 'COER_ST', 'DET_ST', 'RAZAO_JRR']])
ajt.to_csv(os.path.join(PASTA_CSV, 'ajuste_ruido_fc_final.csv'), index=False)
print('Diferença SDDB − NSRDB no Δ, descontando Δruído e ΔFC:'); display(ajt.round(4))

linhas = []
for alvo in ['COER_ST_semtend', 'COER_P_semtend', 'RAZAO_JRR']:
    for mm in ALTERN:
        d = S[[f'{alvo}_delta', f'{mm}_delta']].dropna()
        rho, p = spearmanr(d.iloc[:, 0], d.iloc[:, 1]) if len(d) >= 6 else (np.nan, np.nan)
        linhas.append({'Δ de': alvo, 'contra Δ de': mm, 'n': len(d), 'rho Spearman': rho, 'p': p})
corr = pd.DataFrame(linhas)
corr.to_csv(os.path.join(PASTA_CSV, 'correlacoes_sddb_final.csv'), index=False)
print('Dentro da SDDB (Spearman):'); display(corr.round(3))

linhas = []
regs = sorted(S.registro.unique())
for mm in ['ST60_d2', 'COER_ST_semtend', 'COER_P_semtend', 'DET_ST', 'RAZAO_JRR']:
    gs, fora = [], 0
    for r in regs:
        h = T.hedges_g_boot(S[S.registro != r][f'{mm}_delta'], N[f'{mm}_delta'], n_boot=2000)
        gs.append(h['g']); fora += int(h['ic_lo'] > 0 or h['ic_hi'] < 0)
    linhas.append({'medida': mm, 'g mínimo': np.nanmin(gs), 'g máximo': np.nanmax(gs), 'IC sem zero (de)': f'{fora}/{len(regs)}'})
influ = pd.DataFrame(linhas)
influ.to_csv(os.path.join(PASTA_CSV, 'influencia_final.csv'), index=False)
print('Tirando um registro da SDDB por vez:'); display(influ.round(2))""")

md(r"""## 4. Figuras finais
Todas em `tcc_final/figs`, em português e com vírgula decimal:
1. `fig_final_qrs.png`: F1 dos detectores de QRS na SDDB (fora da última hora × última hora);
2. `fig_final_delineacao.png`: dispersão do erro de cada ponto contra a tolerância CSE;
3. `fig_final_validacao_coerencia_p.png`: feita na seção 1;
4. `fig_final_efeitos.png`: Hedges g com IC das 11 medidas, com destaque para as que passam em Holm;
5. `fig_final_delta_por_registro.png`: o Δ de cada registro nas medidas principais;
6. `fig_final_trajetorias.png`: as 6 horas antes do evento, mediana e intervalo interquartil entre registros.""")
code(r"""quebra = lambda s, n=22: '\n'.join(textwrap.wrap(str(s), n))

# 1. QRS
arq = os.path.join(CSV_QRS, 'qrs_validacao_resumo.csv')
if os.path.exists(arq):
    q = pd.read_csv(arq)
    q = q[q.tolerancia == '±150 ms']
    vari = list(dict.fromkeys(q.variante))
    fig, a = plt.subplots(figsize=(max(10, 2.2 * len(vari)), 5.5))
    x = np.arange(len(vari)); w = 0.38
    for k, (uh, rot, cor) in enumerate([(False, 'Fora da última hora', 'tab:blue'), (True, 'Última hora antes da FV', 'tab:red')]):
        v = [q[(q.variante == n) & (q.ultima_hora.astype(str) == str(uh))]['F1 (%)'].mean() for n in vari]
        b = a.bar(x + (k - 0.5) * w, v, w, color=cor, label=rot)
        for xi, vi in zip(x + (k - 0.5) * w, v):
            if np.isfinite(vi): a.text(xi, vi + 0.3, T.num(vi, 1), ha='center', va='bottom', fontsize=10.5, fontweight='bold')
    a.set_xticks(x); a.set_xticklabels([quebra(n, 18) for n in vari]); a.set_ylabel('F1 (%)')
    lo = np.nanmin(q['F1 (%)']) if len(q) else 80
    a.set_ylim(max(0, np.floor(lo / 5) * 5 - 5), 101.5)
    a.axhline(99, color='0.4', ls=':', lw=1.5, label='Critério de escolha: F1 ≥ 99 %')
    a.legend(loc='lower left'); T.negrito_ticks(a)
    plt.tight_layout(); plt.savefig(os.path.join(PASTA_FIG, 'fig_final_qrs.png')); plt.show()
    for n in vari:
        for uh, rot in ((False, 'fora da última hora'), (True, 'última hora')):
            l = q[(q.variante == n) & (q.ultima_hora.astype(str) == str(uh))]
            if len(l):
                l = l.iloc[0]
                guarda('Detecção de QRS (SDDB, ±150 ms)', f'{n}, {rot}: Se / VPP / F1 (%)',
                       f"{T.num(l['Se (%)'], 1)} / {T.num(l['VPP (%)'], 1)} / {T.num(l['F1 (%)'], 1)}")
else:
    print('Sem', arq, '— rode o notebook 10.')

# 2. Delineação: DP do erro ÷ tolerância CSE (≤ 1 passa)
partes = []
arq_d, arq_p = os.path.join(CSV_DELIN, 'delin_resumo.csv'), os.path.join(CSV_PST, 'p_st_resumo.csv')
if os.path.exists(arq_d):
    d = pd.read_csv(arq_d)
    d = d[(d.conjunto == 'não vistos') & d.ponto.isin(['QRSon', 'QRSfim', 'Tfim'])]
    partes.append(d)
if os.path.exists(arq_p):
    partes.append(pd.read_csv(arq_p).query("ponto in ['Pon', 'Pfim']")
                  .replace({'metodo': {'Matriz': 'Matriz v2', 'Matriz confiável': 'Matriz v2 confiável'}}))
if partes:
    dl = pd.concat(partes, ignore_index=True)
    dl = dl[dl['tol. CSE (ms)'].notna() & (dl.metodo != 'Matriz v1')]      # v1 fica no notebook 11
    TETO = 3.0
    pontos = [p for p in ['Pon', 'Pfim', 'QRSon', 'QRSfim', 'Tfim'] if p in set(dl.ponto)]
    NP = {'Pon': 'Início da P', 'Pfim': 'Fim da P', 'QRSon': 'Início do QRS', 'QRSfim': 'Fim do QRS (J)', 'Tfim': 'Fim da T'}
    metodos = list(dict.fromkeys(dl.metodo))
    cores = plt.cm.tab10(np.arange(len(metodos)))
    fig, a = plt.subplots(figsize=(14, 6))
    w = 0.8 / len(metodos)
    com_rotulo = set()
    for k, (mt, cor) in enumerate(zip(metodos, cores)):
        for i, p in enumerate(pontos):
            l = dl[(dl.metodo == mt) & (dl.ponto == p)]
            if len(l):
                l = l.iloc[0]; r_ = min(l['DP (ms)'] / l['tol. CSE (ms)'], TETO)
                a.bar(i + (k - (len(metodos) - 1) / 2) * w, r_, w, color=cor, label=None if mt in com_rotulo else mt)
                com_rotulo.add(mt)
                a.text(i + (k - (len(metodos) - 1) / 2) * w, r_ + 0.05, T.num(l['DP (ms)'], 0), ha='center', fontsize=9.5)
                guarda('Delineação (QTDB; QRS e T: registros não vistos)', f'{NP[p]}, {mt}: DP do erro (tolerância CSE)',
                       f"{T.num(l['DP (ms)'], 1)} ms ({T.num(l['tol. CSE (ms)'], 1)} ms)")
    a.axhline(1, color='k', ls='--', lw=1.5); a.text(len(pontos) - 0.5, 1.03, 'tolerância CSE', ha='right', fontsize=11)
    a.set_xticks(range(len(pontos))); a.set_xticklabels([NP[p] for p in pontos]); a.set_ylim(0, TETO + 0.35)
    a.set_ylabel('DP do erro ÷ tolerância CSE\n(números: DP em ms; barras cortadas em 3)')
    a.legend(loc='upper left', fontsize=10.5); T.negrito_ticks(a)
    plt.tight_layout(); plt.savefig(os.path.join(PASTA_FIG, 'fig_final_delineacao.png')); plt.show()
else:
    print('Sem os resumos dos notebooks 11 e 12.')
arq = os.path.join(CSV_PST, 'st_resumo.csv')
if os.path.exists(arq):
    for _, l in pd.read_csv(arq).iterrows():
        guarda('ST60 × erro do ponto J (QTDB)', f"derivação {int(l['derivação'])}: mediana |erro| / p95 |erro|",
               f"{T.num(l['mediana |erro| (mV)'], 3)} / {T.num(l['p95 |erro| (mV)'], 3)} mV")""")

code(r"""# 4. Forest plot das 11 medidas (basal −2 h)
f2 = efe[(efe.basal == 'basal antes de −2 h') & (efe['família'] == 'final')].set_index('medida').loc[MED]
fig, a = plt.subplots(figsize=(10, 7.5))
y = np.arange(len(MED))[::-1]
for yi, (mm, l) in zip(y, f2.iterrows()):
    sig_ = l.p_holm < 0.05
    cor = 'tab:red' if sig_ else '0.45'
    a.errorbar(l.g, yi, xerr=[[l.g - l.ic_lo], [l.ic_hi - l.g]], fmt='o', color=cor, ecolor=cor, capsize=4, ms=9 if sig_ else 7,
               mfc=cor if sig_ else 'white', lw=2)
    a.text(max(f2.ic_hi.max(), 0) + 0.15, yi, f'p Holm {T.num(l.p_holm, 3)}', va='center', fontsize=10.5,
           fontweight='bold' if sig_ else 'normal')
a.axvline(0, color='0.5', ls=':')
a.set_yticks(y); a.set_yticklabels([NOMES[m] for m in MED])
a.set_xlabel('Hedges g (SDDB − NSRDB) do efeito "última hora − basal"')
a.set_xlim(min(f2.ic_lo.min(), 0) - 0.3, max(f2.ic_hi.max(), 0) + 1.3)
T.negrito_ticks(a)
plt.tight_layout(); plt.savefig(os.path.join(PASTA_FIG, 'fig_final_efeitos.png')); plt.show()

# 5. Δ por registro nas medidas principais
PRINC = ['ST60_d2', 'COER_ST_semtend', 'COER_P_semtend', 'RAZAO_JRR']
fig, axs = plt.subplots(1, len(PRINC), figsize=(5 * len(PRINC), 5.5))
for a, mm in zip(axs, PRINC):
    for k, (bse, cor) in enumerate([('NSRDB', 'tab:blue'), ('SDDB', 'tab:red')]):
        v = e2[e2.base == bse][f'{mm}_delta'].dropna()
        a.scatter(np.full(len(v), k) + np.random.default_rng(k).uniform(-0.09, 0.09, len(v)), v, s=50, color=cor, zorder=3,
                  edgecolor='k', lw=0.5)
        if len(v): a.hlines(v.median(), k - 0.27, k + 0.27, color='k', lw=3)
    a.axhline(0, color='0.5', ls=':')
    l = f2.loc[mm]
    a.set_title(f"{quebra(NOMES[mm], 26)}\ng = {T.num(l.g)} {T.ic(l.ic_lo, l.ic_hi)}", fontsize=12)
    a.set_xticks([0, 1]); a.set_xticklabels(['Controles\n(NSRDB)', 'Antes da FV\n(SDDB)']); a.set_xlim(-0.6, 1.6); T.negrito_ticks(a)
axs[0].set_ylabel('Última hora − basal')
plt.tight_layout(); plt.savefig(os.path.join(PASTA_FIG, 'fig_final_delta_por_registro.png')); plt.show()

# 6. Trajetórias de 6 h (mediana e IIQ entre registros)
j6 = jan[jan.passed & (jan.min_ate_evento > -360)].assign(hora=lambda d: np.floor((d.min_ate_evento - 1e-6) / 60))
fig, axs = plt.subplots(1, len(PRINC), figsize=(5 * len(PRINC), 5))
for a, mm in zip(axs, PRINC):
    for bse, cor, rot in [('SDDB', 'tab:red', 'Antes da FV (SDDB)'), ('NSRDB', 'tab:blue', 'Controles (NSRDB)')]:
        g = j6[j6.base == bse].groupby(['registro', 'hora'])[mm].median().groupby('hora')
        med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
        a.plot(med.index + 0.5, med.values, '-o', color=cor, lw=2.2, label=rot)
        a.fill_between(med.index + 0.5, q1.values, q3.values, color=cor, alpha=0.15)
    a.set_xlabel('Horas até o evento'); a.set_title(quebra(NOMES[mm], 26)); T.negrito_ticks(a)
axs[0].legend(fontsize=10.5)
plt.tight_layout(); plt.savefig(os.path.join(PASTA_FIG, 'fig_final_trajetorias.png')); plt.show()""")

md(r"""## 5. Tabela única de números do TCC
Tudo o que o texto cita, num CSV só (`tcc_final/csvs/numeros_tcc.csv`), já formatado em português.""")
code(r"""for rot in ('basal antes de −2 h', 'basal antes de −6 h'):
    for mm in MED:
        l = efe[(efe.basal == rot) & (efe.medida == mm)].iloc[0]
        guarda(f'Efeito última hora − basal ({rot})', NOMES[mm],
               f"Δ SDDB {T.num(l['mediana Δ SDDB'], 3)} × NSRDB {T.num(l['mediana Δ NSRDB'], 3)}; g = {T.num(l.g)} "
               f"{T.ic(l.ic_lo, l.ic_hi)}; Holm {p_txt(l.p_holm)} (n = {int(l.n_SDDB)} × {int(l.n_NSRDB)})")
for mm in MED + ANTIGAS:
    for bse in ('SDDB', 'NSRDB'):
        l = dentro[(dentro.grupo == bse) & (dentro.medida == mm)].iloc[0]
        if np.isfinite(l['p Wilcoxon']):
            guarda('Dentro do grupo (Wilcoxon, basal −2 h)', f'{NOMES[mm]}, {bse}',
                   f"mediana Δ {T.num(l['mediana do Δ'], 3)}; {l['Δ > 0 (registros)']}/{l.n} sobem; {p_txt(l['p Wilcoxon'])}")
for _, l in ajt.iterrows():
    if np.isfinite(l.coef):
        guarda('Ajuste por Δruído e ΔFC (SDDB − NSRDB)', NOMES[l.medida], f"{T.num(l.coef, 3)} {T.ic(l.ic_lo, l.ic_hi, 3)} (n = {l.n})")
for _, l in influ.iterrows():
    guarda('Retirando um registro da SDDB por vez', NOMES[l.medida],
           f"g de {T.num(l['g mínimo'])} a {T.num(l['g máximo'])}; IC sem zero em {l['IC sem zero (de)']}")
for mm in ALTERN:
    l = T.hedges_g_boot(S[f'{mm}_delta'], N[f'{mm}_delta'])
    guarda('Variáveis alternativas (SDDB − NSRDB)', NOMES[mm], f"g = {T.num(l['g'])} {T.ic(l['ic_lo'], l['ic_hi'])}")
num_tab = pd.DataFrame(NUMEROS)
num_tab.to_csv(os.path.join(PASTA_CSV, 'numeros_tcc.csv'), index=False)
pd.set_option('display.max_colwidth', 200)
for b, g in num_tab.groupby('bloco', sort=False):
    print(f'\n■ {b}')
    for _, l in g.iterrows():
        print(f'   {l["item"]}: {l["valor"]}')""")

md(r"""## O que me mandar
1. Seção 2: as três linhas de **conferência** com o notebook 15 e o bloco "quanto a correção mudou a coerência".
2. Seção 3: a tabela `efe`, a tabela "dentro de cada grupo", a do ajuste, a de Spearman e a de influência.
3. Seção 4: as figuras `fig_final_*.png` (se alguma ficar feia, me diga qual).
4. Seção 5: o texto impresso da tabela de números (é ela que vira o capítulo de resultados).""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}, "colab": {"provenance": []}}
nbf.write(nb, "/home/claude/tcc/14_figuras_numeros_st_p.ipynb")
print("ok")
