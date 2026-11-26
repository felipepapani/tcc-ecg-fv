import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md(r"""# 15 — Sensibilidade e explicações alternativas do sinal do notebook 13

**O que o notebook 13 achou** (basal antes de −2 h, SDDB 12 × NSRDB 18, Holm para 11 medidas): o ST60 da derivação 2 sobe, a
**coerência** e o **determinismo do ST** caem e a **razão conjunta ST × P** sobe na última hora antes da FV, e não nos controles.

**O que este notebook pergunta, antes de qualquer conclusão:**
1. **O resultado depende das escolhas que fixei?** Raio da recorrência (quantil 5, 10, 20 %), janela (3, 5, 10 min) e mínimo de
   batimentos por janela (50, 100, 150).
2. **A mudança acontece dentro dos próprios pacientes?** Teste pareado (Wilcoxon) da última hora contra o basal, em cada grupo
   separadamente — sem depender da NSRDB.
3. **Explicações alternativas** para a queda de coerência do ST e a subida da razão conjunta:
   - **ruído** (energia acima de 15 Hz fora do QRS): o paciente se mexe, o eletrodo solta perto do evento;
   - **frequência cardíaca** e sua variabilidade: a forma do ST e da P depende da FC, e uma mudança de FC mexe nos dois juntos;
   - **fração de batimentos válidos** (mais ectopia perto da FV);
   - **alternância** do ST (padrão ABAB, a assinatura da alternância da onda T, marcador clássico de risco de FV).
4. **Um registro sozinho segura o resultado?** Tirando um registro da SDDB por vez.
5. Duas pendências: o grupo de janelas em que spline e filtro de mediana discordam no sinal do ST, e o trecho sel36 (FA com P
   coerente).

**Medidas-chave** (as que passaram em Holm no notebook 13): ST60_d2, COER_ST, DET_ST, RAZAO_JRR, e RR_ST (passou com basal −6 h).

**Tempo:** uma passada pelos 31 registros (~1 h a 1 h 30), calculando todas as configurações de uma vez; tudo fica salvo por registro.""")

md("## 0. Configuração")
code(r"""import os, sys, time, importlib, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import wfdb
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'wfdb'], check=True)
    import wfdb
from wfdb import processing
from scipy.stats import wilcoxon, spearmanr

EM_COLAB = os.path.isdir('/content')
if EM_COLAB and not os.path.isdir('/content/drive/MyDrive'):
    from google.colab import drive
    drive.mount('/content/drive')
BASE = os.environ.get('TCC_BASE') or ('/content/drive/MyDrive/Pesquisa/TCC' if EM_COLAB else os.path.abspath('TCC'))
PASTA_CODIGO = os.path.join(BASE, 'codigo')
PASTA_SDDB, PASTA_NSRDB, PASTA_QTDB = (os.path.join(BASE, x) for x in ('SDDB_atr', 'NSRDB_atr', 'QTDB'))
PASTA_DET = os.path.join(BASE, 'tcc_qrs', 'deteccoes')
PASTA_ST_P = os.path.join(BASE, 'tcc_st_p', 'csvs')
PASTA_SENS = os.path.join(BASE, 'tcc_sens')
PASTA_CSV, PASTA_FIG = os.path.join(PASTA_SENS, 'csvs'), os.path.join(PASTA_SENS, 'figs')
PASTA_SINAL = '/content/sinais' if EM_COLAB else os.path.join(BASE, 'sinais')
SINAL = {'sddb': os.path.join(PASTA_SINAL, 'sddb'), 'nsrdb': os.path.join(PASTA_SINAL, 'nsrdb')}
PASTAS_LOCAIS = {'sddb': ['/content/drive/MyDrive/Pesquisa/SDDB_hrv'], 'nsrdb': []}
for p in (PASTA_SDDB, PASTA_NSRDB, PASTA_QTDB, PASTA_DET, PASTA_CSV, PASTA_FIG, *SINAL.values()):
    os.makedirs(p, exist_ok=True)
sys.path.insert(0, PASTA_CODIGO)
import tcc_lib as T
importlib.reload(T)
T.estilo_figuras()

def pasta_do_sinal(base, r):
    for p in PASTAS_LOCAIS.get(base, []):
        if os.path.exists(os.path.join(p, f'{r}.dat')) and os.path.exists(os.path.join(p, f'{r}.hea')):
            return p
    return SINAL[base]

HORAS, TERMINAL, BASAL, GATE, SEED = 24, (-60, 0), -120, 0.80, 0
CONFIGS = {'principal (13)':      dict(win_s=300, min_bat=100, eps_q=0.10),
           'raio 5 %':            dict(win_s=300, min_bat=100, eps_q=0.05),
           'raio 20 %':           dict(win_s=300, min_bat=100, eps_q=0.20),
           'janela 3 min':        dict(win_s=180, min_bat=60,  eps_q=0.10),
           'janela 10 min':       dict(win_s=600, min_bat=200, eps_q=0.10),
           'mínimo 50 batimentos':  dict(win_s=300, min_bat=50,  eps_q=0.10),
           'mínimo 150 batimentos': dict(win_s=300, min_bat=150, eps_q=0.10)}
slug = lambda s: ''.join(ch for ch in s.lower() if ch.isalnum())
KEY = ['ST60_d2', 'COER_ST', 'DET_ST', 'RR_ST', 'RAZAO_JRR']
ALTERN = ['FC', 'RR_dp', 'ruido', 'frac_valida', 'ALT_ST']
NOMES = {'ST60_d2': 'ST60, derivação 2', 'COER_ST': 'Coerência do ST', 'DET_ST': 'Determinismo do ST', 'RR_ST': 'Recorrência do ST',
         'RAZAO_JRR': 'Razão conjunta ST × P', 'FC': 'FC (bpm)', 'RR_dp': 'DP do RR (ms)', 'ruido': 'Ruído > 15 Hz (mV)',
         'frac_valida': 'Fração de batimentos válidos', 'ALT_ST': 'Alternância do ST'}""")

md(r"""## 1. As duas pendências
**(a)** Janelas em que a spline dá ST60 bem negativo (< −0,1 mV) e o filtro de mediana dá positivo, na derivação 1 — de que registros
são. **(b)** O trecho sel36 (FA com coerência da P de 0,69): 40 "ondas P" sobrepostas, comparadas com um trecho sinusal (sel30) e um em
FA típico (sel37).""")
code(r"""jan13 = pd.read_csv(os.path.join(PASTA_ST_P, 'janelas_st_p.csv'))
disc = jan13[jan13.passed & (jan13.ST60_d1 < -0.1) & (jan13.ST60m_d1 > 0)]
print(f'{len(disc)} janelas discordantes; por registro:')
display(disc.groupby(['base', 'registro']).size().rename('janelas').to_frame())

fig, axs = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
for a, r, rot in zip(axs, ['sel30', 'sel36', 'sel37'], ['sinusal', 'FA, coerência 0,69', 'FA típica']):
    if not os.path.exists(os.path.join(PASTA_QTDB, f'{r}.dat')):
        a.set_title(f'{r}: rode o notebook 11 antes'); continue
    rec = wfdb.rdrecord(os.path.join(PASTA_QTDB, r)); fs = float(rec.fs); sig = np.nan_to_num(rec.p_signal)
    rq = processing.xqrs_detect(sig[:, 0], fs=fs, verbose=False)
    de, Vst, Vp = T.extrair_batimentos(sig, fs, rq)
    L = Vp.shape[1] // sig.shape[1]
    tt = np.linspace(-250, -30, L)
    for k in np.linspace(0, len(Vp) - 1, 40).astype(int):
        if np.isfinite(Vp[k]).all():
            a.plot(tt, Vp[k, :L], color='tab:blue', alpha=0.35, lw=1)
    a.set_title(f'{r} ({rot}): coerência {T.num(T.coerencia(Vp), 2)}'); a.set_xlabel('Tempo antes do QRS (ms)'); T.negrito_ticks(a)
axs[0].set_ylabel('Derivação 1 (mV)')
plt.tight_layout(); plt.savefig(os.path.join(PASTA_FIG, 'fig_p_sobreposta.png')); plt.show()""")

md(r"""## 2. Uma passada por registro, todas as configurações
Extrai os batimentos uma vez (mesmo processamento do notebook 13, agora também com o ruído) e calcula as janelas em cada configuração.
A configuração **principal** deve reproduzir o notebook 13.""")
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
arq_cfg = lambda cfg, b, r: os.path.join(PASTA_SENS, 'janelas', slug(cfg), f'{b}_{r}.csv')
for cfg in CONFIGS:
    os.makedirs(os.path.dirname(arq_cfg(cfg, 'x', 'y')), exist_ok=True)
pend = [(b, r) for b, r, _ in ALVOS if not all(os.path.exists(arq_cfg(c, b, r)) for c in CONFIGS)]
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
    if all(os.path.exists(arq_cfg(c, base, r)) for c in CONFIGS):
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
    for cfg, par in CONFIGS.items():
        tab = T.tabela_janelas(de, Vs, Vp, valido, t_evento, fs, base, r, horas=HORAS, gate=GATE, basal_min=BASAL,
                               extras=(cfg == 'principal (13)'), **par)
        tab.to_csv(arq_cfg(cfg, base, r), index=False)

t0 = time.time()
for k, (base, r, tev) in enumerate(ALVOS, 1):
    t1 = time.time()
    try:
        processar(base, r, tev)
        print(f'  [{k}/{len(ALVOS)}] {base} {r}: {time.time() - t1:.0f} s (total {(time.time() - t0) / 60:.1f} min)')
    except Exception as e:
        print(f'  [{k}/{len(ALVOS)}] {base} {r}: FALHOU ({type(e).__name__}: {e}) — rode de novo esta célula')
JAN = {cfg: pd.concat([pd.read_csv(arq_cfg(cfg, b, r)) for b, r, _ in ALVOS if os.path.exists(arq_cfg(cfg, b, r))], ignore_index=True)
       for cfg in CONFIGS}
print({c: round(float(j.passed.mean()), 2) for c, j in JAN.items()})""")

md(r"""## 3. O resultado resiste às escolhas?
Para cada configuração e medida-chave: Hedges g (SDDB − NSRDB) do efeito "última hora − basal", com IC 95 % por bootstrap.
O que importa é a **direção e o tamanho** se manterem; os p não são corrigidos aqui (é estudo de robustez, não um teste novo).""")
code(r"""linhas = []
for cfg, j in JAN.items():
    e = T.efeito_ultima_hora(j, BASAL, TERMINAL, KEY)
    S, N = e[e.base == 'SDDB'], e[e.base == 'NSRDB']
    for mm in KEY:
        linhas.append({'config': cfg, 'medida': mm, **T.hedges_g_boot(S[f'{mm}_delta'], N[f'{mm}_delta'])})
sens = pd.DataFrame(linhas)
sens.to_csv(os.path.join(PASTA_CSV, 'sensibilidade.csv'), index=False)
display(sens.pivot(index='config', columns='medida', values='g').round(2))
fig, axs = plt.subplots(1, len(KEY), figsize=(22, 5.5), sharey=True)
cfgs = list(CONFIGS)
for a, mm in zip(axs, KEY):
    d = sens[sens.medida == mm].set_index('config').loc[cfgs]
    y = np.arange(len(cfgs))[::-1]
    a.errorbar(d.g, y, xerr=[d.g - d.ic_lo, d.ic_hi - d.g], fmt='o', color='tab:red', ecolor='0.4', capsize=4, ms=7)
    a.plot(d.g.iloc[0], y[0], 'D', color='k', ms=9)
    a.axvline(0, color='0.5', ls=':')
    a.set_title(NOMES[mm]); a.set_xlabel('Hedges g (SDDB − NSRDB)'); T.negrito_ticks(a)
axs[0].set_yticks(np.arange(len(cfgs))[::-1]); axs[0].set_yticklabels(cfgs)
plt.tight_layout(); plt.savefig(os.path.join(PASTA_FIG, 'fig_sensibilidade.png')); plt.show()""")

md(r"""## 4. A mudança acontece dentro dos pacientes?
Wilcoxon pareado (última hora × basal) **em cada grupo separadamente**. Se a SDDB muda e a NSRDB não, o efeito é dos pacientes; se a
NSRDB também muda (em sentido contrário), parte da diferença entre grupos vem do controle (relógio, sono).""")
code(r"""ep = T.efeito_ultima_hora(JAN['principal (13)'], BASAL, TERMINAL, KEY + ALTERN)
linhas = []
for bse in ('SDDB', 'NSRDB'):
    g = ep[ep.base == bse]
    for mm in KEY + ALTERN:
        v = g[f'{mm}_delta'].dropna()
        p = wilcoxon(v).pvalue if len(v) >= 6 and (v != 0).any() else np.nan
        linhas.append({'grupo': bse, 'medida': mm, 'n': len(v), 'mediana do Δ': v.median(), 'Δ > 0 (registros)': int((v > 0).sum()),
                       'p Wilcoxon': p})
dentro = pd.DataFrame(linhas)
dentro.to_csv(os.path.join(PASTA_CSV, 'dentro_dos_grupos.csv'), index=False)
display(dentro.pivot(index='medida', columns='grupo', values=['mediana do Δ', 'p Wilcoxon']).round(4))""")

md(r"""## 5. Explicações alternativas
**(a)** As variáveis que poderiam explicar o sinal mudam de forma diferente nos dois grupos? **(b)** Dentro da SDDB, quem teve a
maior queda de coerência do ST foi quem teve mais ruído, mais mudança de FC ou menos batimentos válidos? **(c)** A diferença de
coerência do ST entre grupos continua depois de descontar ruído e FC (regressão do Δ no grupo + Δruído + ΔFC, IC por bootstrap)?""")
code(r"""S, N = ep[ep.base == 'SDDB'], ep[ep.base == 'NSRDB']
alt = pd.DataFrame([{'medida': mm, **T.hedges_g_boot(S[f'{mm}_delta'], N[f'{mm}_delta'])} for mm in ALTERN])
print('(a) Variáveis alternativas, SDDB − NSRDB (Hedges g):'); display(alt.round(3))

linhas = []
for alvo in ['COER_ST', 'RAZAO_JRR']:
    for mm in ['FC', 'RR_dp', 'ruido', 'frac_valida', 'ALT_ST']:
        d = S[[f'{alvo}_delta', f'{mm}_delta']].dropna()
        rho, p = spearmanr(d.iloc[:, 0], d.iloc[:, 1]) if len(d) >= 6 else (np.nan, np.nan)
        linhas.append({'Δ de': alvo, 'contra Δ de': mm, 'n': len(d), 'rho Spearman': rho, 'p': p})
corr = pd.DataFrame(linhas)
print('(b) Dentro da SDDB:'); display(corr.round(3))

d = ep[['base', 'COER_ST_delta', 'ruido_delta', 'FC_delta']].dropna()
X = np.c_[np.ones(len(d)), (d.base == 'SDDB').astype(float), d.ruido_delta, d.FC_delta]
y = d.COER_ST_delta.to_numpy()
coef = np.linalg.lstsq(X, y, rcond=None)[0]
rng = np.random.default_rng(SEED); boots = []
for _ in range(2000):
    i = rng.integers(0, len(d), len(d))
    boots.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0][1])
lo, hi = np.percentile(boots, [2.5, 97.5])
print(f"(c) Diferença de Δcoerência do ST (SDDB − NSRDB) descontando Δruído e ΔFC: {T.num(coef[1], 3)} {T.ic(lo, hi, 3)} (n = {len(d)})")
alt.to_csv(os.path.join(PASTA_CSV, 'alternativas.csv'), index=False); corr.to_csv(os.path.join(PASTA_CSV, 'correlacoes_sddb.csv'), index=False)

fig, axs = plt.subplots(2, 5, figsize=(24, 9))
for i, alvo in enumerate(['COER_ST', 'RAZAO_JRR']):
    for a, mm in zip(axs[i], ['FC', 'RR_dp', 'ruido', 'frac_valida', 'ALT_ST']):
        for bse, cor in [('NSRDB', 'tab:blue'), ('SDDB', 'tab:red')]:
            g = ep[ep.base == bse]
            a.scatter(g[f'{mm}_delta'], g[f'{alvo}_delta'], s=45, color=cor, alpha=0.8, label=bse)
        a.axhline(0, color='0.8'); a.axvline(0, color='0.8')
        a.set_xlabel(f'Δ {NOMES[mm]}'); a.set_ylabel(f'Δ {NOMES[alvo]}' if mm == 'FC' else ''); T.negrito_ticks(a)
axs[0, 0].legend()
plt.tight_layout(); plt.savefig(os.path.join(PASTA_FIG, 'fig_alternativas.png')); plt.show()""")

md(r"""## 6. Um registro sozinho segura o resultado?
Tirando um registro da SDDB por vez: o menor e o maior g e em quantas das retiradas o IC continua sem cruzar o zero.""")
code(r"""linhas = []
regs = sorted(S.registro.unique())
for mm in KEY:
    gs, fora_zero, pior = [], 0, None
    for r in regs:
        s_ = S[S.registro != r][f'{mm}_delta']
        h = T.hedges_g_boot(s_, N[f'{mm}_delta'], n_boot=2000)
        gs.append(h['g']); fora_zero += int(h['ic_lo'] > 0 or h['ic_hi'] < 0)
        if pior is None or abs(h['g']) < abs(pior[1]):
            pior = (r, h['g'])
    linhas.append({'medida': mm, 'g mínimo': min(gs), 'g máximo': max(gs), 'IC sem zero (de)': f'{fora_zero}/{len(regs)}',
                   'registro mais influente': pior[0]})
influ = pd.DataFrame(linhas)
influ.to_csv(os.path.join(PASTA_CSV, 'influencia.csv'), index=False)
display(influ.round(2))""")

md(r"""## O que me mandar
1. Seção 1: a tabela de janelas discordantes e `fig_p_sobreposta.png`.
2. Seção 3: a tabela de g por configuração e `fig_sensibilidade.png`.
3. Seção 4: a tabela `dentro`.
4. Seção 5: as tabelas `alt` e `corr`, a linha **(c)** e `fig_alternativas.png`.
5. Seção 6: a tabela `influ`.""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}, "colab": {"provenance": []}}
nbf.write(nb, "/home/claude/tcc/15_sensibilidade_st_p.ipynb")
print("ok")
