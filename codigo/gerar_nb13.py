import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md(r"""# 13 — Recorrência de ST e de P batimento a batimento antes da FV (SDDB × NSRDB)

**A pergunta do TCC:** nas horas antes da fibrilação ventricular, o **segmento ST** muda de um jeito que a **onda P** não muda?

**Por que a P como controle interno:** ST e P são vistos pelos mesmos eletrodos. Mudanças **globais** — o paciente vira na cama, o
eletrodo descola, o tônus autonômico muda — alteram os dois juntos. Uma mudança **só no ST**, com a P estável, aponta para algo
**ventricular** (por exemplo, isquemia). A recorrência conjunta mede isso.

**Como cada batimento vira um ponto (sem depender das bordas exatas de P e T, que os notebooks 11 e 12 mostraram ser imprecisas):**
- vetor ST = o sinal de J até J + 120 ms, nas duas derivações;
- vetor P = o sinal de QRSon − 250 ms até QRSon − 30 ms;
- linha de base por **spline cúbica** pelos pontos isoelétricos do PR (técnica padrão para ST; a seção 4 compara com o filtro de
  mediana usado antes).

**Medidas por janela de 5 min** (mesmas janelas e portão de qualidade do braço de VFC):

| Medida | O que é | Por que |
|---|---|---|
| ST60_d1, ST60_d2 | nível em J + 60 ms, por derivação (mV, com sinal) | desvio clássico do ST |
| ST_dist | o mesmo, sem sinal (as duas derivações juntas) | desvio em qualquer direção |
| COER_ST, COER_P | mediana da correlação entre formas de batimentos | o quanto a forma se repete (1 = idêntica) |
| RR_ST, RR_P | taxa de recorrência (raio fixado no basal do próprio registro) | o quanto os batimentos voltam à forma basal |
| DET_ST, DET_P | determinismo (recorrências em sequência) | se a forma muda em blocos (deriva) ou ao acaso |
| JRR, RAZAO_JRR | recorrência conjunta ST × P e razão sobre a esperada sem relação | ST e P mudam juntos (global) ou separados |

**Validação antes da aplicação (seção 1):** a coerência da P deve separar ritmo sinusal de **fibrilação atrial** na QT Database (na FA
não há P organizada) — o teste que o critério de tamanho do notebook 12 não passou.

**Desenho:** efeito "última hora − basal" por registro; SDDB (13, FV real) × NSRDB (18, pseudo-eventos do mesmo sorteio do notebook
06); Hedges g com IC por bootstrap, Mann–Whitney e **correção de Holm para as 11 medidas** (fixadas antes de ver os resultados);
basal antes de −2 h e, como sensibilidade, antes de −6 h.

**Tempo:** uns 2 a 3 min por registro na primeira vez (cerca de 1 h a 1 h 30 para os 31); as tabelas por registro ficam salvas e uma
segunda rodada leva segundos.""")

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

EM_COLAB = os.path.isdir('/content')
if EM_COLAB and not os.path.isdir('/content/drive/MyDrive'):
    from google.colab import drive
    drive.mount('/content/drive')
BASE = os.environ.get('TCC_BASE') or ('/content/drive/MyDrive/Pesquisa/TCC' if EM_COLAB else os.path.abspath('TCC'))
PASTA_CODIGO = os.path.join(BASE, 'codigo')
PASTA_SDDB   = os.path.join(BASE, 'SDDB_atr')
PASTA_NSRDB  = os.path.join(BASE, 'NSRDB_atr')
PASTA_QTDB   = os.path.join(BASE, 'QTDB')
PASTA_DET    = os.path.join(BASE, 'tcc_qrs', 'deteccoes')        # batimentos do xqrs salvos no notebook 10
PASTA_JAN    = os.path.join(BASE, 'tcc_st_p', 'janelas')
PASTA_CSV    = os.path.join(BASE, 'tcc_st_p', 'csvs')
PASTA_FIG    = os.path.join(BASE, 'tcc_st_p', 'figs')
PASTA_SINAL  = '/content/sinais' if EM_COLAB else os.path.join(BASE, 'sinais')
SINAL = {'sddb': os.path.join(PASTA_SINAL, 'sddb'), 'nsrdb': os.path.join(PASTA_SINAL, 'nsrdb')}
PASTAS_LOCAIS = {'sddb': ['/content/drive/MyDrive/Pesquisa/SDDB_hrv'], 'nsrdb': []}
for p in (PASTA_SDDB, PASTA_NSRDB, PASTA_QTDB, PASTA_DET, PASTA_JAN, PASTA_CSV, PASTA_FIG, *SINAL.values()):
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

WIN_S, HORAS = 300, 24
TERMINAL, BASAL, BASAL_SENS = (-60, 0), -120, -360
MIN_BAT, GATE = 100, 0.80
SEED = 0
MEDIDAS = list(T.MEDIDAS_ST_P)
NOMES = {'ST60_d1': 'ST60, derivação 1 (mV)', 'ST60_d2': 'ST60, derivação 2 (mV)', 'ST_dist': 'Desvio do ST (mV)',
         'COER_ST': 'Coerência do ST', 'COER_P': 'Coerência da P', 'RR_ST': 'Recorrência do ST', 'DET_ST': 'Determinismo do ST',
         'RR_P': 'Recorrência da P', 'DET_P': 'Determinismo da P', 'JRR': 'Recorrência conjunta ST × P',
         'RAZAO_JRR': 'Razão conjunta / esperada'}""")

md(r"""## 1. Validação: a coerência da P separa ritmo sinusal de fibrilação atrial?
Nos 24 trechos da QT Database (15 min cada): coerência da P e do ST em todos os batimentos. Esperado: P coerente no ritmo sinusal,
incoerente na FA (sel35, sel36, sel37, sel50); o ST deve ser coerente nos dois (a FA não muda a repolarização ventricular).""")
code(r"""FA = {'sel35', 'sel36', 'sel37', 'sel50'}
QT_SDDB = [f'sel{n}' for n in range(30, 53)] + ['sel17152']
okq = T.baixar_wfdb('qtdb', QT_SDDB, PASTA_QTDB, exts=('hea', 'dat'))
val = []
for r in okq:
    rec = wfdb.rdrecord(os.path.join(PASTA_QTDB, r)); fs = float(rec.fs); sig = np.nan_to_num(rec.p_signal)
    rq = processing.xqrs_detect(sig[:, 0], fs=fs, verbose=False)
    de, Vst, Vp = T.extrair_batimentos(sig, fs, rq)
    val.append({'registro': r, 'ritmo': 'fibrilação atrial' if r in FA else 'sinusal', 'batimentos': len(de),
                'COER_P': T.coerencia(Vp), 'COER_ST': T.coerencia(Vst)})
val = pd.DataFrame(val).set_index('registro')
val.to_csv(os.path.join(PASTA_CSV, 'validacao_coerencia_qtdb.csv'))
display(val.round(3))
from scipy.stats import mannwhitneyu
for med in ['COER_P', 'COER_ST']:
    s, f = val[val.ritmo == 'sinusal'][med].dropna(), val[val.ritmo != 'sinusal'][med].dropna()
    u = mannwhitneyu(s, f, alternative='two-sided')
    print(f"{med}: sinusal {T.num(s.median(), 2)} × FA {T.num(f.median(), 2)} | AUC = {T.num(u.statistic / (len(s) * len(f)), 2)} | p = {T.num(u.pvalue, 3)}")
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for a, med in zip(axs, ['COER_P', 'COER_ST']):
    for k, (rit, cor) in enumerate([('sinusal', 'tab:blue'), ('fibrilação atrial', 'tab:red')]):
        v = val[val.ritmo == rit][med]
        a.scatter(np.full(len(v), k) + np.random.default_rng(k).uniform(-0.08, 0.08, len(v)), v, s=55, color=cor, zorder=3)
        a.hlines(v.median(), k - 0.25, k + 0.25, color='k', lw=3)
    a.set_xticks([0, 1]); a.set_xticklabels(['Ritmo sinusal', 'Fibrilação atrial']); a.set_xlim(-0.6, 1.6)
    a.set_ylabel('Coerência (correlação mediana entre batimentos)'); a.set_title(NOMES[med]); T.negrito_ticks(a)
plt.tight_layout(); plt.savefig(os.path.join(PASTA_FIG, 'fig_validacao_coerencia.png')); plt.show()""")

md("## 2. Registros, sinais e batimentos (do notebook 10)")
code(r"""meta = T.SDDB_META.copy()
base_eleg = (meta.rhythm == 'sinus') & (~meta.paced) & meta.vf_onset_s.notna()
REG_SDDB = meta[base_eleg].index.tolist()
ok_s = T.baixar_wfdb('sddb', REG_SDDB, PASTA_SDDB, exts=('hea',))
info = {r: T.info_header(os.path.join(PASTA_SDDB, r)) for r in ok_s}
NSRDB_LISTA = ['16265', '16272', '16273', '16420', '16483', '16539', '16773', '16786', '16795',
               '17052', '17453', '18177', '18184', '19088', '19090', '19093', '19140', '19830']
nsr_ok = T.baixar_wfdb('nsrdb', NSRDB_LISTA, PASTA_NSRDB)
info_nsr = {x: T.info_header(os.path.join(PASTA_NSRDB, x)) for x in nsr_ok}
pseudo = T.pseudo_eventos(nsr_ok, {x: h['dur_s'] for x, h in info_nsr.items()}, meta.vf_onset_s.dropna().tolist(), seed=SEED)
ALVOS = [('sddb', r, info[r]['vfon_s']) for r in ok_s] + [('nsrdb', r, pseudo[r]) for r in nsr_ok]
pendentes = [(b, r) for b, r, _ in ALVOS if not os.path.exists(os.path.join(PASTA_JAN, f'{b}_{r}.csv'))]
for base in ('sddb', 'nsrdb'):
    baixar = [r for b, r in pendentes if b == base and pasta_do_sinal(base, r) == SINAL[base]]
    if baixar:
        print(f'Baixando o sinal de {len(baixar)} registros da {base.upper()}…')
        T.baixar_wfdb(base, baixar, SINAL[base], exts=('hea', 'dat'))
print(f'SDDB: {len(ok_s)} registros | NSRDB: {len(nsr_ok)} | já processados: {len(ALVOS) - len(pendentes)}')

def batimentos_xqrs(base, r, pasta, t_evento):
    arq = os.path.join(PASTA_DET, f'{base}_{r}_xqrswfdb.npz')
    if os.path.exists(arq):
        z = np.load(arq)
        return np.asarray(z['marcas'], int), float(z['fs'])
    marcas, fs = [], None                          # sem o arquivo do notebook 10: detecta agora
    for s in np.arange(max(0.0, t_evento - HORAS * 3600 - 60), t_evento, 3600):
        e = min(s + 3600, t_evento)
        sig, fs, a0 = T.ler_sinal(base, r, pasta, s - 5, e + 5)
        q = processing.xqrs_detect(sig[:, 0], fs=fs, verbose=False) + a0
        marcas.extend(q[(q >= s * fs) & (q < e * fs)].tolist())
    return np.array(sorted(marcas), int), fs""")

md(r"""## 3. Processamento por registro
Em blocos de 1 h: delineação (v2), ST60 e vetores ST e P de cada batimento. Batimento **válido** = os dois RR em volta passam no
filtro de ectópicos de 20 %. Raio de recorrência do registro = quantil 10 % das distâncias entre batimentos **basais** (antes de −2 h).
Janela aprovada = pelo menos 100 batimentos válidos e 80 % de válidos. Cada registro vira um CSV em `tcc_st_p/janelas`.""")
code(r"""def processar(base, r, t_evento):
    arq = os.path.join(PASTA_JAN, f'{base}_{r}.csv')
    if os.path.exists(arq):
        return pd.read_csv(arq)
    pasta = pasta_do_sinal(base, r)
    marcas, fs = batimentos_xqrs(base, r, pasta, t_evento)
    tabs, VS, VP = [], [], []
    ini_s = max(0.0, t_evento - HORAS * 3600 - 60)
    for s in np.arange(ini_s, t_evento, 3600):
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
    if not tabs:
        return pd.DataFrame()
    de = pd.concat(tabs, ignore_index=True); Vs = np.vstack(VS); Vp = np.vstack(VP)
    # validade pelo RR (filtro de ectópicos de 20 %)
    t_rr, rr, v_rr = T.serie_rr_detectada(marcas, fs)
    pos = {int(m): k for k, m in enumerate(np.sort(marcas))}
    k = np.array([pos.get(int(x), -1) for x in de.R])
    valido = (k >= 1) & (k < len(v_rr))
    valido[valido] = v_rr[k[valido] - 1] & v_rr[np.minimum(k[valido], len(v_rr) - 1)]
    t_b = de.R.to_numpy() / fs
    basal = valido & (t_b <= t_evento + BASAL * 60)
    if basal.sum() < 30:
        basal = valido & (t_b <= t_b.min() + 1800)
    eps_st, eps_p = T.eps_do_basal(Vs[basal]), T.eps_do_basal(Vp[basal])
    linhas = []
    for s0 in np.arange(t_evento - WIN_S, max(ini_s, t_evento - HORAS * 3600) - 1e-9, -WIN_S)[::-1]:
        m = (t_b >= s0) & (t_b < s0 + WIN_S)
        n_tot, mv = int(m.sum()), m & valido
        reg = {'base': base.upper(), 'registro': r, 't_inicio_s': s0, 'min_ate_evento': (s0 + WIN_S - t_evento) / 60,
               'n_bat': n_tot, 'frac_valida': mv.sum() / n_tot if n_tot else np.nan, 'eps_st': eps_st, 'eps_p': eps_p}
        reg['passed'] = bool(n_tot and mv.sum() >= MIN_BAT and mv.sum() / n_tot >= GATE)
        if reg['passed']:
            d = de[mv]
            for c in [x for x in de.columns if x.startswith('ST60') or x == 'ST_dist']:
                reg[c] = float(np.nanmedian(d[c]))
            reg.update(T.medidas_janela(Vs[mv], Vp[mv], eps_st, eps_p))
        linhas.append(reg)
    out = pd.DataFrame(linhas)
    out.to_csv(arq, index=False)
    return out

partes, t0 = [], time.time()
for k, (base, r, tev) in enumerate(ALVOS, 1):
    t1 = time.time()
    try:
        partes.append(processar(base, r, tev))
        print(f'  [{k}/{len(ALVOS)}] {base} {r}: {int(partes[-1].passed.sum()) if len(partes[-1]) else 0} janelas aprovadas | '
              f'{time.time() - t1:.0f} s (total {(time.time() - t0) / 60:.1f} min)')
    except Exception as e:
        print(f'  [{k}/{len(ALVOS)}] {base} {r}: FALHOU ({type(e).__name__}: {e}) — rode de novo esta célula depois')
jan = pd.concat([p for p in partes if len(p)], ignore_index=True)
jan.to_csv(os.path.join(PASTA_CSV, 'janelas_st_p.csv'), index=False)
print('Janelas aprovadas:', jan.groupby('base').passed.mean().round(2).to_dict())""")

md(r"""## 4. ST60 com spline × com filtro de mediana
Se o filtro de mediana de 600 ms "engole" parte do ST, o ST60 dele fica menor que o da spline (inclinação < 1). Por janela aprovada da
SDDB.""")
code(r"""s = jan[(jan.base == 'SDDB') & jan.passed]
fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
for a, c in zip(axs, [1, 2]):
    x, y = s[f'ST60_d{c}'], s[f'ST60m_d{c}']
    ok_ = x.notna() & y.notna()
    incl = np.polyfit(x[ok_], y[ok_], 1)[0] if ok_.sum() > 10 else np.nan
    a.scatter(x, y, s=8, alpha=0.4, color='tab:blue')
    lim = np.nanpercentile(np.abs(np.r_[x, y]), 99)
    a.plot([-lim, lim], [-lim, lim], ':', color='0.5'); a.axhline(0, color='0.8'); a.axvline(0, color='0.8')
    a.set_xlabel('ST60 com spline (mV)'); a.set_ylabel('ST60 com filtro de mediana (mV)')
    a.set_title(f'Derivação {c}: inclinação {T.num(incl, 2)}'); T.negrito_ticks(a)
plt.tight_layout(); plt.savefig(os.path.join(PASTA_FIG, 'fig_st_spline_mediana.png')); plt.show()""")

md(r"""## 5. Efeito "última hora − basal": SDDB × NSRDB
Por registro: mediana das janelas da última hora menos a mediana das janelas basais (pelo menos 3 de cada lado). Holm sobre as 11
medidas, separadamente para cada basal.""")
code(r"""efeitos = []
guard = {}
for rot, lim in [('basal antes de -2 h', BASAL), ('basal antes de -6 h', BASAL_SENS)]:
    e = T.efeito_ultima_hora(jan, lim, TERMINAL, MEDIDAS)
    guard[rot] = e
    S, N = e[e.base == 'SDDB'], e[e.base == 'NSRDB']
    linhas = [{'basal': rot, 'medida': mm, **T.hedges_g_boot(S[f'{mm}_delta'], N[f'{mm}_delta'])} for mm in MEDIDAS]
    tab = pd.DataFrame(linhas)
    tab['p_holm'] = T.holm(tab.p_mw.to_numpy())
    efeitos.append(tab)
efe = pd.concat(efeitos, ignore_index=True)
efe.to_csv(os.path.join(PASTA_CSV, 'efeitos_st_p.csv'), index=False)
print('g = Hedges g (SDDB − NSRDB); media_a = SDDB, media_b = NSRDB; p_holm = Mann–Whitney corrigido para 11 medidas')
display(efe.round(3))
e2 = guard['basal antes de -2 h']
print('Registros SDDB com efeito calculável:', e2[e2.base == 'SDDB'].dropna(subset=['COER_ST_delta']).registro.tolist())""")

code(r"""e2 = guard['basal antes de -2 h']
fig, axs = plt.subplots(3, 4, figsize=(20, 13))
for a, mm in zip(axs.ravel(), MEDIDAS):
    for k, (bse, cor) in enumerate([('NSRDB', 'tab:blue'), ('SDDB', 'tab:red')]):
        v = e2[e2.base == bse][f'{mm}_delta'].dropna()
        a.scatter(np.full(len(v), k) + np.random.default_rng(k).uniform(-0.08, 0.08, len(v)), v, s=40, color=cor, zorder=3)
        if len(v): a.hlines(v.median(), k - 0.25, k + 0.25, color='k', lw=3)
    a.axhline(0, color='0.5', ls=':')
    rr_ = efe[(efe.basal == 'basal antes de -2 h') & (efe.medida == mm)].iloc[0]
    a.set_title(f"{NOMES[mm]}\ng = {T.num(rr_.g)} {T.ic(rr_.ic_lo, rr_.ic_hi)} · p Holm = {T.num(rr_.p_holm, 2)}", fontsize=11.5)
    a.set_xticks([0, 1]); a.set_xticklabels(['NSRDB\npseudo', 'SDDB\nFV']); a.set_xlim(-0.6, 1.6); T.negrito_ticks(a)
axs.ravel()[-1].axis('off')
for a in axs[:, 0]: a.set_ylabel('Última hora − basal')
plt.tight_layout(); plt.savefig(os.path.join(PASTA_FIG, 'fig_efeitos_st_p.png')); plt.show()""")

md(r"""## 6. Trajetórias nas 6 horas antes do evento
Mediana entre registros, por hora, de três medidas-chave. Serve para ver **quando** algo muda (se muda) e se a NSRDB também muda (o que
apontaria para efeito de relógio/grupo, como na FC do notebook 10).""")
code(r"""chave = ['ST_dist', 'COER_ST', 'RAZAO_JRR']
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
j6 = jan[jan.passed & (jan.min_ate_evento > -360)].assign(hora=lambda d: np.floor((d.min_ate_evento - 1e-6) / 60))
for a, mm in zip(axs, chave):
    for bse, cor in [('SDDB', 'tab:red'), ('NSRDB', 'tab:blue')]:
        g = j6[j6.base == bse].groupby(['registro', 'hora'])[mm].median().groupby('hora').median()
        a.plot(g.index + 0.5, g.values, '-o', color=cor, lw=2.2, label=bse)
    a.set_xlabel('Horas antes do evento'); a.set_title(NOMES[mm]); T.negrito_ticks(a)
axs[0].legend()
plt.tight_layout(); plt.savefig(os.path.join(PASTA_FIG, 'fig_trajetorias_st_p.png')); plt.show()""")

md(r"""## O que me mandar
1. Seção 1: a tabela `val`, as duas linhas de AUC e `fig_validacao_coerencia.png` — **valida (ou não) a recorrência de P**.
2. Seção 4: `fig_st_spline_mediana.png` (as duas inclinações).
3. Seção 5: a tabela `efe`, a lista de registros com efeito calculável e `fig_efeitos_st_p.png`.
4. Seção 6: `fig_trajetorias_st_p.png`.""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}, "colab": {"provenance": []}}
nbf.write(nb, "/home/claude/tcc/13_recorrencia_st_p.ipynb")
print("ok")
