import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md(r"""# 10 — Detecção de QRS pela matriz de distância (método da Nicole) aplicada à SDDB

**De onde vem:** TCC de Nicole Charron (UFPE) e o repositório `Ni-cole17/PTB-Diagnostic-Visualization`
(`segment.ipynb`, seção *Distância R-R*; `tests/generate_test_R.py`). Lá o método foi testado na PTB (1000 Hz, 12 derivações,
trechos de 2 s) e na LUDB (500 Hz, 4 s), com acurácia de ~99 % (limiarização) e ~98,5 % (K-means).

**A ideia (critério de amplitude):** a matriz de distância `D[i, j] = ‖x(i) − x(j)‖` é o gráfico de recorrência sem limiar
do próprio ECG — o mesmo objeto da análise de amplitude do artigo do CBEB. O QRS é a deflexão de maior amplitude, então as
amostras do QRS são as que ficam **longe de quase todas as outras**: viram faixas claras na matriz. Somando cada coluna da matriz
binarizada, as colunas do QRS têm as maiores somas; um limiar nessa soma e um agrupamento (DBSCAN) dão a posição de cada QRS.
A distância entre QRS consecutivos é o **intervalo RR**.

**Por que isso importa para o TCC:** no artigo, o braço de variabilidade usou só os 7 registros da SDDB que têm `.atr` auditado.
Há **6 registros elegíveis sem `.atr`** (33, 38, 39, 44, 47, 48: ritmo sinusal, sem marcapasso, com início da FV anotado).
Se o detector da Nicole recupera o RR com qualidade conhecida, o braço passa de **n = 7 para n = 13**.

**Desenho (nesta ordem, porque cada passo só vale se o anterior der certo):**
1. Figura do método num trecho real (registro 30).
2. **Validação contra o `.atr`** nos 7 registros anotados: sensibilidade, VPP, F1 e erro de tempo — separando a **última hora
   antes da FV** (onde há mais ectopia) do resto, porque um detector que piora perto do evento cria um viés que depende do desfecho.
   Comparação com o `xqrs` do WFDB (a referência que a própria Nicole usou na PTB).
3. Detecção completa (até 24 h antes do evento) nos 13 registros SDDB e nos 18 da NSRDB (pseudo-eventos, mesmo sorteio do notebook 06).
4. **Concordância da VFC** (FC, SDNN, RMSSD, AR(1)) entre RR do `.atr` e RR do detector, nos 7 registros anotados.
5. Só então: efeito "última hora − basal" com **n = 13** contra os pseudo-eventos — os dois lados com RR do mesmo detector.

**Adaptações ao método original (todas explícitas, marcadas `[ADAPTAÇÃO]` no `tcc_lib.py`):**
- janelas deslizantes de 4 s com passo de 3 s (o original analisa um trecho de 2 s; um Holter tem 24 h) e fusão de marcas a < 150 ms;
- `eps` do DBSCAN e `min_samples` convertidos para tempo (100 ms e 2 ms), para valerem a 250 Hz (SDDB) e 128 Hz (NSRDB);
- refino opcional do ponto fiducial: o centroide marca o meio da faixa do QRS; para RR com precisão de milissegundos, a marca vai
  para o máximo da norma das derivações filtradas a até ±60 ms (o mesmo critério de amplitude);
- a matriz, o percentil e o DBSCAN 1D foram reescritos para serem mais rápidos, com **resultado idêntico** (o DBSCAN 1D foi testado
  contra o sklearn em 2000 casos aleatórios, sem nenhuma diferença).

**Tempo de execução (Colab):** ~5 min para baixar os sinais (cerca de 1,4 GB, no disco do Colab, não no Drive), ~5 min por registro SDDB de 24 h e ~1–2 min por registro NSRDB. As detecções ficam salvas
(`tcc_qrs/deteccoes`), então rodar de novo é rápido.

**Saídas:** `qrs_validacao.csv`, `qrs_validacao_resumo.csv`, `qrs_concordancia_janelas.csv`, `qrs_concordancia_registros.csv`,
`qrs_janelas.csv`, `qrs_efeitos.csv` e as figuras `fig_qrs_metodo.png`, `fig_qrs_validacao.png`, `fig_qrs_concordancia.png`,
`fig_qrs_efeito.png`.""")

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
PASTA_SDDB   = os.path.join(BASE, 'SDDB_atr')     # .hea e .atr (o sinal é lido do PhysioNet se o .dat não estiver aqui)
PASTA_NSRDB  = os.path.join(BASE, 'NSRDB_atr')
# Sinal (.dat): baixado UMA vez, inteiro, para o disco do Colab (não ocupa o Drive; some quando a sessão fecha).
# Ler trechos direto do PhysioNet faz milhares de pedidos e o servidor responde 502 de vez em quando.
PASTA_SINAL  = '/content/sinais' if EM_COLAB else os.path.join(BASE, 'sinais')
SINAL_SDDB, SINAL_NSRDB = os.path.join(PASTA_SINAL, 'sddb'), os.path.join(PASTA_SINAL, 'nsrdb')
# Pastas do Drive onde você JÁ TEM os .hea + .dat: o que estiver nelas não é baixado de novo.
PASTAS_LOCAIS = {'sddb':  ['/content/drive/MyDrive/Pesquisa/SDDB_hrv'],
                 'nsrdb': []}          # se tiver os .dat da NSRDB no Drive, ponha a pasta aqui

def pasta_do_sinal(base, r):
    for p in PASTAS_LOCAIS.get(base, []):
        if os.path.exists(os.path.join(p, f'{r}.dat')) and os.path.exists(os.path.join(p, f'{r}.hea')):
            return p
    return SINAL_SDDB if base == 'sddb' else SINAL_NSRDB
PASTA_DET    = os.path.join(BASE, 'tcc_qrs', 'deteccoes')
PASTA_CSV    = os.path.join(BASE, 'tcc_qrs', 'csvs')
PASTA_FIG    = os.path.join(BASE, 'tcc_qrs', 'figs')
for p in (PASTA_SDDB, PASTA_NSRDB, PASTA_DET, PASTA_CSV, PASTA_FIG, SINAL_SDDB, SINAL_NSRDB):
    os.makedirs(p, exist_ok=True)
sys.path.insert(0, PASTA_CODIGO)
import tcc_lib as T
importlib.reload(T)
T.estilo_figuras()

# Método da Nicole (limiarização) — valores do repositório
PARAM_NICOLE = dict(q=0.95, frac=0.75, eps_s=0.100, min_s=0.002)
JANELA_S, PASSO_S = 4.0, 3.0
# Variantes comparadas na validação: (rótulo, parâmetros, usar refino do pico?)
# (rótulo, método, parâmetros, refinar o pico?)
#   'original' = exatamente o método da Nicole (limiares relativos à própria janela de 4 s)
#   'movel'    = correção proposta: mesmos passos, com o percentil e o máximo tomados como MEDIANA das janelas vizinhas
VARIANTES = [('Nicole (centroide)',        'original', PARAM_NICOLE,                   False),
             ('Nicole + refino do pico',   'original', PARAM_NICOLE,                   True),
             ('Limiar 50 % + refino',      'original', {**PARAM_NICOLE, 'frac': 0.50}, True),
             ('Referência móvel 75 %',     'movel',    {'q': 0.95, 'frac': 0.75},      False),
             ('Referência móvel 50 %',     'movel',    {'q': 0.95, 'frac': 0.50},      False)]
VIZ = 15                     # referência móvel: ±15 janelas de 3 s ≈ ±45 s (fixado antes de ver os resultados)
VARIANTE_ANALISE = 'auto'    # 'auto' = regra da seção 3 (definida antes dos resultados); ou o rótulo de uma variante
COMPARAR_XQRS = True                           # xqrs do WFDB nos trechos de validação
TOL_AAMI, TOL_NICOLE = 0.150, 0.060            # tolerâncias de casamento: ±150 ms (AAMI EC57) e ±60 ms (janela de 120 ms da Nicole)

# Mesmos parâmetros do notebook 06
WIN_S, STEP_S, GATE = 300, 300, 0.80
HORAS = 24
TERMINAL = (-60, 0)
BASAL, BASAL_SENS = -120, -360
INDICES = ['FC', 'SDNN', 'RMSSD', 'AR1']
SEED = 0
NOMES = {'FC': 'FC (bpm)', 'SDNN': 'SDNN (ms)', 'RMSSD': 'RMSSD (ms)', 'AR1': 'AR(1) do NN sem tendência'}""")

md(r"""## 1. Registros
**Anotados (validação + análise):** os 7 do artigo. **Novos (só análise):** sinusais, sem marcapasso, com `vfon`, **sem** `.atr`.
`VALIDAR_EXTRAS = True` acrescenta à validação os outros 5 registros com `.atr` (32, 49, 51 com marcapasso; 35, 36 em fibrilação
atrial) — servem para ver onde o método quebra, não entram na análise de VFC.""")
code(r"""VALIDAR_EXTRAS = False
meta = T.SDDB_META.copy()
base_eleg = (meta.rhythm == 'sinus') & (~meta.paced) & meta.vf_onset_s.notna()
REG_ATR = meta[base_eleg & meta.has_atr].index.tolist()
REG_NOVOS = meta[base_eleg & ~meta.has_atr].index.tolist()
REG_EXTRAS = meta[meta.has_atr & ~meta.index.isin(REG_ATR) & meta.vf_onset_s.notna()].index.tolist()
assert tuple(REG_NOVOS) == T.SDDB_SEM_ATR_ELEGIVEIS, REG_NOVOS
print('Anotados (artigo):', REG_ATR)
print('Novos, sem .atr:  ', REG_NOVOS)
print('Extras p/ validar:', REG_EXTRAS if VALIDAR_EXTRAS else '(desligado)')

ok_atr = T.baixar_wfdb('sddb', REG_ATR + (REG_EXTRAS if VALIDAR_EXTRAS else []), PASTA_SDDB, exts=('hea', 'atr'))
ok_nov = T.baixar_wfdb('sddb', REG_NOVOS, PASTA_SDDB, exts=('hea',))
faltando = [r for r in REG_ATR if r not in ok_atr] + [r for r in REG_NOVOS if r not in ok_nov]
if faltando:   # o PhysioNet limita pedidos seguidos: espere 1 min e rode esta célula de novo (o que já baixou é pulado)
    raise RuntimeError(f'Não consegui baixar o .hea de {faltando}. Espere 1 minuto e rode esta célula de novo.')

# Sinais completos (~65 MB por registro de 24 h; ~5 min no total). Só baixa o que ainda não está no disco.
precisa = REG_ATR + REG_NOVOS + (REG_EXTRAS if VALIDAR_EXTRAS else [])
no_drive = [r for r in precisa if pasta_do_sinal('sddb', r) != SINAL_SDDB]
baixar = [r for r in precisa if r not in no_drive]
print('Sinal já no seu Drive:', no_drive)
print('Sinal a baixar do PhysioNet:', baixar or 'nenhum')
ok_sinal = T.baixar_wfdb('sddb', baixar, SINAL_SDDB, exts=('hea', 'dat'))
if set(baixar) - set(ok_sinal):
    raise RuntimeError(f'Faltou o sinal de {sorted(set(baixar) - set(ok_sinal))}. Espere 1 minuto e rode esta célula de novo.')
info = {r: T.info_header(os.path.join(PASTA_SDDB, r)) for r in ok_atr + ok_nov}
display(pd.DataFrame({r: {'fs (Hz)': h['fs'], 'duração (h)': round(h['dur_s'] / 3600, 1), 'vfon (h)': round(h['vfon_s'] / 3600, 1),
                          'tem .atr': r in ok_atr, 'na análise': r in REG_ATR + REG_NOVOS} for r, h in info.items()}).T)""")

md(r"""## 2. O método passo a passo num trecho real (registro 30, 1 h antes da FV)
Figura para o capítulo de método (equivalente à Figura 22 da Nicole): sinal com as marcas do `.atr` e do método, a matriz de
distância, a matriz binarizada (percentil 95 %), a soma das colunas com o limiar (75 % do máximo) e os grupos do DBSCAN.""")
code(r"""r, t0 = '30', info['30']['vfon_s'] - 3600
sig, fs, a0 = T.ler_sinal('sddb', r, pasta_do_sinal('sddb', r), t0 - 2, t0 + JANELA_S + 2)
filt = T.filtro_passa_banda(sig, fs)
i0 = int(2 * fs); W = int(JANELA_S * fs)
trecho = filt[i0:i0 + W]
cent, inf = T.qrs_janela_matriz(trecho, fs, devolver_tudo=True, **PARAM_NICOLE)
pico = T.refinar_pico(trecho, cent.round().astype(int), fs)
ann = wfdb.rdann(os.path.join(PASTA_SDDB, r), 'atr', sampfrom=a0 + i0, sampto=a0 + i0 + W)
ref = np.array([s - (a0 + i0) for s, sy in zip(ann.sample, ann.symbol) if sy in T.SIMB_BATIMENTO])
tt = np.arange(W) / fs

fig = plt.figure(figsize=(16, 11))
g = fig.add_gridspec(2, 3, height_ratios=[1, 1.25])
a = fig.add_subplot(g[0, :])
for c, cor in zip(range(trecho.shape[1]), ['tab:blue', 'tab:gray']):
    a.plot(tt, trecho[:, c], color=cor, lw=1.2, label=f'Derivação {c + 1} (filtrada 1–50 Hz)')
for k, x in enumerate(ref):
    a.axvline(x / fs, color='tab:green', ls=':', lw=2.5, label='Batimento no .atr' if k == 0 else None)
for k, x in enumerate(pico):
    a.axvline(x / fs, color='tab:red', ls='--', lw=1.5, label='QRS detectado (refinado)' if k == 0 else None)
a.set_xlabel('Tempo (s)'); a.set_ylabel('Amplitude (mV)'); a.set_title('(a) Sinal e marcações')
a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=11); T.negrito_ticks(a)
a = fig.add_subplot(g[1, 0]); im = a.imshow(inf['D'], cmap='jet', extent=[0, JANELA_S, JANELA_S, 0])
a.set_title('(b) Matriz de distância'); a.set_xlabel('Tempo (s)'); a.set_ylabel('Tempo (s)')
plt.colorbar(im, ax=a, fraction=0.046, label='Distância (mV)')
a = fig.add_subplot(g[1, 1]); a.imshow(inf['binaria'], cmap='gray', extent=[0, JANELA_S, JANELA_S, 0])
a.set_title(f"(c) Binarizada (D ≥ percentil 95 % = {T.num(inf['limiar_d'], 2)} mV)"); a.set_xlabel('Tempo (s)')
a = fig.add_subplot(g[1, 2]); a.plot(tt, inf['soma'], color='k', lw=1.2)
a.axhline(inf['limiar_soma'], color='tab:red', ls='--', lw=2, label='Limiar (75 % do máximo)')
for k, gr in enumerate(inf['grupos']):
    a.plot(gr / fs, inf['soma'][gr.astype(int)], 'o', ms=4, color=plt.cm.tab10(k % 10))
a.set_title('(d) Soma das colunas e grupos (DBSCAN)'); a.set_xlabel('Tempo (s)'); a.set_ylabel('Soma da coluna (pixels)')
a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(PASTA_FIG, 'fig_qrs_metodo.png')); plt.show()
print(f'{len(ref)} batimentos no .atr | {len(pico)} QRS detectados neste trecho de {T.num(JANELA_S, 0)} s')""")

md(r"""## 3. Validação contra o `.atr`
Por registro: a **última hora antes da FV** e **3 trechos de 20 min sorteados** nas horas anteriores (até 24 h antes).
É uma amostra — rodar as 24 h inteiras com 4 variantes levaria horas; a detecção completa (seção 4) é feita uma vez só, com a variante
escolhida, e a concordância da seção 5 usa as 24 h inteiras.

**Como ler:** casamento um-a-um a ±150 ms (padrão AAMI EC57 para detectores de QRS) e a ±60 ms (a janela de 120 ms da Nicole).
O `.atr` da SDDB marca o batimento num ponto do QRS que não é necessariamente o pico R; por isso o "erro de tempo" tem um
deslocamento constante (viés) que **não afeta o RR**. O que afeta o RR é a **dispersão** do erro (coluna `erro_abs_p95_ms`
mede o pior caso típico).""")
code(r"""def trechos_validacao(vfon, dur, rng, n_extra=3, dur_extra=1200):
    out = [('última hora', max(0, vfon - 3600), vfon)]
    ini = max(0, vfon - HORAS * 3600)
    fim = vfon - 3600 - dur_extra
    if fim > ini:
        for k, s in enumerate(np.sort(rng.uniform(ini, fim, n_extra))):
            out.append((f'antes ({k + 1})', s, s + dur_extra))
    return out

def detectar_trecho(sig, fs, ini, fim, metodo, param, refinar):
    if metodo == 'xqrs':
        xq = processing.xqrs_detect(sig[:, 0], fs=fs, verbose=False)
        return xq[(xq >= ini) & (xq < fim)]
    if metodo == 'movel':
        d = T.detectar_qrs_matriz_movel(sig, fs, ini=ini, fim=fim, win_s=JANELA_S, passo_s=PASSO_S, viz=VIZ,
                                        refinar=refinar, **param)
    else:
        d = T.detectar_qrs_matriz(sig, fs, ini=ini, fim=fim, win_s=JANELA_S, passo_s=PASSO_S, refinar=refinar, **param)
    return d['pico'] if refinar else d['centroide']

rng = np.random.default_rng(SEED)
linhas, erros = [], []
reg_val = REG_ATR + (REG_EXTRAS if VALIDAR_EXTRAS else [])
for r in reg_val:
    h = info[r]
    for nome_tr, s, e in trechos_validacao(h['vfon_s'], h['dur_s'], rng):
        sig, fs, a0 = T.ler_sinal('sddb', r, pasta_do_sinal('sddb', r), s - 5, e + 5)
        ini, fim = int((s - a0 / fs) * fs), int((e - a0 / fs) * fs)
        ann = wfdb.rdann(os.path.join(PASTA_SDDB, r), 'atr', sampfrom=a0 + ini, sampto=a0 + fim)
        ref = np.array([x - a0 for x, sy in zip(ann.sample, ann.symbol) if sy in T.SIMB_BATIMENTO])
        n_ect = sum(sy in T.SIMB_BATIMENTO and sy not in T.SIMB_SINUSAL for sy in ann.symbol)
        dets = {nome: detectar_trecho(sig, fs, ini, fim, met, p, rf) for nome, met, p, rf in VARIANTES}
        if COMPARAR_XQRS:
            dets['xqrs (WFDB)'] = detectar_trecho(sig, fs, ini, fim, 'xqrs', None, False)
        for nome, det in dets.items():
            for tol_nome, tol in [('±150 ms', TOL_AAMI), ('±60 ms', TOL_NICOLE)]:
                m = T.comparar_com_referencia(det, ref, fs, tol)
                if tol == TOL_AAMI:
                    erros.append(pd.DataFrame({'registro': r, 'trecho': nome_tr, 'variante': nome, 'erro_ms': m['erros_ms']}))
                linhas.append({'registro': r, 'trecho': nome_tr, 'ultima_hora': nome_tr == 'última hora', 'variante': nome,
                               'tolerancia': tol_nome, 'n_ref': len(ref), 'ectopicos_ref': n_ect,
                               **{k: v for k, v in m.items() if k != 'erros_ms'}})
    print(f'  {r}: ok')
val = pd.DataFrame(linhas)
err = pd.concat(erros, ignore_index=True)
val.to_csv(os.path.join(PASTA_CSV, 'qrs_validacao.csv'), index=False)

def resumo(df):
    s = df[['VP', 'FP', 'FN']].sum()
    se, vpp = s.VP / (s.VP + s.FN), s.VP / (s.VP + s.FP)
    return pd.Series({'VP': s.VP, 'FP': s.FP, 'FN': s.FN, 'Se (%)': 100 * se, 'VPP (%)': 100 * vpp,
                      'F1 (%)': 200 * se * vpp / (se + vpp)})
res_val = val.groupby(['tolerancia', 'variante', 'ultima_hora']).apply(resumo, include_groups=False).round(2)
res_val.to_csv(os.path.join(PASTA_CSV, 'qrs_validacao_resumo.csv'))
display(res_val)
disp = err.groupby('variante').erro_ms.agg(vies_ms='median',
                                           dispersao_iqr_ms=lambda x: np.subtract(*np.percentile(x, [75, 25])),
                                           p95_abs_ms=lambda x: np.percentile(np.abs(x - np.median(x)), 95)).round(1)
print('Erro de tempo (detecção − .atr), casamento a ±150 ms. O viés não afeta o RR; a dispersão sim:')
display(disp)""")

code(r"""fig, axs = plt.subplots(1, 2, figsize=(16, 6))
a = axs[0]
v = val[val.tolerancia == '±150 ms']
nomes_var = list(v.variante.unique())
for k, nome in enumerate(nomes_var):
    for uh, cor, mk in [(False, 'tab:blue', 'o'), (True, 'tab:red', 's')]:
        d = v[(v.variante == nome) & (v.ultima_hora == uh)]
        x = np.full(len(d), k) + (0.15 if uh else -0.15) + np.random.default_rng(k).uniform(-0.05, 0.05, len(d))
        a.scatter(x, 100 * d.F1, color=cor, marker=mk, s=45, alpha=0.8,
                  label=('Última hora antes da FV' if uh else 'Horas anteriores') if k == 0 else None)
import textwrap
a.set_xticks(range(len(nomes_var))); a.set_xticklabels([textwrap.fill(n, 12) for n in nomes_var], fontsize=10.5)
a.set_ylabel('F1 por trecho (%)'); a.set_title('Validação contra o .atr (±150 ms)')
a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2); T.negrito_ticks(a)
a = axs[1]
for nome, cor in zip(nomes_var, plt.cm.tab10.colors):
    e = err[err.variante == nome].erro_ms
    a.hist(e - np.median(e), bins=np.arange(-80, 81, 4), histtype='step', lw=2, color=cor, label=nome)
a.set_xlabel('Erro de tempo − viés (ms)'); a.set_ylabel('Batimentos'); a.set_yscale('log')
a.set_title('Dispersão do instante detectado')
a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=11); T.negrito_ticks(a)
plt.tight_layout(); plt.savefig(os.path.join(PASTA_FIG, 'fig_qrs_validacao.png')); plt.show()""")

md(r"""### 3b. Por que os batimentos se perdem (diagnóstico do método original)
Para cada batimento do `.atr` que a variante original não achou na última hora antes da FV: quem definiu o máximo da janela
(batimento normal, extrassístole ou artefato), a altura do batimento perdido em relação a esse máximo (o limiar é 0,75) e o tipo do
batimento perdido. Foi este diagnóstico que motivou a variante com **referência móvel**.""")
code(r"""import collections
v = val[val.tolerancia == '±150 ms']
print('Sensibilidade (%) por registro:')
display((100 * v.pivot_table(index='registro', columns=['variante', 'ultima_hora'], values='Se')).round(1))
exemplos = []
W, P = int(JANELA_S * 250), int(PASSO_S * 250)
for r in REG_ATR:
    h = info[r]; s, e = h['vfon_s'] - 3600, h['vfon_s']
    sig, fs, a0 = T.ler_sinal('sddb', r, pasta_do_sinal('sddb', r), s - 5, e + 5)
    ini, fim = int((s - a0 / fs) * fs), int((e - a0 / fs) * fs)
    ann = wfdb.rdann(os.path.join(PASTA_SDDB, r), 'atr', sampfrom=a0 + ini, sampto=a0 + fim)
    simb = {x - a0: sy for x, sy in zip(ann.sample, ann.symbol)}
    ref = np.array([x for x, sy in simb.items() if sy in T.SIMB_BATIMENTO])
    det = T.detectar_qrs_matriz(sig, fs, ini=ini, fim=fim, **PARAM_NICOLE)['pico']
    filt = T.filtro_passa_banda(sig, fs)
    borda = (W - P) // 2
    for x in ref:
        if len(det) and np.min(np.abs(det - x)) <= 0.15 * fs:
            continue
        s0 = ini + max(0, (x - ini - borda) // P) * P
        _, inf = T.qrs_janela_matriz(filt[s0:s0 + W], fs, devolver_tudo=True, **PARAM_NICOLE)
        if not inf:
            exemplos.append({'registro': r, 'perdido': simb.get(x, '?'), 'dono_do_maximo': 'janela plana', 'razao': np.nan}); continue
        j = x - s0
        razao = inf['soma'][max(0, j - 8):j + 9].max() / inf['soma'].max()
        jmax = s0 + int(np.argmax(inf['soma']))
        perto = [sy for xx, sy in simb.items() if abs(xx - jmax) < 0.1 * fs and sy in T.SIMB_BATIMENTO]
        dono = ('normal (N)' if perto[0] == 'N' else f'ectópico ({perto[0]})') if perto else 'sem batimento: artefato/ruído'
        exemplos.append({'registro': r, 'perdido': simb.get(x, '?'), 'dono_do_maximo': dono, 'razao': razao})
ex = pd.DataFrame(exemplos)
ex.to_csv(os.path.join(PASTA_CSV, 'qrs_diagnostico_perdidos.csv'), index=False)
print(f'{len(ex)} batimentos perdidos na última hora (7 registros)')
display(ex.dono_do_maximo.value_counts())
display(ex.groupby('dono_do_maximo').razao.describe().round(2))
display(ex.perdido.value_counts())""")

md(r"""### 3c. Erro no RR (o que importa para a VFC)
O erro de tempo contra o `.atr` mistura defasagens que são constantes dentro de cada registro (cada paciente tem seu eixo elétrico,
e a marca do `.atr` foi posta numa derivação) e que se cancelam no RR. Aqui: o erro depois de tirar o viés **de cada registro** e o
**erro no próprio RR** (diferença entre os erros de dois batimentos seguidos).""")
code(r"""linhas = []
for var, d in err.groupby('variante'):
    e_rel = d.erro_ms - d.groupby('registro').erro_ms.transform('median')
    e_rr = d.groupby(['registro', 'trecho']).erro_ms.transform(lambda x: x.diff()).dropna()
    linhas.append({'variante': var, 'p95 |erro|, viés por registro (ms)': np.percentile(np.abs(e_rel), 95),
                   'mediana |erro no RR| (ms)': np.median(np.abs(e_rr)), 'p95 |erro no RR| (ms)': np.percentile(np.abs(e_rr), 95),
                   '% RR com erro > 20 ms': 100 * np.mean(np.abs(e_rr) > 20)})
erro_rr = pd.DataFrame(linhas).round(1)
erro_rr.to_csv(os.path.join(PASTA_CSV, 'qrs_erro_rr.csv'), index=False)
display(erro_rr)""")

md(r"""### 3d. Escolha da variante (regra fixada antes de ver os resultados)
Critério de detecção: **F1 ≥ 99 %** a ±150 ms nas horas anteriores **e** na última hora, com **queda ≤ 1 ponto percentual** na última
hora. Entre as variantes da matriz que passam, fica a de maior F1 na última hora. Se nenhuma variante da matriz passar, a detecção
completa usa o `xqrs` e o texto registra que o método não se transferiu ao Holter.

**Mudança declarada no critério de tempo:** o critério original (dispersão do erro ≤ 20 ms contra o `.atr`) não foi atingido por
nenhum detector, nem pelo `xqrs` (p95 ≈ 40 ms), o que indica variação de posição da própria marca do `.atr`. A precisão de tempo
passou a ser julgada pelo erro no RR (3c) e pela concordância da VFC (seção 5). Esta mudança foi feita depois de ver os resultados e
deve constar no texto.""")
code(r"""f1 = res_val.xs('±150 ms', level='tolerancia')['F1 (%)'].unstack('ultima_hora')
f1.columns = ['F1 horas anteriores', 'F1 última hora']
f1['queda (p.p.)'] = f1['F1 horas anteriores'] - f1['F1 última hora']
f1['passa'] = (f1['F1 horas anteriores'] >= 99) & (f1['F1 última hora'] >= 99) & (f1['queda (p.p.)'] <= 1)
display(f1.round(2))
matriz = f1.drop(index='xqrs (WFDB)', errors='ignore')
if VARIANTE_ANALISE != 'auto':
    ESCOLHIDA = VARIANTE_ANALISE
elif matriz.passa.any():
    ESCOLHIDA = matriz[matriz.passa]['F1 última hora'].idxmax()
else:
    ESCOLHIDA = 'xqrs (WFDB)'
print('Variante escolhida para as seções 4 a 7:', ESCOLHIDA)""")

md(r"""## 4. Detecção completa (até 24 h antes do evento) — SDDB (13) e NSRDB (18)
Lê o sinal em blocos de 1 h (do `.dat` baixado para o disco do Colab), detecta com a variante escolhida e salva as marcas em
`tcc_qrs/deteccoes/<base>_<registro>.npz`. Se o arquivo já existe, pula. Os pseudo-eventos NSRDB são os mesmos do notebook 06
(mesmo sorteio, `seed = 0`).""")
code(r"""NSRDB_LISTA = ['16265', '16272', '16273', '16420', '16483', '16539', '16773', '16786', '16795',
               '17052', '17453', '18177', '18184', '19088', '19090', '19093', '19140', '19830']
try:
    lista = [str(x) for x in wfdb.get_record_list('nsrdb')]
except Exception:
    lista = NSRDB_LISTA
nsr_ok = T.baixar_wfdb('nsrdb', lista, PASTA_NSRDB)
nsr_sinal = T.baixar_wfdb('nsrdb', [x for x in nsr_ok if pasta_do_sinal('nsrdb', x) == SINAL_NSRDB], SINAL_NSRDB, exts=('hea', 'dat'))
info_nsr = {x: T.info_header(os.path.join(PASTA_NSRDB, x)) for x in nsr_ok}
pseudo = T.pseudo_eventos(nsr_ok, {x: h['dur_s'] for x, h in info_nsr.items()}, T.SDDB_META.vf_onset_s.dropna().tolist(), seed=SEED)

TODAS = VARIANTES + [('xqrs (WFDB)', 'xqrs', None, False)]
_, MET_AN, PARAM_AN, REF_AN = next(v for v in TODAS if v[0] == ESCOLHIDA)
SUFIXO = '_' + ''.join(ch for ch in ESCOLHIDA.lower() if ch.isalnum())     # uma pasta de cache por variante
print('Detecção completa com:', ESCOLHIDA)

def detectar_registro(base, r, pasta, t_evento, bloco_s=3600):
    arq = os.path.join(PASTA_DET, f'{base}_{r}{SUFIXO}.npz')
    if os.path.exists(arq):
        z = np.load(arq)
        return z['marcas'], float(z['fs'])
    ini_s = max(0.0, t_evento - HORAS * 3600 - 60)
    marcas, fs = [], None
    for s in np.arange(ini_s, t_evento, bloco_s):
        e = min(s + bloco_s, t_evento)
        sig, fs, a0 = T.ler_sinal(base, r, pasta, s - 5, e + 5)
        ini, fim = int(round(s * fs)) - a0, int(round(e * fs)) - a0
        m = detectar_trecho(sig, fs, ini, fim, MET_AN, PARAM_AN, REF_AN)
        marcas.extend((np.asarray(m) + a0).tolist())
    marcas = np.array(sorted(marcas))
    if len(marcas):                                     # funde duplicatas na emenda dos blocos
        marcas = marcas[np.r_[True, np.diff(marcas) > 0.150 * fs]]
    np.savez(arq, marcas=marcas, fs=fs, t_evento=t_evento)
    return marcas, fs

DET = {}
alvos = [('sddb', r, pasta_do_sinal('sddb', r), info[r]['vfon_s']) for r in REG_ATR + REG_NOVOS] + \
        [('nsrdb', r, pasta_do_sinal('nsrdb', r), pseudo[r]) for r in pseudo]
t_ini = time.time()
for k, (base, r, pasta, tev) in enumerate(alvos, 1):
    t1 = time.time()
    DET[(base, r)] = detectar_registro(base, r, pasta, tev)
    print(f'  [{k}/{len(alvos)}] {base} {r}: {len(DET[(base, r)][0])} QRS | {time.time() - t1:.0f} s '
          f'(total {(time.time() - t_ini) / 60:.1f} min)')""")

md(r"""## 5. Concordância da VFC: RR do `.atr` × RR do detector (7 registros anotados)
As mesmas janelas de 5 min do notebook 06. Diferença inevitável na **máscara de validade**: com o `.atr`, um intervalo é NN se os dois
batimentos são sinusais (rótulo do anotador); com o detector não há rótulo, então vale o filtro de ectópicos de 20 % usado no resto do
TCC. Por isso a comparação é do **pipeline inteiro** (detecção + validade), que é o que a análise com n = 13 vai usar.""")
code(r"""def janelas(base, r, t, rr, v, tev, fonte):
    d = T.indices_janelas(t, rr, v, tev, HORAS, WIN_S, STEP_S, GATE)
    d.insert(0, 'fonte', fonte); d.insert(0, 'registro', r); d.insert(0, 'base', base)
    return d

partes = []
for r in REG_ATR:
    tev = info[r]['vfon_s']
    t, rr, v = T.serie_rr_anotada(os.path.join(PASTA_SDDB, r), info[r]['fs'])
    partes.append(janelas('SDDB', r, t, rr, v, tev, 'anotado'))
for (base, r), (m, fs) in DET.items():
    tev = info[r]['vfon_s'] if base == 'sddb' else pseudo[r]
    t, rr, v = T.serie_rr_detectada(m, fs)
    partes.append(janelas(base.upper(), r, t, rr, v, tev, 'detector'))
for r in pseudo:                                       # NSRDB anotado, para reproduzir o notebook 06
    t, rr, v = T.serie_rr_anotada(os.path.join(PASTA_NSRDB, r), info_nsr[r]['fs'])
    partes.append(janelas('NSRDB', r, t, rr, v, pseudo[r], 'anotado'))
jan = pd.concat(partes, ignore_index=True)
jan.to_csv(os.path.join(PASTA_CSV, 'qrs_janelas.csv'), index=False)

par = jan[(jan.base == 'SDDB') & jan.registro.isin(REG_ATR)].pivot_table(
    index=['registro', 't_inicio_s'], columns='fonte', values=['passed'] + INDICES, aggfunc='first')
aprov = par['passed'].astype(float)
print(f"Janelas aprovadas no portão — anotado: {aprov['anotado'].mean():.0%} | detector: {aprov['detector'].mean():.0%} | "
      f"ambos: {(aprov.all(axis=1)).mean():.0%}")
amb = par[par['passed'].fillna(False).astype(bool).all(axis=1)]
conc = []
for ind in INDICES:
    x, y = amb[(ind, 'anotado')].astype(float), amb[(ind, 'detector')].astype(float)
    d = y - x
    conc.append({'indice': ind, 'n_janelas': len(d), 'vies': d.mean(), 'lim_inf': d.mean() - 1.96 * d.std(ddof=1),
                 'lim_sup': d.mean() + 1.96 * d.std(ddof=1), 'r_spearman': x.corr(y, method='spearman'),
                 'mediana_anotado': x.median()})
conc = pd.DataFrame(conc)
conc.to_csv(os.path.join(PASTA_CSV, 'qrs_concordancia_janelas.csv'), index=False)
print('Bland–Altman por janela (detector − anotado):'); display(conc.round(3))

ef_conc = T.efeito_ultima_hora(jan[(jan.base == 'SDDB') & jan.registro.isin(REG_ATR)], BASAL, TERMINAL, INDICES)
ef_conc = ef_conc.pivot(index='registro', columns='fonte')
ef_conc.to_csv(os.path.join(PASTA_CSV, 'qrs_concordancia_registros.csv'))
print('Efeito "última hora − basal" por registro, anotado × detector:')
display(ef_conc[[f'{i}_delta' for i in INDICES]].round(2))""")

code(r"""fig, axs = plt.subplots(2, 4, figsize=(19, 9.5))
for k, ind in enumerate(INDICES):
    x, y = amb[(ind, 'anotado')].astype(float), amb[(ind, 'detector')].astype(float)
    c = conc.iloc[k]
    a = axs[0, k]
    a.scatter((x + y) / 2, y - x, s=8, alpha=0.4, color='tab:blue')
    for val_, ls in [(c.vies, '-'), (c.lim_inf, '--'), (c.lim_sup, '--')]:
        a.axhline(val_, color='tab:red', ls=ls, lw=1.8)
    a.set_title(f"{NOMES[ind]}\nviés {T.num(c.vies)} [{T.num(c.lim_inf)}; {T.num(c.lim_sup)}]", fontsize=12)
    a.set_xlabel('Média dos dois'); T.negrito_ticks(a)
    a = axs[1, k]
    xa = ef_conc[(f'{ind}_delta', 'anotado')]; ya = ef_conc[(f'{ind}_delta', 'detector')]
    a.scatter(xa, ya, s=60, color='tab:red', zorder=3)
    for r_, xx, yy in zip(ef_conc.index, xa, ya):
        a.annotate(r_, (xx, yy), textcoords='offset points', xytext=(5, 4), fontsize=11)
    lim = np.nanmax(np.abs(np.r_[xa, ya])) * 1.15 if np.isfinite(np.r_[xa, ya]).any() else 1
    a.plot([-lim, lim], [-lim, lim], color='0.5', ls=':'); a.axhline(0, color='0.8'); a.axvline(0, color='0.8')
    a.set_xlabel('Efeito com .atr'); T.negrito_ticks(a)
axs[0, 0].set_ylabel('Detector − anotado (por janela)')
axs[1, 0].set_ylabel('Efeito com o detector (por registro)')
plt.tight_layout(); plt.savefig(os.path.join(PASTA_FIG, 'fig_qrs_concordancia.png')); plt.show()""")

md(r"""## 6. Braço de VFC com n = 13 (detector) contra os pseudo-eventos (detector)
Três comparações lado a lado, para separar o efeito **do detector** do efeito **de acrescentar registros**:
- **A** — SDDB 7 anotado × NSRDB anotado (reproduz o notebook 06);
- **B** — SDDB 7 detector × NSRDB detector (mesmos registros, só troca a fonte do RR);
- **C** — SDDB 13 detector × NSRDB detector (a análise nova).

Se A e B concordam, o detector não muda a conclusão; C diz o que os 6 registros novos acrescentam.""")
code(r"""def comparar(sddb_df, nsr_df, rotulo, basal):
    e_s = T.efeito_ultima_hora(sddb_df, basal, TERMINAL, INDICES)
    e_n = T.efeito_ultima_hora(nsr_df, basal, TERMINAL, INDICES)
    out = []
    for ind in INDICES:
        out.append({'comparacao': rotulo, 'indice': ind, **T.hedges_g_boot(e_s[f'{ind}_delta'], e_n[f'{ind}_delta'])})
    return out, e_s, e_n

S, N = jan[jan.base == 'SDDB'], jan[jan.base == 'NSRDB']
efeitos, guard = [], {}
for basal_rot, lim in [('basal antes de -2 h', BASAL), ('basal antes de -6 h', BASAL_SENS)]:
    for rot, sd, nd in [('A: 7 anotado', S[(S.fonte == 'anotado')], N[N.fonte == 'anotado']),
                        ('B: 7 detector', S[(S.fonte == 'detector') & S.registro.isin(REG_ATR)], N[N.fonte == 'detector']),
                        ('C: 13 detector', S[S.fonte == 'detector'], N[N.fonte == 'detector'])]:
        o, e_s, e_n = comparar(sd, nd, rot, lim)
        for x in o:
            x['basal'] = basal_rot
        efeitos += o
        guard[(basal_rot, rot)] = (e_s, e_n)
efe = pd.DataFrame(efeitos)
efe.to_csv(os.path.join(PASTA_CSV, 'qrs_efeitos.csv'), index=False)
print('g = Hedges g (SDDB − NSRDB) do efeito "última hora − basal"; media_a = SDDB, media_b = NSRDB')
display(efe[efe.basal == 'basal antes de -2 h'].round(3))
e13 = guard[('basal antes de -2 h', 'C: 13 detector')][0]
print('Registros com efeito calculável na análise C (precisam de >= 3 janelas aprovadas na última hora e no basal):',
      e13.dropna(subset=['FC_delta']).registro.tolist())""")

code(r"""fig, axs = plt.subplots(1, 4, figsize=(18, 5.5))
e_s, e_n = guard[('basal antes de -2 h', 'C: 13 detector')]
for a, ind in zip(axs, INDICES):
    vn = e_n[f'{ind}_delta'].dropna()
    a.scatter(np.random.default_rng(1).uniform(-0.08, 0.08, len(vn)), vn, color='tab:blue', s=40, zorder=3)
    a.hlines(vn.median(), -0.25, 0.25, color='k', lw=3)
    for orig, mk, cor, lab in [(REG_ATR, 'o', 'tab:red', 'SDDB anotados (7)'), (REG_NOVOS, 'D', 'darkorange', 'SDDB novos (6)')]:
        d = e_s[e_s.registro.isin(orig)][f'{ind}_delta'].dropna()
        a.scatter(1 + np.random.default_rng(2).uniform(-0.08, 0.08, len(d)), d, color=cor, marker=mk, s=50, zorder=3,
                  label=lab if ind == 'FC' else None)
    a.hlines(e_s[f'{ind}_delta'].median(), 0.75, 1.25, color='k', lw=3)
    a.axhline(0, color='0.5', ls=':')
    rr_ = efe[(efe.basal == 'basal antes de -2 h') & (efe.comparacao == 'C: 13 detector') & (efe.indice == ind)].iloc[0]
    a.set_title(f"{NOMES[ind]}\ng = {T.num(rr_.g)} {T.ic(rr_.ic_lo, rr_.ic_hi)}", fontsize=12)
    a.set_xticks([0, 1]); a.set_xticklabels(['NSRDB\npseudo', 'SDDB\nFV']); a.set_xlim(-0.5, 1.5); T.negrito_ticks(a)
axs[0].set_ylabel('Última hora − basal (por registro)')
fig.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.04))
plt.tight_layout(rect=(0, 0.06, 1, 1)); plt.savefig(os.path.join(PASTA_FIG, 'fig_qrs_efeito.png')); plt.show()""")

md(r"""## 7. Portão de qualidade com o detector
Se o detector perde batimentos perto da FV (ectopia, ruído), mais janelas da última hora são reprovadas — o mesmo viés de seleção
dependente do desfecho discutido no notebook 06, agora possivelmente maior.""")
code(r"""cob = jan.assign(hora=np.floor((jan.min_ate_evento - 1e-6) / 60.0)).groupby(['base', 'fonte', 'hora']).passed.mean()
cob = (100 * cob).unstack(['base', 'fonte'])
fig, a = plt.subplots(figsize=(11, 5))
for (b, f), cor, ls in [(('SDDB', 'anotado'), 'tab:red', '-'), (('SDDB', 'detector'), 'tab:red', '--'),
                        (('NSRDB', 'anotado'), 'tab:blue', '-'), (('NSRDB', 'detector'), 'tab:blue', '--')]:
    if (b, f) in cob:
        a.plot(cob.index + 0.5, cob[(b, f)], ls, marker='o', color=cor, lw=2.2, label=f'{b} — {f}')
a.set_xlabel('Horas antes do evento'); a.set_ylabel('Janelas aprovadas no portão (%)')
a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=11); T.negrito_ticks(a)
plt.tight_layout(); plt.savefig(os.path.join(PASTA_FIG, 'fig_qrs_portao.png')); plt.show()
display(cob.round(0).tail(6))""")

md(r"""## O que me mandar
1. A tabela da seção 1 e a figura `fig_qrs_metodo.png`.
2. `res_val` e `disp` (seção 3) e a figura `fig_qrs_validacao.png` — **isto decide se dá para seguir**.
3. A tabela `conc`, a tabela de efeitos por registro e `fig_qrs_concordancia.png` (seção 5).
4. A tabela `efe` (seção 6), a lista de registros com efeito calculável e `fig_qrs_efeito.png`.
5. `fig_qrs_portao.png`.

**Limitações a declarar no texto:** a validação da Nicole na PTB usou o `xqrs` como verdade (não anotação humana); aqui a verdade é o
`.atr` auditado da SDDB, o que é mais forte, mas só existe para os registros anotados — nos 6 novos a qualidade é **inferida** da
validação, não medida. O limiar relativo (75 % do máximo da janela) pode perder batimentos normais numa janela com uma extrassístole de
grande amplitude; a validação separada da última hora existe para medir exatamente isso.""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}, "colab": {"provenance": []}}
nbf.write(nb, "/home/claude/tcc/10_qrs_matriz_sddb.ipynb")
print("ok")
