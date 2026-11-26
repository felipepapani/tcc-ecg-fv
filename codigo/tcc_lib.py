"""
tcc_lib.py — funções reutilizáveis do TCC (MVTDB / SDDB, alerta precoce da FV).

Uso no Colab:
    import sys; sys.path.append('/content/drive/MyDrive/Pesquisa/TCC/codigo')
    import importlib, tcc_lib as T; importlib.reload(T)

Convenções
----------
* RR sempre em milissegundos (np.ndarray de float).
* "evento"  = série que termina na detecção da TV/FV pelo CDI (RRdata1/*.vtN, *.vfN).
* "controle" = série de ritmo de base do mesmo paciente (mr/*.mrN, *.mra).
* Tudo que é texto de figura sai em INGLÊS e em NEGRITO (estilo_figuras()).
"""
from __future__ import annotations

import io
import os
import re
import zipfile
import urllib.request

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes da base
# ---------------------------------------------------------------------------
URL_ZIP_MVTDB = "https://physionet.org/content/mvtdb/get-zip/1.0/"
RE_EVENTO = re.compile(r"\b(\d{4})\.(vf|vt)(\d+)\b", re.IGNORECASE)
RE_CONTROLE = re.compile(r"\b(\d{4})\.mr(a|\d+)\b", re.IGNORECASE)

# Tamanhos dos conjuntos descritos na documentação da MVTDB (PhysioNet)
ESPERADO = {"MRVT": 58, "MRVF": 25, "MultiVT": 48, "MultiVF": 4}


# ---------------------------------------------------------------------------
# 1. Download e localização dos arquivos
# ---------------------------------------------------------------------------
def achar_raiz_mvtdb(pasta: str) -> str | None:
    """Procura, a partir de `pasta`, o diretório que contém RRdata1/ e mr/."""
    if not os.path.isdir(pasta):
        return None
    for raiz, dirs, _ in os.walk(pasta):
        nomes = {d.lower() for d in dirs}
        if "rrdata1" in nomes and "mr" in nomes:
            return raiz
    return None


def baixar_mvtdb(destino: str, url: str = URL_ZIP_MVTDB) -> str:
    """Baixa o ZIP da MVTDB (~2.6 MB) e extrai em `destino`, se ainda não existir.
    Retorna a raiz (pasta com RRdata1/ e mr/)."""
    raiz = achar_raiz_mvtdb(destino)
    if raiz:
        print(f"MVTDB já presente em: {raiz}")
        return raiz
    os.makedirs(destino, exist_ok=True)
    print(f"Baixando {url} ...")
    with urllib.request.urlopen(url) as resp:
        dados = resp.read()
    with zipfile.ZipFile(io.BytesIO(dados)) as z:
        z.extractall(destino)
    raiz = achar_raiz_mvtdb(destino)
    if raiz is None:
        raise RuntimeError("ZIP extraído, mas RRdata1/ e mr/ não foram encontrados.")
    print(f"MVTDB extraída em: {raiz}")
    return raiz


def achar_planilha(raiz: str) -> str | None:
    """Localiza phys.xls (ou variações) dentro da raiz."""
    for r, _, arqs in os.walk(raiz):
        for a in arqs:
            if a.lower().startswith("phys") and a.lower().endswith((".xls", ".xlsx", ".csv")):
                return os.path.join(r, a)
    return None


def listar_arquivos(raiz: str) -> dict:
    """Mapeia nome do arquivo (ex. '0001.vt1') -> caminho, para eventos e controles."""
    saida = {}
    for r, _, arqs in os.walk(raiz):
        for a in arqs:
            nome = a.lower()
            if RE_EVENTO.fullmatch(nome) or RE_CONTROLE.fullmatch(nome):
                saida[nome] = os.path.join(r, a)
    return saida


def ler_rr(caminho: str) -> np.ndarray:
    """Lê um arquivo texto com um RR (ms) por linha. Ignora linhas não numéricas."""
    vals = []
    with open(caminho, "r", errors="ignore") as f:
        for linha in f:
            s = linha.strip().replace(",", ".")
            if not s:
                continue
            try:
                vals.append(float(s.split()[0]))
            except ValueError:
                continue
    return np.asarray(vals, dtype=float)


# ---------------------------------------------------------------------------
# 2. Tabela de pares evento–controle
# ---------------------------------------------------------------------------
def _info_evento(nome: str) -> dict:
    m = RE_EVENTO.fullmatch(nome.lower())
    return {"paciente": m.group(1), "tipo": m.group(2).upper(), "seq": int(m.group(3))}


def ler_planilha(caminho: str) -> dict:
    """Lê TODAS as abas de phys.xls / .xlsx (ou um .csv) sem assumir cabeçalho.
    Retorna {nome_da_aba: DataFrame}. Para .xls o Colab usa o pacote xlrd."""
    ext = os.path.splitext(caminho)[1].lower()
    if ext == ".csv":
        return {"csv": pd.read_csv(caminho, header=None, dtype=object)}
    abas = pd.read_excel(caminho, header=None, sheet_name=None)
    return {k: v for k, v in abas.items() if not v.dropna(how="all").empty}


# Colunas clínicas úteis do phys.xls -> nome no TCC (busca pelo texto do cabeçalho)
COLUNAS_CLINICAS = {
    "id_paciente": ("patient no",),
    "classe_cdi": ("device class",),
    "data_episodio": ("episode date",),
    "hora_episodio": ("episode time",),
    "classe_manual": ("s&d class",),
    "egm_prior": ("egm prior",),
    "egm_fonte": ("egm source",),
    "sexo": ("sex", "gender"),
    "idade": ("age",),
    "altura": ("height",),
    "peso": ("weight",),
    "fe": ("ef", "ejection fraction", "lvef"),
    "nyha": ("nyha",),
}


def _linha_cabecalho(df: pd.DataFrame) -> int | None:
    """Índice da linha que contém 'file name' (cabeçalho da aba)."""
    for i in range(min(15, len(df))):
        txt = " ".join(str(v).lower() for v in df.iloc[i].tolist())
        if "file name" in txt:
            return i
    return None


def _mapear_colunas(cabecalho: list) -> dict:
    """Para cada coluna clínica, posição da 1ª coluna cujo cabeçalho casa com a chave
    (primeiro igualdade exata, depois 'começa com')."""
    norm = lambda x: re.sub(r"[^a-z0-9]", "", str(x).lower())
    cab = [norm(c) for c in cabecalho]
    pos = {}
    for nome, chaves in COLUNAS_CLINICAS.items():
        chaves = [norm(k) for k in chaves]
        achou = next((j for j, c in enumerate(cab) for k in chaves if c == k), None)
        if achou is None:
            achou = next((j for j, c in enumerate(cab) for k in chaves if c.startswith(k)), None)
        if achou is not None:
            pos[nome] = achou
    return pos


def resumo_abas(planilha: dict) -> pd.DataFrame:
    """Uma linha por aba: nome, nº de linhas de dados, cabeçalho da coluna de arquivo."""
    out = []
    for aba, df in planilha.items():
        h = _linha_cabecalho(df)
        cab = df.iloc[h].tolist() if h is not None else []
        col_arq = next((str(c) for c in cab if "file name" in str(c).lower()), "")
        n = len(df) - (h + 1 if h is not None else 0)
        out.append({"aba": aba, "linhas_dados": n, "coluna_arquivo": col_arq,
                    "conjunto_pela_aba": _conjunto_da_aba(aba)})
    return pd.DataFrame(out)


def _data_da_linha(valores) -> pd.Timestamp | None:
    """Primeira célula da linha que seja (ou pareça) data. None se não houver."""
    for v in valores:
        if isinstance(v, (pd.Timestamp, np.datetime64)) or hasattr(v, "year"):
            try:
                ts = pd.Timestamp(v)
                if 1980 <= ts.year <= 2030:
                    return ts
            except Exception:
                pass
    for v in valores:
        s = str(v)
        if re.search(r"\d{1,4}[/-]\d{1,2}[/-]\d{1,4}", s):
            ts = pd.to_datetime(s, errors="coerce")
            if ts is not pd.NaT and not pd.isna(ts) and 1980 <= ts.year <= 2030:
                return ts
    return None


def _vazio(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() in ("", "nan", "NaT")


def _data_hora(data, hora) -> pd.Timestamp | None:
    """Junta 'Episode date' + 'Episode time' num Timestamp (None se a data faltar)."""
    if _vazio(data) or str(data).strip().upper() == "A":
        return None
    ts = pd.to_datetime(data, errors="coerce")
    if pd.isna(ts):
        return None
    if not _vazio(hora) and str(hora).strip().upper() != "A":
        td = pd.to_timedelta(str(hora).strip(), errors="coerce")
        if not pd.isna(td):
            ts = ts.normalize() + td
    return ts


def pares_da_planilha(planilha) -> pd.DataFrame:
    """Extrai os pares de TODAS as abas da planilha clínica.
    Em cada linha procura (regex) um nome de evento (ssss.vtN / ssss.vfN) e um de
    controle (ssss.mrN / ssss.mra); guarda a aba de origem e as colunas clínicas
    reconhecidas pelo cabeçalho (COLUNAS_CLINICAS). 'A' na planilha = ausente -> NaN.
    `planilha`: caminho, DataFrame único ou dict {aba: DataFrame}.
    Linhas cujo evento e controle são de pacientes diferentes recebem alerta."""
    if isinstance(planilha, str):
        planilha = ler_planilha(planilha)
    elif isinstance(planilha, pd.DataFrame):
        planilha = {"aba1": planilha}

    linhas = []
    for aba, df in planilha.items():
        h = _linha_cabecalho(df)
        pos = _mapear_colunas(df.iloc[h].tolist()) if h is not None else {}
        inicio = h + 1 if h is not None else 0
        for i in range(inicio, len(df)):
            row = df.iloc[i].tolist()
            valores = [v for v in row if not _vazio(v)]
            texto = " ".join(str(v) for v in valores).lower()
            ev, ct = RE_EVENTO.search(texto), RE_CONTROLE.search(texto)
            if not ev:
                continue
            evento = ev.group(0)
            info = _info_evento(evento)
            if ct:
                controle = ct.group(0)
                alerta = "" if ct.group(1) == info["paciente"] else "paciente_diferente"
            else:   # coluna do controle digitada errado (ex.: '0095.vt2' no lugar de '0095.mr2')
                controle = f"{info['paciente']}.mr{info['seq']}"
                alerta = "controle_inferido"
            reg = {**info, "evento": evento, "controle": controle, "aba": aba,
                   "linha_planilha": i, "alerta": alerta}
            for nome, j in pos.items():
                v = row[j]
                reg[nome] = np.nan if (_vazio(v) or str(v).strip().upper() == "A") else v
            if "data_episodio" in pos:
                reg["data"] = _data_hora(row[pos["data_episodio"]],
                                         row[pos["hora_episodio"]] if "hora_episodio" in pos else None)
            else:
                reg["data"] = _data_da_linha(valores)
            linhas.append(reg)

    out = pd.DataFrame(linhas)
    if out.empty:
        return out
    out = out.drop_duplicates(subset=["aba", "evento"]).reset_index(drop=True)
    dup = out[out.duplicated("evento", keep=False)]
    if len(dup):
        print(f"AVISO: {dup['evento'].nunique()} evento(s) aparecem em mais de uma aba:",
              sorted(dup['evento'].unique())[:10])
    return out


def corrigir_nomes(pares: pd.DataFrame, arquivos: dict) -> pd.DataFrame:
    """Corrige nomes de evento que não existem em disco (erro de digitação na planilha,
    ex.: '0159.vt1' na aba MRVF quando o arquivo é '0159.vf1').
    Para cada evento ausente procura, em ordem:
      1) mesmo paciente e sequência com o tipo da ABA (MRVF/MultVF -> vf; MRVT/MultVT -> vt);
      2) mesmo paciente e sequência com o outro tipo;
    aceitando só candidatos que existam em disco e não estejam usados por outro par.
    Atualiza evento/tipo/seq e registra em 'correcao'. O mesmo para controles ausentes
    (ssss.mrN <-> ssss.mra)."""
    p = pares.copy()
    p["correcao"] = ""
    usados = set(p["evento"])
    for i, r in p.iterrows():
        if r["evento"] in arquivos:
            continue
        cj = _conjunto_da_aba(r.get("aba", "")) or ""
        tipo_aba = cj[-2:].lower() if cj else r["tipo"].lower()
        outro = "vt" if tipo_aba == "vf" else "vf"
        cands = [f"{r['paciente']}.{t}{r['seq']}" for t in (tipo_aba, outro)]
        novo = next((c for c in cands if c in arquivos and c not in usados), None)
        if novo:
            info = _info_evento(novo)
            p.loc[i, ["evento", "tipo", "seq"]] = [novo, info["tipo"], info["seq"]]
            p.loc[i, "correcao"] = f"evento {r['evento']} -> {novo}"
            usados.add(novo)
    for i, r in p.iterrows():
        if r["controle"] in arquivos:
            continue
        pac = r["controle"][:4]
        cands = [f"{pac}.mra", f"{pac}.mr{r['seq']}"]
        novo = next((c for c in cands if c in arquivos), None)
        if novo:
            p.loc[i, "controle"] = novo
            p.loc[i, "correcao"] = (p.loc[i, "correcao"] + f"; controle {r['controle']} -> {novo}").strip("; ")
    return p


def eventos_fora_da_planilha(pares: pd.DataFrame, arquivos: dict) -> pd.DataFrame:
    """Arquivos de evento que existem em disco mas não aparecem em nenhum par,
    com os controles do mesmo paciente disponíveis (para achar pares faltantes)."""
    usados = set(pares["evento"])
    linhas = []
    for nome in sorted(arquivos):
        if RE_EVENTO.fullmatch(nome) and nome not in usados:
            pac = nome[:4]
            ctrls = sorted(c for c in arquivos if RE_CONTROLE.fullmatch(c) and c.startswith(pac))
            ctrls_livres = [c for c in ctrls if c not in set(pares["controle"])]
            linhas.append({"evento": nome, "controles_do_paciente": ", ".join(ctrls),
                           "controles_nao_usados": ", ".join(ctrls_livres)})
    return pd.DataFrame(linhas)


def pares_por_nome(arquivos: dict) -> tuple[pd.DataFrame, list]:
    """Plano B (sem planilha): pareia cada evento ssss.xxN com ssss.mrN do mesmo paciente;
    para vf1 usa ssss.mra se existir. Retorna (pares, eventos_sem_par)."""
    linhas, sem_par = [], []
    for nome in sorted(arquivos):
        if not RE_EVENTO.fullmatch(nome):
            continue
        info = _info_evento(nome)
        p, n = info["paciente"], info["seq"]
        candidatos = []
        if info["tipo"] == "VF" and n == 1:
            candidatos.append(f"{p}.mra")
        candidatos.append(f"{p}.mr{n}")
        ctrl = next((c for c in candidatos if c in arquivos), None)
        if ctrl is None:
            sem_par.append(nome)
            continue
        linhas.append({**info, "evento": nome, "controle": ctrl, "data": None,
                       "linha_planilha": np.nan, "alerta": "pareado_por_nome"})
    return pd.DataFrame(linhas), sem_par


def _conjunto_da_aba(nome) -> str | None:
    """'MRVT', 'MR VT', 'mr_vt' -> 'MRVT'; 'Multi VF' -> 'MultiVF'; senão None."""
    s = re.sub(r"[^a-z]", "", str(nome).lower()).replace("multi", "mult")
    for chave, c in [("multvt", "MultiVT"), ("multvf", "MultiVF"), ("mrvt", "MRVT"), ("mrvf", "MRVF")]:
        if chave in s:
            return c
    return None


def marcar_conjuntos(pares: pd.DataFrame) -> pd.DataFrame:
    """Define o conjunto de cada par.

    * conjunto_calc: calculado — evento MAIS RECENTE de cada paciente×tipo = MRVT/MRVF,
      demais = MultiVT/MultiVF (maior data+hora se todas existirem; senão maior seq).
    * conjunto: se todas as abas tiverem nome reconhecível (MRVT, MRVF, MultiVT,
      MultiVF), usa a ABA (é a definição da própria base); senão usa conjunto_calc.
    A coluna 'concorda' mostra se aba e cálculo batem (checagem de sanidade)."""
    pares = pares.copy()
    unicos = pares.drop_duplicates("evento").set_index("evento")
    calc, crit = {}, {}
    for (_, tipo), g in unicos.groupby(["paciente", "tipo"]):
        datas = pd.to_datetime(g["data"], errors="coerce")
        if datas.notna().all() and datas.nunique() == len(g):
            rec, c = datas.idxmax(), "data"
        else:
            rec, c = g["seq"].idxmax(), "seq"
        for ev in g.index:
            calc[ev] = f"MR{tipo}" if ev == rec else f"Multi{tipo}"
            crit[ev] = c
    pares["conjunto_calc"] = pares["evento"].map(calc)
    pares["criterio_recente"] = pares["evento"].map(crit)

    if "aba" in pares and pares["aba"].map(_conjunto_da_aba).notna().all():
        pares["conjunto"] = pares["aba"].map(_conjunto_da_aba)
        pares["origem_conjunto"] = "aba"
    else:
        pares["conjunto"] = pares["conjunto_calc"]
        pares["origem_conjunto"] = "calculado"
    pares["concorda"] = pares["conjunto"] == pares["conjunto_calc"]
    return pares


def diagnostico_conjuntos(pares: pd.DataFrame) -> pd.DataFrame:
    """Explica diferenças entre as contagens obtidas e as da documentação.
    Para cada par marca:
      dup_entre_abas    - o mesmo evento aparece em mais de uma aba
      sem_MR_do_paciente - par Multi* cujo paciente não tem evento MR* do mesmo tipo
      mais_novo_que_MR  - par Multi* com data posterior ao evento MR* do paciente
      igual_ao_MR       - par Multi* que é o próprio evento MR* (mesmo arquivo)
    Retorna só as linhas com pelo menos uma marca."""
    p = pares.copy()
    p["dup_entre_abas"] = p.duplicated("evento", keep=False)
    mr = p[p["conjunto"].str.startswith("MR")].set_index(["paciente", "tipo"])
    mr = mr[~mr.index.duplicated()]
    def _flags(r):
        if not str(r["conjunto"]).startswith("Multi"):
            return pd.Series({"sem_MR_do_paciente": False, "mais_novo_que_MR": False, "igual_ao_MR": False})
        chave = (r["paciente"], r["tipo"])
        if chave not in mr.index:
            return pd.Series({"sem_MR_do_paciente": True, "mais_novo_que_MR": False, "igual_ao_MR": False})
        ref = mr.loc[chave]
        d, dref = pd.to_datetime(r["data"], errors="coerce"), pd.to_datetime(ref["data"], errors="coerce")
        return pd.Series({"sem_MR_do_paciente": False,
                          "mais_novo_que_MR": bool(pd.notna(d) and pd.notna(dref) and d > dref),
                          "igual_ao_MR": r["evento"] == ref["evento"]})
    p = pd.concat([p, p.apply(_flags, axis=1)], axis=1)
    cols = ["dup_entre_abas", "sem_MR_do_paciente", "mais_novo_que_MR", "igual_ao_MR"]
    print("Contagem de marcas:", {c: int(p[c].sum()) for c in cols})
    return p.loc[p[cols].any(axis=1), ["aba", "evento", "controle", "data", "conjunto",
                                         "conjunto_calc"] + cols]


def conferir_conjuntos(pares: pd.DataFrame) -> pd.DataFrame:
    """Compara as contagens obtidas com as da documentação (58/25/48/4)."""
    obtido = pares["conjunto"].value_counts()
    return pd.DataFrame({"esperado": pd.Series(ESPERADO),
                         "obtido": obtido.reindex(list(ESPERADO)).fillna(0).astype(int)})


# ---------------------------------------------------------------------------
# 3. Pré-processamento das séries RR
# ---------------------------------------------------------------------------
def localizar_degrau(rr: np.ndarray, frac: float = 0.75, janela: int = 8, n_pre: int = 30,
                     busca_max: int = 200, estat: str = "mediana") -> tuple:
    """Posição s do degrau (primeiro intervalo rápido) usada por remover_inicio_arritmia,
    ou (None, nan) se nenhum candidato passa. Retorna (s, razao).
    Para cada s nos últimos `busca_max` intervalos:
        razao  = estat(rr[s : s+janela]) / estat(rr[s-n_pre : s])
        sufixo = estat(rr[s : ])          / estat(rr[s-n_pre : s])
    válido se razao < frac e sufixo < frac; escolhe o de menor razao."""
    f = np.median if estat == "mediana" else np.mean
    rr = np.asarray(rr, float)
    n = len(rr)
    ini = max(n_pre, n - busca_max)
    melhor, s_melhor = np.inf, None
    for s in range(ini, n - janela + 1):
        pre = f(rr[s - n_pre:s])
        if pre <= 0:
            continue
        razao = f(rr[s:s + janela]) / pre
        sufixo = f(rr[s:]) / pre
        if razao < frac and sufixo < frac and razao < melhor:
            melhor, s_melhor = razao, s
    return (s_melhor, (melhor if s_melhor is not None else np.nan))


def refinar_inicio(rr: np.ndarray, s, frac: float = 0.75, n_ref: int = 60):
    """Recua a posição do degrau até o PRIMEIRO intervalo rápido.
    Por que existe: localizar_degrau escolhe o s de menor razão, e como a referência é uma
    MEDIANA de 30 intervalos (robusta), ela quase não muda quando os primeiros batimentos de
    TV entram nela — então, com ruído na TV, o s escolhido pode cair alguns batimentos DEPOIS
    do início real. Aqui: nível de referência = mediana dos `n_ref` intervalos antes de s;
    recua enquanto o intervalo anterior for < frac × referência. Retorna s_ref <= s (ou None)."""
    if s is None:
        return None
    rr = np.asarray(rr, float)
    ref = np.median(rr[max(0, s - n_ref):s])
    j = s
    while j - 1 >= 0 and rr[j - 1] < frac * ref:
        j -= 1
    return j


def corte_de_degrau(n: int, s, guarda: int = 10, n_min: int = 30) -> tuple:
    """Regra final do corte a partir da posição do degrau (s ou None): (k, metodo).
    Separada de localizar_degrau para o estudo de limiares variar guarda e n_min sem
    procurar o degrau de novo. Idêntica à usada em remover_inicio_arritmia."""
    if s is None:
        k, metodo = n - n_min, "minimo"
    else:
        k, metodo = s - guarda, "degrau"
        if k > n - n_min:
            k, metodo = n - n_min, "minimo"
    return max(0, k), metodo


def remover_inicio_arritmia(rr: np.ndarray, frac: float = 0.75, janela: int = 8,
                            n_pre: int = 30, guarda: int = 10, busca_max: int = 200,
                            n_min: int = 30, detalhe: bool = False, estat: str = "mediana"):
    """A série de evento termina na detecção do CDI, então seus últimos intervalos já são
    a taquiarritmia. Retorna o índice de corte k (use rr[:k]); com detalhe=True retorna
    (k, metodo).

    Detecta o INÍCIO da taquiarritmia como um DEGRAU: para cada candidato s nos últimos
    `busca_max` intervalos, compara
        razao  = mediana(rr[s : s+janela]) / mediana(rr[s-n_pre : s])   (queda abrupta)
        sufixo = mediana(rr[s : ])          / mediana(rr[s-n_pre : s])   (fica rápido até o fim)
    Candidatos válidos: razao < frac e sufixo < frac. Escolhe o de MENOR razao (maior degrau).
    Uma aceleração gradual (sinusal) não produz degrau e por isso NÃO é cortada.
    O corte fica `guarda` intervalos antes do degrau, com corte mínimo de `n_min`
    (cobre a detecção do CDI quando a TV não é 25 % mais rápida que o ritmo anterior).
    metodo: 'degrau' (achou o início) ou 'minimo' (usou n_min).
    `estat`: 'mediana' (padrão) ou 'media' — só para o estudo de limiares (notebook 09)."""
    rr = np.asarray(rr, float)
    n = len(rr)
    s_melhor, _ = localizar_degrau(rr, frac=frac, janela=janela, n_pre=n_pre,
                                   busca_max=busca_max, estat=estat)
    k, metodo = corte_de_degrau(n, s_melhor, guarda=guarda, n_min=n_min)
    return (k, metodo) if detalhe else k


def fc_valida(rr: np.ndarray, lo: float = 250, hi: float = 2500) -> float:
    """FC média (bpm) usando TODOS os intervalos fisiologicamente possíveis
    (frequência ventricular real; não remove ectópicos)."""
    rr = np.asarray(rr, float)
    return fc_media(rr[(rr >= lo) & (rr <= hi)])


def marcar_ectopicos(rr: np.ndarray, lo: float = 250, hi: float = 2500, dif: float = 0.20,
                     janela: int = 11) -> dict:
    """Máscaras do filtro de ectópicos (mesma regra de limpar_rr):
    ok      = intervalo fisiologicamente possível ([lo, hi] ms);
    ect     = difere mais de `dif` da mediana local de `janela` intervalos;
    curto   = ect e mais curto que a mediana local (batimento PREMATURO — extrassístole);
    longo   = ect e mais longo (pausa, batimento falhado);
    seg     = intervalo logo após um ect (pausa compensatória);
    base    = a mediana local usada como referência."""
    rr = np.asarray(rr, float)
    ok = (rr >= lo) & (rr <= hi)
    base = pd.Series(np.where(ok, rr, np.nan)).rolling(janela, center=True, min_periods=3).median()
    base = base.bfill().ffill().to_numpy()
    ect = ok & (np.abs(rr - base) > dif * base)
    seg = np.zeros_like(ect)
    seg[1:] = ect[:-1]
    return {"ok": ok, "ect": ect, "curto": ect & (rr < base), "longo": ect & (rr > base),
            "seg": seg, "base": base}


def carga_ectopica(rr: np.ndarray, dif: float = 0.20, janela: int = 11, ini=None, fim=None) -> dict:
    """Carga de ectopia de um trecho RR (notebook 08):
    prematuros_100 = batimentos prematuros por 100 intervalos;
    ectopicos_100  = qualquer intervalo marcado (curto ou longo) por 100 intervalos;
    salvas         = sequências de >= 2 prematuros consecutivos (pares/salvas);
    algum          = 1 se houve pelo menos um prematuro.
    Limitação: só a partir do RR não dá para separar extrassístole ventricular de atrial
    nem de artefato — é uma medida de irregularidade prematura, não de origem.
    `ini`, `fim`: trecho analisado (fatia rr[ini:fim]). As marcas são calculadas na série
    INTEIRA e só depois fatiadas, para a mediana local não ter efeito de borda no trecho."""
    rr = np.asarray(rr, float)
    vazio = {"prematuros_100": np.nan, "ectopicos_100": np.nan, "salvas": np.nan, "algum": np.nan}
    if len(rr) == 0:
        return vazio
    mk = marcar_ectopicos(rr, dif=dif, janela=janela)
    fat = slice(ini, fim)
    mk = {k: v[fat] for k, v in mk.items()}
    if len(mk["ect"]) == 0:
        return vazio
    c = mk["curto"].astype(int)
    inicio_run = np.diff(np.r_[0, c]) == 1
    fim_run = np.diff(np.r_[c, 0]) == -1
    comp = np.flatnonzero(fim_run) - np.flatnonzero(inicio_run) + 1
    return {"prematuros_100": 100 * c.mean(), "ectopicos_100": 100 * mk["ect"].mean(),
            "salvas": int(np.sum(comp >= 2)), "algum": int(c.any())}


def limpar_rr(rr: np.ndarray, lo: float = 250, hi: float = 2500, dif: float = 0.20,
              janela: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Remove RR fisiologicamente impossíveis (fora de [lo, hi] ms) e prováveis ectópicos:
    intervalo que difere mais de `dif` (fração) da mediana local, junto com o intervalo
    seguinte (pausa compensatória). Retorna (rr_limpo, mascara_valida)."""
    rr = np.asarray(rr, float)
    mk = marcar_ectopicos(rr, lo=lo, hi=hi, dif=dif, janela=janela)
    ok, ect, seg = mk["ok"], mk["ect"], mk["seg"]
    valido = ok & ~ect & ~seg
    return rr[valido], valido


def prop_ectopia(rr: np.ndarray, dif: float = 0.20, janela: int = 11) -> float:
    """Fração dos intervalos marcados como ectópicos por limpar_rr (proxy simples)."""
    _, valido = limpar_rr(rr, dif=dif, janela=janela)
    return float(1 - valido.mean()) if len(valido) else np.nan


def fc_media(rr: np.ndarray) -> float:
    """Frequência cardíaca média (bpm) a partir de RR em ms."""
    rr = np.asarray(rr, float)
    return float(60000.0 / rr.mean()) if len(rr) else np.nan


def interpolar_ectopicos(rr: np.ndarray, **kw) -> tuple[np.ndarray, np.ndarray]:
    """Substitui os intervalos marcados por limpar_rr (impossíveis, ectópicos e a pausa
    seguinte) por interpolação linear entre os vizinhos válidos. Mantém o número de
    batimentos (alinhamento no tempo), o que é necessário para as janelas móveis do CSD.
    Retorna (rr_interpolado, mascara_valida)."""
    rr = np.asarray(rr, float)
    _, valido = limpar_rr(rr, **kw)
    if valido.sum() < 2:
        return rr.copy(), valido
    idx = np.arange(len(rr))
    out = rr.copy()
    out[~valido] = np.interp(idx[~valido], idx[valido], rr[valido])
    return out, valido


# ---------------------------------------------------------------------------
# 3b. Indicadores de desaceleração crítica (CSD) — Scheffer 2009, Dakos 2012
# ---------------------------------------------------------------------------
def remover_tendencia(x: np.ndarray, metodo: str = "gauss", bw: float = 0.10):
    """Remove a tendência lenta antes de calcular os indicadores.
    metodo: 'gauss' (filtro gaussiano, sigma = bw * n), 'linear' ou 'nenhum'.
    Retorna (residuo, tendencia)."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import detrend
    x = np.asarray(x, float)
    if metodo == "gauss":
        tend = gaussian_filter1d(x, sigma=max(1.0, bw * len(x)), mode="nearest")
    elif metodo == "linear":
        tend = x - detrend(x, type="linear")
    elif metodo == "nenhum":
        tend = np.full_like(x, x.mean())
    else:
        raise ValueError(f"metodo desconhecido: {metodo}")
    return x - tend, tend


def indicadores_janela(x_bruto: np.ndarray, residuo: np.ndarray, w: float = 0.50) -> pd.DataFrame:
    """Indicadores em janela móvel (tamanho = w * n batimentos), indexados pelo fim da janela:
    sd (desvio padrão do resíduo), cv (sd / média do RR bruto), ar1 (autocorrelação lag-1
    do resíduo), skew (assimetria do resíduo)."""
    n = len(residuo)
    win = max(10, int(round(w * n)))
    r = pd.Series(residuo)
    sd = r.rolling(win).std()
    df = pd.DataFrame({
        "sd": sd,
        "cv": sd / pd.Series(x_bruto).rolling(win).mean(),
        "ar1": r.rolling(win).corr(r.shift(1)),
        "skew": r.rolling(win).skew(),
    })
    return df.iloc[win - 1:]


def tendencia_kendall(ind: pd.DataFrame) -> dict:
    """Tau de Kendall de cada indicador contra o tempo (tau > 0 = indicador subindo)."""
    from scipy.stats import kendalltau
    out = {}
    for c in ind.columns:
        y = ind[c].to_numpy()
        m = np.isfinite(y)
        out[f"tau_{c}"] = kendalltau(np.arange(m.sum()), y[m]).statistic if m.sum() > 10 else np.nan
    return out


def csd_serie(rr: np.ndarray, w: float = 0.50, bw: float = 0.10, metodo: str = "gauss",
              retornar_janelas: bool = False):
    """Pipeline completo para uma série RR já cortada e interpolada:
    remove tendência -> indicadores em janela móvel -> tau de Kendall."""
    res, tend = remover_tendencia(rr, metodo=metodo, bw=bw)
    ind = indicadores_janela(rr, res, w=w)
    taus = tendencia_kendall(ind)
    if retornar_janelas:
        return taus, ind, res, tend
    return taus


def preparar_series(fc_df: pd.DataFrame, arquivos: dict, comum: bool = True) -> tuple[dict, int]:
    """Séries prontas para CSD/RQA a partir da tabela_fc.csv do notebook 01:
    evento cortado (n_cortado), ectópicos interpolados, evento e controle com o mesmo
    comprimento L (últimos L batimentos). Retorna ({evento: (rr_ev, rr_ct)}, L)."""
    series = {}
    for _, r in fc_df.iterrows():
        ev = ler_rr(arquivos[r["evento"]])
        ev = ev[:len(ev) - int(r["n_cortado"])]
        ct = ler_rr(arquivos[r["controle"]])
        series[r["evento"]] = (interpolar_ectopicos(ev)[0], interpolar_ectopicos(ct)[0])
    L = min(min(len(e), len(c)) for e, c in series.values())
    if comum:
        series = {k: (e[-L:], c[-L:]) for k, (e, c) in series.items()}
    return series, L


# ---------------------------------------------------------------------------
# 3c. Surrogados (etapa 8)
# ---------------------------------------------------------------------------
def surrogado_fase(x: np.ndarray, rng) -> np.ndarray:
    """Surrogado por randomização de fase: mesmo espectro (autocorrelação linear),
    estacionário. Distribuição de amplitudes tende à gaussiana."""
    x = np.asarray(x, float)
    n = len(x)
    F = np.fft.rfft(x - x.mean())
    fases = rng.uniform(0, 2 * np.pi, len(F))
    fases[0] = 0
    if n % 2 == 0:
        fases[-1] = 0
    return np.fft.irfft(np.abs(F) * np.exp(1j * fases), n) + x.mean()


def surrogado_iaaft(x: np.ndarray, rng, n_iter: int = 30) -> np.ndarray:
    """IAAFT (Schreiber & Schmitz, 1996): preserva o espectro E a distribuição de
    amplitudes da série original, mas destrói qualquer tendência não estacionária
    (é a hipótese nula de 'processo linear estacionário')."""
    x = np.asarray(x, float)
    n = len(x)
    ordenado = np.sort(x)
    amp = np.abs(np.fft.rfft(x))
    s = rng.permutation(x)
    for _ in range(n_iter):
        S = np.fft.rfft(s)
        s = np.fft.irfft(amp * np.exp(1j * np.angle(S)), n)
        s = ordenado[np.argsort(np.argsort(s))]
    return s


def teste_surrogados(rr: np.ndarray, n_surr: int = 200, metodo_surr: str = "iaaft",
                     w: float = 0.50, bw: float = 0.10, detrend: str = "gauss",
                     seed: int = 0) -> dict:
    """Para uma série: tau observado de cada indicador e p-valor UNILATERAL contra
    `n_surr` surrogados (H1: tendência de subida maior que a de um processo estacionário
    com o mesmo espectro). p = (1 + #{tau_surr >= tau_obs}) / (n_surr + 1)."""
    rng = np.random.default_rng(seed)
    obs = csd_serie(rr, w, bw, detrend)
    gerar = surrogado_iaaft if metodo_surr == "iaaft" else surrogado_fase
    nulos = {k: [] for k in obs}
    for _ in range(n_surr):
        t = csd_serie(gerar(rr, rng), w, bw, detrend)
        for k in obs:
            nulos[k].append(t[k])
    out = {}
    for k, v in obs.items():
        nul = np.asarray(nulos[k], float)
        nul = nul[np.isfinite(nul)]
        out[k] = v
        out[f"p_{k[4:]}"] = (1 + np.sum(nul >= v)) / (len(nul) + 1) if np.isfinite(v) else np.nan
    return out


def mcnemar_exato(sig_ev, sig_ct) -> dict:
    """Teste de McNemar exato (binomial nos pares discordantes) para proporções pareadas.
    n10 = só o evento significativo; n01 = só o controle significativo."""
    from scipy.stats import binomtest
    a, b = np.asarray(sig_ev, bool), np.asarray(sig_ct, bool)
    n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    p = binomtest(n10, n10 + n01, 0.5).pvalue if (n10 + n01) > 0 else 1.0
    return {"n": len(a), "prop_ev": a.mean(), "prop_ct": b.mean(),
            "so_ev": n10, "so_ct": n01, "p_mcnemar": float(p)}


# ---------------------------------------------------------------------------
# 3d. Quantização de 10 ms (etapa 9)
# ---------------------------------------------------------------------------
def truncar(x: np.ndarray, passo: float = 10.0) -> np.ndarray:
    """Truncamento (não arredondamento) para múltiplos de `passo`, como na MVTDB."""
    return np.floor(np.asarray(x, float) / passo) * passo


def dither(x: np.ndarray, rng, passo: float = 10.0) -> np.ndarray:
    """Desfaz o truncamento de forma aleatória: o valor real está em [x, x + passo),
    então soma U(0, passo). Repetir várias vezes mostra a incerteza causada pela quantização."""
    return np.asarray(x, float) + rng.uniform(0, passo, len(x))


def simular_ar1(n: int, phi: float, sd: float, media: float, rng) -> np.ndarray:
    """Processo AR(1) estacionário com desvio padrão marginal `sd` e média `media`."""
    e = rng.normal(0, sd * np.sqrt(1 - phi ** 2), n)
    x = np.empty(n)
    x[0] = rng.normal(0, sd)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x + media


# ---------------------------------------------------------------------------
# 3e. Takens + RQA (etapas 10–11) — MESMAS definições do artigo (rqa_vf_analysis.ipynb)
# ---------------------------------------------------------------------------
METRICAS_RQA = ['RR', 'DET', 'L', 'L_max', 'DIV', 'L_entr', 'LAM', 'TT', 'V_max',
                'V_entr', 'W', 'W_max', 'W_entr', 'DET_RR', 'LAM_DET']


def embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Matriz de embedding (N x m) por atrasos: linha i = [x_i, x_{i+tau}, ...]."""
    x = np.asarray(x, dtype=float)
    N = len(x) - (m - 1) * tau
    if N <= 0:
        return np.empty((0, m))
    idx = np.arange(N)[:, None] + np.arange(m)[None, :] * tau
    return x[idx]


def info_mutua(x: np.ndarray, max_lag: int = 40, bins: int = 16) -> np.ndarray:
    """Informação mútua média I(tau) entre x[t] e x[t+tau] (histograma 2D)."""
    x = np.asarray(x, dtype=float)
    vals = []
    for lag in range(1, max_lag + 1):
        c, _, _ = np.histogram2d(x[:-lag], x[lag:], bins=bins)
        pab = c / c.sum()
        prod = pab.sum(axis=1, keepdims=True) @ pab.sum(axis=0, keepdims=True)
        nz = (pab > 0) & (prod > 0)
        vals.append(float(np.sum(pab[nz] * np.log(pab[nz] / prod[nz]))))
    return np.array(vals)


def tau_por_info_mutua(x: np.ndarray, max_lag: int = 40, bins: int = 16) -> tuple[int, np.ndarray]:
    """tau = primeiro mínimo local da informação mútua (Fraser & Swinney, 1986)."""
    im = info_mutua(x, max_lag=max_lag, bins=bins)
    for i in range(1, len(im) - 1):
        if im[i] < im[i - 1] and im[i] <= im[i + 1]:
            return i + 1, im
    return int(np.argmin(im)) + 1, im


def m_por_fnn(x: np.ndarray, tau: int, m_max: int = 10, Rtol: float = 15.0,
              Atol: float = 2.0, limiar: float = 0.01) -> tuple[int, np.ndarray]:
    """m = menor dimensão com fração de falsos vizinhos < limiar (Kennel et al., 1992)."""
    from scipy.spatial import cKDTree
    x = np.asarray(x, dtype=float)
    sigma = x.std()
    fracoes = []
    for m in range(1, m_max + 1):
        Y = embed(x, m, tau)
        N = len(Y) - tau
        if N < 20:
            break
        d, viz = cKDTree(Y[:N]).query(Y[:N], k=2)
        d1, viz = d[:, 1], viz[:, 1]
        delta = np.abs(x[np.arange(N) + m * tau] - x[viz + m * tau])
        with np.errstate(divide='ignore', invalid='ignore'):
            c1 = np.where(d1 > 0, delta / d1, np.inf) > Rtol
        c2 = (np.sqrt(d1 ** 2 + delta ** 2) / sigma) > Atol if sigma > 0 else np.zeros(N, bool)
        f = float(np.mean(c1 | c2))
        fracoes.append(f)
        if f < limiar:
            return m, np.array(fracoes)
    return (int(np.argmin(fracoes)) + 1 if fracoes else 2), np.array(fracoes)


def _runs(arr, valor):
    mask = (np.asarray(arr) == valor).astype(np.int8)
    if mask.size == 0:
        return np.empty(0, dtype=int)
    d = np.diff(np.concatenate(([0], mask, [0])))
    return np.where(d == -1)[0] - np.where(d == 1)[0]


def _entropia(lengths):
    if len(lengths) == 0:
        return 0.0
    _, counts = np.unique(lengths, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


def calcular_rqa(rp: np.ndarray, l_min: int = 2, v_min: int = 2) -> dict:
    """As 15 métricas do RQA a partir da matriz binária rp (N x N) — idêntico ao artigo."""
    N = rp.shape[0]
    rp = rp.astype(np.int8)
    RR = rp.sum() / (N * N)
    dl = [_runs(np.diagonal(rp, k), 1) for k in range(1, N)]
    dl = np.concatenate(dl) if dl else np.empty(0, dtype=int)
    tot_d = dl.sum()
    dge = dl[dl >= l_min]
    DET = (dge.sum() / tot_d * 100) if tot_d > 0 else 0.0
    L = dge.mean() if len(dge) else 0.0
    L_max = int(dl.max()) if len(dl) else 0
    DIV = (1.0 / L_max) if L_max > 0 else 0.0
    L_entr = _entropia(dge)
    vl, wl = [], []
    for j in range(N):
        col = rp[:, j]
        vl.append(_runs(col, 1))
        wl.append(_runs(col, 0))
    vl, wl = np.concatenate(vl), np.concatenate(wl)
    tot_v = vl.sum()
    vge = vl[vl >= v_min]
    LAM = (vge.sum() / tot_v * 100) if tot_v > 0 else 0.0
    TT = vge.mean() if len(vge) else 0.0
    V_max = int(vl.max()) if len(vl) else 0
    V_entr = _entropia(vge)
    wge = wl[wl >= v_min]
    W = wge.mean() if len(wge) else 0.0
    W_max = int(wl.max()) if len(wl) else 0
    W_entr = _entropia(wge)
    DET_RR = (DET / 100) / RR if RR > 0 else 0.0
    LAM_DET = (LAM / 100) / (DET / 100) if DET > 0 else 0.0
    return {'RR': RR, 'DET': DET, 'L': L, 'L_max': L_max, 'DIV': DIV, 'L_entr': L_entr,
            'LAM': LAM, 'TT': TT, 'V_max': V_max, 'V_entr': V_entr, 'W': W, 'W_max': W_max,
            'W_entr': W_entr, 'DET_RR': DET_RR, 'LAM_DET': LAM_DET}


def matriz_recorrencia(x: np.ndarray, m: int, tau: int, percentil: float = 10) -> np.ndarray | None:
    """Matriz de recorrência binária: embedding de Takens, distância euclidiana e limiar
    = percentil das distâncias POSITIVAS (o mesmo critério do artigo)."""
    from scipy.spatial.distance import pdist, squareform
    Y = embed(x, m, tau)
    if len(Y) < 10:
        return None
    d = pdist(Y)
    pos = d[d > 0]
    if len(pos) == 0:
        return None
    eps = np.percentile(pos, percentil)
    return (squareform(d) <= eps).astype(np.int8)


def rqa_janelas(rr: np.ndarray, m: int, tau: int, janela: int = 256, passo: int = 32,
                percentil: float = 10) -> pd.DataFrame:
    """RQA em janela deslizante de `janela` batimentos (passo `passo`), uma linha por
    janela, indexada pelo batimento final; inclui NN médio, SDNN e RMSSD da janela."""
    rr = np.asarray(rr, float)
    linhas = []
    for ini in range(0, len(rr) - janela + 1, passo):
        x = rr[ini:ini + janela]
        reg = {'ini': ini, 'fim': ini + janela, 'NN_medio': x.mean(),
               'SDNN': x.std(ddof=1), 'RMSSD': np.sqrt(np.mean(np.diff(x) ** 2))}
        rp = matriz_recorrencia(x, m, tau, percentil)
        if rp is not None:
            reg.update(calcular_rqa(rp))
        linhas.append(reg)
    # sempre as mesmas colunas (NaN quando a janela não comporta o embedding)
    return pd.DataFrame(linhas, columns=['ini', 'fim', 'NN_medio', 'SDNN', 'RMSSD'] + METRICAS_RQA)


# ---------------------------------------------------------------------------
# 4. Estatística pareada
# ---------------------------------------------------------------------------
def dz_pareado(pre, ctrl, n_boot: int = 10000, seed: int = 0) -> dict:
    """Cohen's d pareado (d_z = média(dif)/dp(dif)), dif = pre - ctrl, com IC 95%
    por bootstrap percentil reamostrando PACIENTES. Também retorna Wilcoxon pareado."""
    from scipy.stats import wilcoxon
    pre, ctrl = np.asarray(pre, float), np.asarray(ctrl, float)
    m = ~(np.isnan(pre) | np.isnan(ctrl))
    d = pre[m] - ctrl[m]
    n = len(d)
    if n < 3 or d.std(ddof=1) == 0:
        return {"n": n, "dz": np.nan, "ic_lo": np.nan, "ic_hi": np.nan, "p_wilcoxon": np.nan}
    dz = d.mean() / d.std(ddof=1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    amostras = d[idx]
    sd = amostras.std(axis=1, ddof=1)
    boots = np.divide(amostras.mean(axis=1), sd, out=np.full(n_boot, np.nan), where=sd > 0)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    p = wilcoxon(pre[m], ctrl[m]).pvalue
    return {"n": n, "dz": float(dz), "ic_lo": float(lo), "ic_hi": float(hi),
            "p_wilcoxon": float(p), "media_dif": float(d.mean())}


# ---------------------------------------------------------------------------
# 5. Estilo das figuras (português + negrito)
# ---------------------------------------------------------------------------
def estilo_figuras(base: int = 13, virgula: bool = True):
    """Padrão das figuras do TCC: textos em português, negrito, 300 dpi.
    virgula=True: números dos eixos com vírgula decimal (0,5 em vez de 0.5). Isso é feito
    trocando o ponto pela vírgula no formatador numérico padrão do matplotlib (ScalarFormatter),
    então vale para todos os eixos e barras de cor da sessão; rótulos escritos à mão usam num()/ic()."""
    import matplotlib as mpl
    from matplotlib import ticker
    if virgula and not getattr(ticker.ScalarFormatter, "_virgula_tcc", False):
        _orig = ticker.ScalarFormatter.__call__
        def _com_virgula(self, x, pos=None):
            return _orig(self, x, pos).replace(".", ",")
        ticker.ScalarFormatter.__call__ = _com_virgula
        ticker.ScalarFormatter._virgula_tcc = True
    mpl.rcParams.update({
        "font.size": base, "font.weight": "bold",
        "axes.titlesize": base + 2, "axes.titleweight": "bold",
        "axes.labelsize": base + 1, "axes.labelweight": "bold",
        "xtick.labelsize": base - 1, "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1, "figure.titlesize": base + 3,
        "figure.titleweight": "bold", "axes.spines.top": False,
        "axes.spines.right": False, "savefig.dpi": 300,
        "savefig.bbox": "tight", "figure.dpi": 110,
    })


def num(v, casas: int = 2) -> str:
    """Número com vírgula decimal para títulos e anotações das figuras (0,78)."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:.{casas}f}".replace(".", ",").replace("-", "−")


def ic(lo, hi, casas: int = 2) -> str:
    """Intervalo de confiança no padrão do texto: [0,54; 1,09]."""
    return f"[{num(lo, casas)}; {num(hi, casas)}]"


def rotulo_parametros(txt: str) -> str:
    """'w=0.5, bw=0.1' -> 'janela 50 %, gaussiano 10 %'; 'w=0.25, linear' -> 'janela 25 %, linear'."""
    m = re.match(r"w=([\d.]+),\s*(?:bw=([\d.]+)|(\w+))", str(txt))
    if not m:
        return str(txt)
    j = f"janela {round(100 * float(m.group(1)))} %"
    return f"{j}, gaussiano {round(100 * float(m.group(2)))} %" if m.group(2) else f"{j}, {m.group(3)}"


def negrito_ticks(ax):
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")


def holm(pvals) -> np.ndarray:
    """Correção de Holm–Bonferroni para várias comparações (retorna p ajustados)."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    out = np.full_like(p, np.nan)
    idx = np.where(ok)[0][np.argsort(p[ok])]
    m = len(idx)
    acum = 0.0
    for r, i in enumerate(idx):
        acum = max(acum, min(1.0, (m - r) * p[i]))
        out[i] = acum
    return out


# ---------------------------------------------------------------------------
# 6. SDDB — horizonte longo e pseudo-eventos (etapas 17–18)
#    Metadados e desenho do controle: hrv_sddb.py (material do orientador)
# ---------------------------------------------------------------------------
SDDB_META = pd.DataFrame(
    [
        # rec, rhythm, paced, atr,  dur_s, vf_onset_s, sex, age
        ("30", "sinus", False, True,  88397, 28473, "M", 43),
        ("31", "sinus", False, True,  50320, 49344, "F", 72),
        ("32", "sinus", True,  True,  87600, 60318, "?", 62),
        ("33", "sinus", False, False, 88380, 17179, "F", 30),
        ("34", "sinus", False, True,  25520, 23744, "M", 34),
        ("35", "afib",  False, True,  89520, 88496, "F", 72),
        ("36", "afib",  False, True,  73280, 68341, "M", 75),
        ("37", "afib",  False, False, 90480,  5473, "F", 89),
        ("38", "sinus", False, False, 65905, 28914, "?", None),
        ("39", "sinus", False, False, 20820, 16671, "M", 66),
        ("40", "paced", True,  False, 89580,  None, "M", 79),
        ("41", "sinus", False, True,  14160, 10764, "M", None),
        ("42", "sinus", False, False, 90490,  None, "M", 17),
        ("43", "sinus", True,  False, 83270, 56231, "M", 35),
        ("44", "sinus", False, False, 84000, 70725, "M", None),
        ("45", "sinus", False, True,  86960, 65357, "M", 68),
        ("46", "sinus", False, True,  15310, 13307, "F", None),
        ("47", "sinus", False, False, 84950, 22381, "M", 34),
        ("48", "sinus", False, False, 88575,  8980, "M", 80),
        ("49", "sinus", True,  True,  90100,  None, "M", 73),
        ("50", "afib",  False, False, 83258, 42343, "F", 68),
        ("51", "sinus", True,  True,  90510, 82703, "F", 67),
        ("52", "sinus", False, True,  27065,  9160, "F", 82),
    ],
    columns=["record", "rhythm", "paced", "has_atr", "dur_s", "vf_onset_s", "sex", "age"],
).set_index("record")

SIMB_BATIMENTO = set('NLRejAaJSVEFP/fQ?')
SIMB_SINUSAL = set('NLRej')


def _com_retentativas(func, tentativas: int = 5, espera: float = 3.0, rotulo: str = ""):
    """Chama func() de novo quando o PhysioNet falha de forma passageira (NetFileError, timeout, 429/503).
    Espera 3, 6, 12, 24 s entre as tentativas. Na última falha, repassa o erro."""
    import time
    for k in range(tentativas):
        try:
            return func()
        except Exception as e:
            if k == tentativas - 1:
                raise
            print(f'  {rotulo}: {type(e).__name__}, tentando de novo em {espera * 2 ** k:.0f} s '
                  f'({k + 2}/{tentativas})')
            time.sleep(espera * 2 ** k)


def baixar_wfdb(base: str, registros: list, pasta: str, exts=('hea', 'atr'), tentativas: int = 5) -> list:
    """Baixa só os arquivos pedidos (por padrão .hea e .atr, sem o sinal .dat) do PhysioNet.
    Pula o que já existe e tenta de novo quando o servidor falha (ele limita pedidos seguidos).
    Retorna os registros que ficaram com todos os arquivos."""
    import wfdb
    os.makedirs(pasta, exist_ok=True)
    ok = []
    for r in registros:
        faltam = [f'{r}.{e}' for e in exts if not os.path.exists(os.path.join(pasta, f'{r}.{e}'))]
        if faltam:
            try:
                _com_retentativas(lambda: wfdb.dl_files(base, pasta, faltam), tentativas, rotulo=f'{base}/{r}')
            except Exception as e:
                print(f'  {base}/{r}: falha ao baixar {faltam} depois de {tentativas} tentativas ({type(e).__name__})')
        if all(os.path.exists(os.path.join(pasta, f'{r}.{e}')) for e in exts):
            ok.append(r)
    return ok


def info_header(caminho_base: str) -> dict:
    """fs, duração (s) e vfon (s, do comentário '#vfon: HH:MM:SS', se houver) a partir do .hea."""
    import wfdb
    h = wfdb.rdheader(caminho_base)
    vf = None
    for c in (h.comments or []):
        m = re.search(r"vfon:\s*(\d+):(\d{2}):(\d{2})", c, re.IGNORECASE)
        if m:
            vf = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
    return {'fs': float(h.fs), 'dur_s': h.sig_len / float(h.fs) if h.sig_len else np.nan, 'vfon_s': vf}


def serie_rr_anotada(caminho_base: str, fs: float, guard: int = 1, lo: float = 300, hi: float = 2000):
    """Intervalos entre batimentos anotados (.atr) com máscara de validade NN:
    válido só se os DOIS batimentos são sinusais, nenhum está a menos de `guard`
    batimentos de um ectópico e o intervalo está em [lo, hi] ms (mesma regra do artigo).
    Retorna (t_s do 2º batimento, rr_ms, valido)."""
    import wfdb
    ann = wfdb.rdann(caminho_base, 'atr')
    simb, amos = np.asarray(ann.symbol), np.asarray(ann.sample)
    bat = np.array([s in SIMB_BATIMENTO for s in simb])
    simb, amos = simb[bat], amos[bat]
    if len(amos) < 3:
        return np.array([]), np.array([]), np.array([], bool)
    ok = np.array([s in SIMB_SINUSAL for s in simb])
    if guard > 0:
        mau = ~ok
        bloq = mau.copy()
        for g in range(1, guard + 1):
            bloq[g:] |= mau[:-g]
            bloq[:-g] |= mau[g:]
        ok = ok & ~bloq
    t = amos / fs
    rr = np.diff(t) * 1000.0
    valido = ok[:-1] & ok[1:] & (rr >= lo) & (rr <= hi)
    return t[1:], rr, valido


def indices_janelas(t: np.ndarray, rr: np.ndarray, valido: np.ndarray, t_evento: float,
                    horas_antes: float = 24, win_s: float = 300, step_s: float = 300,
                    gate: float = 0.80) -> pd.DataFrame:
    """Janelas de `win_s` antes do evento (ou pseudo-evento), com portão de qualidade
    (fração de intervalos NN válidos >= gate, como no artigo). Índices nas janelas aprovadas,
    com os inválidos interpolados: FC média, SDNN, RMSSD e AR(1) do NN sem tendência linear.
    Janelas reprovadas ficam na tabela com passed=False (a taxa de reprovação é informação)."""
    from scipy.signal import detrend
    ini = max(t.min() if len(t) else 0, t_evento - horas_antes * 3600)
    linhas = []
    for s in np.arange(t_evento - win_s, ini - 1e-9, -step_s)[::-1]:
        e = s + win_s
        m = (t >= s) & (t < e)
        n_tot = int(m.sum())
        reg = {'t_inicio_s': s, 'min_ate_evento': (e - t_evento) / 60.0, 'n_total': n_tot,
               'frac_valida': np.nan, 'passed': False}
        if n_tot >= 30:
            v = valido[m]
            reg['frac_valida'] = float(v.mean())
            if v.mean() >= gate and v.sum() >= 30:
                x = rr[m].copy()
                idx = np.arange(len(x))
                x[~v] = np.interp(idx[~v], idx[v], x[v])
                r = detrend(x, type='linear')
                reg.update({'passed': True, 'FC': 60000.0 / x.mean(), 'SDNN': x.std(ddof=1),
                            'RMSSD': float(np.sqrt(np.mean(np.diff(x) ** 2))),
                            'AR1': float(np.corrcoef(r[:-1], r[1:])[0, 1])})
        linhas.append(reg)
    return pd.DataFrame(linhas)


def pseudo_eventos(registros: list, duracoes_s: dict, pool_s: list, margem_s: float = 3600,
                   seed: int = 0) -> dict:
    """Pseudo-evento para cada registro controle, sorteado do POOL de instantes reais de FV
    da SDDB (tempo decorrido desde o início do Holter), como em hrv_sddb.py. Só aceita
    instantes que caibam no registro com `margem_s` de folga; se nenhum couber, usa a mesma
    FRAÇÃO da duração."""
    rng = np.random.default_rng(seed)
    pool = [float(p) for p in pool_s if p is not None and np.isfinite(p)]
    out = {}
    for r in registros:
        dur = duracoes_s.get(r)
        if not dur or not np.isfinite(dur):
            continue
        cand = [p for p in pool if margem_s <= p <= dur - 60]
        if cand:
            out[r] = float(rng.choice(cand))
        else:
            out[r] = float(rng.choice([p / max(pool) for p in pool]) * dur)
    return out


def hedges_g_boot(a, b, n_boot: int = 10000, seed: int = 0) -> dict:
    """Hedges g (a − b, grupos independentes) com IC 95 % por bootstrap e Mann–Whitney."""
    from scipy.stats import mannwhitneyu
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return {'n_a': len(a), 'n_b': len(b), 'g': np.nan, 'ic_lo': np.nan, 'ic_hi': np.nan, 'p_mw': np.nan}
    def g_(x, y):
        sp = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
        j = 1 - 3 / (4 * (len(x) + len(y)) - 9)
        return j * (x.mean() - y.mean()) / sp if sp > 0 else np.nan
    rng = np.random.default_rng(seed)
    boots = [g_(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(n_boot)]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {'n_a': len(a), 'n_b': len(b), 'g': g_(a, b), 'ic_lo': lo, 'ic_hi': hi,
            'p_mw': float(mannwhitneyu(a, b).pvalue), 'media_a': a.mean(), 'media_b': b.mean()}


# ---------------------------------------------------------------------------
# 7. Estudo de limiares do corte (notebook 09)
# ---------------------------------------------------------------------------
CENARIOS_TV = ("tv_rapida", "tv_lenta", "aceleracao_gradual", "inicio_gradual", "ectopicos_antes")


def serie_com_tv(fundo: np.ndarray, rng, cenario: str = "tv_rapida", n_tv=None,
                 n_total: int = 1024) -> tuple:
    """Série sintética com início de TV CONHECIDO: ritmo sinusal real (`fundo`, uma série
    controle da MVTDB) seguido de um trecho de TV artificial. Retorna (rr, t0, info), com
    t0 = índice do primeiro intervalo de TV (o corte ideal é k <= t0, o mais perto possível).
    Cenários:
      tv_rapida          TV com RR = 45–65 % do ritmo anterior (degrau nítido)
      tv_lenta           TV com RR = 72–85 % (queda menor que os 25 % do detector)
      aceleracao_gradual taquicardia sinusal: os últimos 150 batimentos sinusais aceleram
                         até 85 % do RR; depois TV rápida (o detector NÃO deve cortar a rampa)
      inicio_gradual     TV rápida que se instala em 6 batimentos (transição conta como TV)
      ectopicos_antes    TV rápida com extrassístoles (curto-longo) nos 80 batimentos antes
    O RR da TV tem ruído com desvio padrão sorteado entre 5 e 30 ms e é truncado em 10 ms,
    como na MVTDB.
    `n_tv`: batimentos de TV antes da detecção (padrão: sorteado entre 16 e 40)."""
    fundo = np.asarray(fundo, float)
    if n_tv is None:
        n_tv = int(rng.integers(16, 41))
    n_sin = min(len(fundo), n_total - n_tv)
    sin = fundo[-n_sin:].copy()
    info = {"cenario": cenario, "n_tv": n_tv}
    if cenario == "aceleracao_gradual":
        m = min(150, n_sin)
        sin[-m:] = sin[-m:] * np.linspace(1.0, 0.85, m)
    if cenario == "ectopicos_antes":
        zona = np.arange(max(1, n_sin - 80), n_sin - 1)
        pos = zona[rng.random(len(zona)) < 0.05]
        for p in pos:
            ref = np.median(sin[max(0, p - 10):p]) if p > 0 else sin[p]
            sin[p] = 0.65 * ref
            sin[p + 1] = 1.35 * ref
        info["n_ectopicos"] = len(pos)
    nivel = np.median(sin[-30:])
    lo, hi = (0.72, 0.85) if cenario == "tv_lenta" else (0.45, 0.65)
    razao = float(rng.uniform(lo, hi))
    sd_tv = float(rng.uniform(5, 30))          # TV monomórfica regular (~5 ms) a irregular (~30 ms)
    tv = nivel * razao + rng.normal(0, sd_tv, n_tv)
    if cenario == "inicio_gradual":
        nt = min(6, n_tv)
        tv[:nt] = nivel * np.linspace(1.0, razao, nt + 1)[1:] + rng.normal(0, sd_tv, nt)
    tv = np.floor(np.clip(tv, 200, None) / 10) * 10
    info["razao_tv"] = razao
    info["sd_tv"] = sd_tv
    rr = np.r_[sin, tv]
    return rr, len(sin), info


def avaliar_corte(k: int, t0: int) -> dict:
    """Erro do corte k contra o início verdadeiro t0:
    contaminacao = intervalos de TV que ficaram na série (k > t0) — o erro grave;
    perda        = intervalos sinusais removidos sem necessidade (k < t0) — o custo."""
    return {"contaminacao": max(0, k - t0), "perda": max(0, t0 - k)}


# ---------------------------------------------------------------------------
# 8. Detecção de QRS pela matriz de distância (método de Charron, TCC UFPE)
#    Porte do repositório github.com/Ni-cole17/PTB-Diagnostic-Visualization
#    (segment.ipynb, seção "Distância R-R"; abordagem de LIMIARIZAÇÃO).
#    Adaptações para Holter longo (SDDB/NSRDB) estão marcadas com [ADAPTAÇÃO].
# ---------------------------------------------------------------------------
SDDB_SEM_ATR_ELEGIVEIS = ("33", "38", "39", "44", "47", "48")   # sinusal, sem marcapasso, com vfon, SEM .atr


def filtro_passa_banda(sig: np.ndarray, fs: float, fl: float = 1.0, fh: float = 50.0, ordem: int = 3) -> np.ndarray:
    """Butterworth passa-banda de ordem 3, 1–50 Hz, ida e volta (filtfilt) — igual ao da Nicole.
    [ADAPTAÇÃO] fh é limitado a 0,45·fs (na NSRDB, 128 Hz, o Nyquist é 64 Hz)."""
    from scipy.signal import butter, filtfilt
    fh = min(fh, 0.45 * fs)
    b, a = butter(ordem, [fl / (fs / 2), fh / (fs / 2)], btype="band")
    return filtfilt(b, a, np.asarray(sig, float), axis=0)


def matriz_distancia(x: np.ndarray) -> np.ndarray:
    """D[i, j] = distância euclidiana entre as amostras i e j do sinal (todas as derivações juntas).
    É o gráfico de recorrência SEM limiar com dimensão de imersão = nº de derivações.
    (Mesmo resultado do generate_image da Nicole; calculado derivação a derivação, em float32, ~4x mais rápido.)"""
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        x = x[:, None]
    acc = np.zeros((len(x), len(x)), np.float32)
    for c in range(x.shape[1]):
        d = x[:, None, c] - x[None, :, c]
        acc += d * d
    return np.sqrt(acc, out=acc)


def _quantil_rapido(D: np.ndarray, q: float) -> float:
    """Igual a np.quantile(D, q) (interpolação linear), via np.partition — ~8x mais rápido."""
    v = D.ravel()
    pos = q * (v.size - 1)
    k = int(np.floor(pos))
    k2 = min(k + 1, v.size - 1)
    p = np.partition(v, [k, k2])
    return float(p[k] + (pos - k) * (p[k2] - p[k]))


def _dbscan_1d(pontos: np.ndarray, eps: float, min_amostras: int = 2) -> list:
    """DBSCAN em 1 dimensão sem sklearn (para min_samples 1 ou 2): em 1D, com min_samples = 2, todo ponto que tem
    um vizinho a <= eps é núcleo e os grupos são as sequências de pontos ordenados com saltos <= eps; ponto
    sozinho é ruído (com min_samples = 1 ele vira um grupo de 1). Dá exatamente os mesmos grupos do sklearn (testado em tcc_lib_testes) e é ~25x mais rápido."""
    p = np.sort(np.asarray(pontos, float))
    if len(p) == 0:
        return []
    if min_amostras > 2:
        from sklearn.cluster import DBSCAN
        rot = DBSCAN(eps=eps, min_samples=min_amostras).fit_predict(p.reshape(-1, 1))
        return [p[rot == k] for k in np.unique(rot) if k != -1]
    corte = np.flatnonzero(np.diff(p) > eps) + 1
    return [g for g in np.split(p, corte) if len(g) >= min_amostras]


def qrs_janela_matriz(x: np.ndarray, fs: float, q: float = 0.95, frac: float = 0.75,
                      eps_s: float = 0.100, min_s: float = 0.002, devolver_tudo: bool = False):
    """Um trecho curto (Nicole: 2 s a 1000 Hz) -> posições dos QRS (amostras dentro do trecho).
    1. matriz de distância D;
    2. binariza: branco onde D >= percentil q de D (Nicole: 95 %);
    3. soma de cada coluna da imagem binária (= quantas amostras estão "muito longe" desta);
    4. colunas com soma > frac · máximo (Nicole: 75 %);
    5. DBSCAN nas colunas escolhidas (eps = 100 amostras a 1000 Hz = 100 ms; min_samples = 2) — ver _dbscan_1d;
    6. centroide de cada grupo = posição do QRS.
    [ADAPTAÇÃO] eps e min_samples em SEGUNDOS, para valerem o mesmo tempo a 250 e 128 Hz:
    eps = 100 ms (eps_s·fs amostras) e min_samples = 2 amostras a 1000 Hz = 2 ms (min_s) -> 1 amostra a 250 Hz.
    Sem isso, a 250 Hz um QRS estreito pode ter só 1 coluna acima do limiar e o DBSCAN o descartaria como ruído."""
    min_amostras = max(1, int(round(min_s * fs)))
    D = matriz_distancia(x)
    limiar_d = _quantil_rapido(D, q)
    vazio = (np.array([], float), None) if not devolver_tudo else (np.array([], float), {})
    if not np.isfinite(limiar_d) or limiar_d <= 1e-9:
        return vazio
    binaria = D >= limiar_d
    soma = binaria.sum(axis=0).astype(float)
    if soma.max() <= 0:
        return vazio
    limiar_s = frac * soma.max()
    cols = np.flatnonzero(soma > limiar_s)
    if len(cols) < min_amostras:
        return vazio
    grupos = _dbscan_1d(cols, max(1.0, eps_s * fs), min_amostras)
    cent = np.array(sorted(g.mean() for g in grupos))
    if devolver_tudo:
        return cent, {"D": D, "limiar_d": limiar_d, "binaria": binaria, "soma": soma,
                      "limiar_soma": limiar_s, "colunas": cols, "grupos": grupos}
    return cent, None


def refinar_pico(sig_filt: np.ndarray, idx, fs: float, raio_s: float = 0.06) -> np.ndarray:
    """[ADAPTAÇÃO] O centroide marca o MEIO da faixa do QRS, não o pico R; para RR com precisão de ms,
    move cada marca para o máximo da norma das derivações filtradas a até ±raio_s. Usa o mesmo
    critério de amplitude da matriz (a coluna mais longe de todas é a de maior norma)."""
    x = np.asarray(sig_filt, float)
    nrm = np.sqrt((x ** 2).sum(axis=1)) if x.ndim == 2 else np.abs(x)
    r = max(1, int(round(raio_s * fs)))
    out = []
    for i in np.asarray(idx, float):
        a, b = max(0, int(i) - r), min(len(nrm), int(i) + r + 1)
        out.append(a + int(np.argmax(nrm[a:b])) if b > a else int(i))
    return np.unique(np.array(out, int))


def detectar_qrs_matriz(sig: np.ndarray, fs: float, ini: int = 0, fim=None, win_s: float = 4.0,
                        passo_s: float = 3.0, refinar: bool = True, mostrar_progresso: bool = False, **kw) -> dict:
    """[ADAPTAÇÃO] Aplica qrs_janela_matriz em janelas deslizantes (padrão 4 s, passo 3 s) a um registro longo.
    Cada janela só "vale" no seu miolo (as 0,5 s das bordas ficam para a vizinha), o que evita marcas
    duplicadas e QRS cortados na borda. Marcas a menos de 150 ms são fundidas.
    Devolve {'centroide': amostras, 'pico': amostras refinadas (ou None), 'n_janelas': int}."""
    sig = np.asarray(sig, float)
    if sig.ndim == 1:
        sig = sig[:, None]
    fim = len(sig) if fim is None else min(int(fim), len(sig))
    pad = int(2 * fs)                                     # filtra com folga para não ter efeito de borda
    a0, b0 = max(0, ini - pad), min(len(sig), fim + pad)
    filt = filtro_passa_banda(sig[a0:b0], fs)
    W, P = int(round(win_s * fs)), int(round(passo_s * fs))
    borda = (W - P) // 2
    marcas, n_jan = [], 0
    inicios = range(ini - a0, max(ini - a0 + 1, fim - a0 - W + 1), P)
    if mostrar_progresso:
        try:
            from tqdm.auto import tqdm
            inicios = tqdm(inicios, leave=False)
        except ImportError:
            pass
    for s in inicios:
        trecho = filt[s:s + W]
        if len(trecho) < W // 2:
            break
        cent, _ = qrs_janela_matriz(trecho, fs, **kw)
        n_jan += 1
        lo = 0 if s == ini - a0 else borda
        hi = len(trecho) if s + W >= fim - a0 else W - borda
        marcas.extend(s + c for c in cent if lo <= c < hi)
    marcas = np.array(sorted(marcas))
    if len(marcas):
        grupos = np.r_[0, np.cumsum(np.diff(marcas) > 0.150 * fs)]
        marcas = np.array([marcas[grupos == g].mean() for g in np.unique(grupos)])
    cent = (marcas + a0).round().astype(int)
    pico = (refinar_pico(filt, marcas.round().astype(int), fs) + a0) if refinar and len(marcas) else None
    return {"centroide": cent, "pico": pico, "n_janelas": n_jan}


def detectar_qrs_matriz_movel(sig: np.ndarray, fs: float, ini: int = 0, fim=None, win_s: float = 4.0,
                              passo_s: float = 3.0, q: float = 0.95, frac: float = 0.50, viz: int = 15,
                              eps_s: float = 0.100, min_s: float = 0.002, refinar: bool = False) -> dict:
    """Variante da Nicole com REFERÊNCIA MÓVEL (correção proposta no TCC para o Holter).
    Diagnóstico que a motivou (SDDB, última hora antes da FV): 55 % dos batimentos perdidos estavam em
    janelas cujo máximo era uma extrassístole ventricular — o percentil 95 e o limiar de 75 % são relativos à
    PRÓPRIA janela, então um batimento gigante apaga os normais da mesma janela.
    O que muda (o resto é igual: matriz, binarização, soma das colunas, DBSCAN, centroide):
      1. o limiar de distância da binarização deixa de ser o percentil q da janela e passa a ser a MEDIANA
         desse percentil nas janelas vizinhas (±viz janelas, ≈ ±45 s): uma janela com extrassístole não muda
         a escala;
      2. o limiar da soma deixa de ser frac × máximo da janela e passa a ser frac × MEDIANA dos máximos das
         janelas vizinhas;
      3. como a referência não depende da janela, uma janela sem batimento (pausa) não "acha" ruído.
    Custo: ~1,3× o original (o percentil da 1ª passada usa a matriz amostrada de 2 em 2, que dá o mesmo
    valor com erro < 1 %)."""
    sig = np.asarray(sig, float)
    if sig.ndim == 1:
        sig = sig[:, None]
    fim = len(sig) if fim is None else min(int(fim), len(sig))
    pad = int(2 * fs)
    a0, b0 = max(0, ini - pad), min(len(sig), fim + pad)
    filt = filtro_passa_banda(sig[a0:b0], fs)
    W, P = int(round(win_s * fs)), int(round(passo_s * fs))
    borda = (W - P) // 2
    inicios = [s for s in range(ini - a0, max(ini - a0 + 1, fim - a0 - W + 1), P) if len(filt[s:s + W]) >= W // 2]
    if not inicios:
        return {"centroide": np.array([], int), "pico": None, "n_janelas": 0}
    # 1ª passada: percentil q de cada janela (matriz amostrada de 2 em 2)
    qs = np.array([_quantil_rapido(matriz_distancia(filt[s:s + W][::2]), q) for s in inicios])
    lim_d = pd.Series(qs).rolling(2 * viz + 1, center=True, min_periods=1).median().to_numpy()
    # 2ª passada: binariza com a referência móvel e guarda a soma das colunas
    somas = []
    for s, L in zip(inicios, lim_d):
        D = matriz_distancia(filt[s:s + W])
        somas.append((D >= L).sum(axis=0).astype(float) if L > 1e-9 else np.zeros(len(D)))
    maxs = np.array([x.max() for x in somas])
    ref_s = pd.Series(maxs).rolling(2 * viz + 1, center=True, min_periods=1).median().to_numpy()
    # 3ª passada: limiar móvel, DBSCAN e centroides (igual à Nicole)
    min_a = max(1, int(round(min_s * fs)))
    marcas = []
    for s, soma, R in zip(inicios, somas, ref_s):
        if R <= 0:
            continue
        cols = np.flatnonzero(soma > frac * R)
        if len(cols) < min_a:
            continue
        cent = [g.mean() for g in _dbscan_1d(cols, max(1.0, eps_s * fs), min_a)]
        lo = 0 if s == ini - a0 else borda
        hi = len(soma) if s + W >= fim - a0 else W - borda
        marcas.extend(s + c for c in cent if lo <= c < hi)
    marcas = np.array(sorted(marcas))
    if len(marcas):
        grupos = np.r_[0, np.cumsum(np.diff(marcas) > 0.150 * fs)]
        marcas = np.array([marcas[grupos == g].mean() for g in np.unique(grupos)])
    cent = (marcas + a0).round().astype(int)
    pico = (refinar_pico(filt, marcas.round().astype(int), fs) + a0) if refinar and len(marcas) else None
    return {"centroide": cent, "pico": pico, "n_janelas": len(inicios)}


def comparar_com_referencia(det, ref, fs: float, tol_s: float = 0.150) -> dict:
    """Casa cada marca de referência (.atr) com no máximo uma detecção a até ±tol_s (AAMI EC57: 150 ms;
    a Nicole usou janela de 120 ms = ±60 ms). VP, FP, FN, sensibilidade, VPP (precisão), F1 e o erro de
    tempo (ms, detecção − referência) dos pares casados."""
    det = np.sort(np.asarray(det, float))
    ref = np.sort(np.asarray(ref, float))
    usado = np.zeros(len(det), bool)
    erros = []
    tol = tol_s * fs
    for r in ref:
        k = np.searchsorted(det, r)
        cand = [j for j in (k - 1, k, k + 1) if 0 <= j < len(det) and not usado[j] and abs(det[j] - r) <= tol]
        if cand:
            j = min(cand, key=lambda j: abs(det[j] - r))
            usado[j] = True
            erros.append((det[j] - r) / fs * 1000)
    vp = len(erros)
    fn = len(ref) - vp
    fp = len(det) - vp
    se = vp / (vp + fn) if vp + fn else np.nan
    vpp = vp / (vp + fp) if vp + fp else np.nan
    f1 = 2 * se * vpp / (se + vpp) if se and vpp else np.nan
    e = np.array(erros)
    return {"VP": vp, "FP": fp, "FN": fn, "Se": se, "VPP": vpp, "F1": f1,
            "erro_med_ms": float(np.median(e)) if len(e) else np.nan,
            "erro_abs_p95_ms": float(np.percentile(np.abs(e), 95)) if len(e) else np.nan,
            "erros_ms": e}


def serie_rr_detectada(amostras, fs: float, lo: float = 300, hi: float = 2000, dif: float = 0.20,
                       janela: int = 11):
    """RR a partir de detecções sem rótulo de batimento (não sabemos o que é sinusal):
    válido = dentro de [lo, hi] ms e passou no filtro de ectópicos de 20 % (Kleiger), descartando também o
    intervalo seguinte a um ectópico — a mesma regra de limpar_rr do resto do TCC.
    Mesmo formato de serie_rr_anotada: (t_s do 2º batimento, rr_ms, valido)."""
    a = np.sort(np.asarray(amostras, float))
    if len(a) < 3:
        return np.array([]), np.array([]), np.array([], bool)
    t = a / fs
    rr = np.diff(t) * 1000.0
    mk = marcar_ectopicos(rr, lo=lo, hi=hi, dif=dif, janela=janela)
    valido = mk["ok"] & ~mk["ect"] & ~mk["seg"]
    return t[1:], rr, valido


def ler_sinal(base: str, registro: str, pasta: str, ini_s: float = 0, fim_s=None) -> tuple:
    """Lê o trecho [ini_s, fim_s) de um registro WFDB. Usa o .dat local se existir em `pasta`;
    senão lê direto do PhysioNet (pn_dir) só o trecho pedido, sem salvar o sinal no Drive.
    Devolve (sinal amostras × derivações, fs, amostra_inicial)."""
    import wfdb
    local = os.path.exists(os.path.join(pasta, f"{registro}.dat"))
    tem_hea = os.path.exists(os.path.join(pasta, f"{registro}.hea"))
    kw = {} if local else {"pn_dir": base}
    caminho = os.path.join(pasta, registro) if local else registro
    h = wfdb.rdheader(os.path.join(pasta, registro)) if tem_hea else \
        _com_retentativas(lambda: wfdb.rdheader(caminho, **kw), rotulo=f"{base}/{registro}.hea")
    fs = float(h.fs)
    a = max(0, int(ini_s * fs))
    b = h.sig_len if fim_s is None else min(h.sig_len, int(fim_s * fs))
    rec = wfdb.rdrecord(caminho, sampfrom=a, sampto=b, **kw) if local else \
        _com_retentativas(lambda: wfdb.rdrecord(caminho, sampfrom=a, sampto=b, **kw), rotulo=f"{base}/{registro}")
    sig = rec.p_signal
    sig = np.where(np.isfinite(sig), sig, 0.0)
    return sig, fs, a


def detectar_pendente(pasta_det: str, base: str, registro: str, sufixo: str = "") -> bool:
    """True se a detecção completa deste registro ainda não foi salva (notebook 10, seção 4)."""
    return not os.path.exists(os.path.join(pasta_det, f"{base}_{registro}{sufixo}.npz"))


def efeito_ultima_hora(df: pd.DataFrame, basal_lim: float, terminal=(-60, 0),
                       indices=("FC", "SDNN", "RMSSD", "AR1"), n_min: int = 3) -> pd.DataFrame:
    """Efeito por registro (mesma regra do notebook 06): mediana das janelas da última hora −
    mediana das janelas basais (min_ate_evento <= basal_lim). Exige >= n_min janelas aprovadas em cada lado."""
    out = []
    chaves = [c for c in ("base", "fonte", "registro") if c in df.columns]
    for k, g in df[df.passed].groupby(chaves):
        k = k if isinstance(k, tuple) else (k,)
        b = g[g.min_ate_evento <= basal_lim]
        t = g[(g.min_ate_evento > terminal[0]) & (g.min_ate_evento <= terminal[1])]
        reg = dict(zip(chaves, k)) | {"n_basal": len(b), "n_terminal": len(t)}
        for ind in indices:
            reg[f"{ind}_delta"] = (t[ind].median() - b[ind].median()) if len(b) >= n_min and len(t) >= n_min else np.nan
        out.append(reg)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# 9. Delineação do batimento: início/fim do QRS (ponto J) e início/pico/fim da onda T
#    Ancorada nos QRS do detector da Nicole (seção 8). O que a Nicole NÃO fez: as bordas da T.
#    Ideia central: com a linha isoelétrica b do batimento (o trecho mais "parado" do segmento PR),
#    d(n) = ‖x(n) − b‖ é a COLUNA da matriz de distância no instante isoelétrico — a distância de cada
#    amostra ao estado de repouso. Ondas são os trechos que se afastam do repouso (não recorrentes com
#    ele); por ser uma norma, d(n) é sempre positivo: T invertida ou positiva viram a mesma "corcova".
# ---------------------------------------------------------------------------
TOL_CSE_MS = {"Pon": 10.2, "Pfim": 12.7, "QRSon": 6.5, "QRSfim": 11.6, "Tfim": 30.6}   # 2·s_CSE (CSE Working Party, 1985)


def preparar_para_delinear(sig: np.ndarray, fs: float, fc_baixa: float = 40.0) -> np.ndarray:
    """Sinal para medir ondas lentas (P, ST, T). Remove a linha de base com dois filtros de mediana em
    sequência (200 ms tira QRS e P; 600 ms tira a T; o que sobra é a linha de base — de Chazal et al., 2004)
    e aplica passa-baixas de 40 Hz. NÃO usa o passa-alta de 1 Hz do detector de QRS, que deforma ST e T."""
    from scipy.ndimage import median_filter
    from scipy.signal import butter, filtfilt
    x = np.asarray(sig, float)
    x = x[:, None] if x.ndim == 1 else x
    k1, k2 = int(0.2 * fs) | 1, int(0.6 * fs) | 1
    base = median_filter(median_filter(x, size=(k1, 1), mode="nearest"), size=(k2, 1), mode="nearest")
    b, a = butter(3, min(fc_baixa, 0.45 * fs) / (fs / 2))
    return filtfilt(b, a, x - base, axis=0)


def ponto_isoeletrico(x: np.ndarray, r: int, fs: float, ini_s: float = -0.12, fim_s: float = -0.03,
                      larg_s: float = 0.02) -> tuple:
    """Linha isoelétrica do batimento: o trecho de 20 ms mais "parado" (menor soma de |derivada|) entre
    120 e 30 ms antes do R, ou seja, no segmento PR. Devolve (vetor de nível por derivação, índice central)."""
    w = max(2, int(round(larg_s * fs)))
    a = max(0, r + int(round(ini_s * fs)))
    b = min(len(x), max(a + w + 1, r + int(round(fim_s * fs))))
    seg = x[a:b]
    dx = np.abs(np.diff(seg, axis=0)).sum(axis=1)
    if len(dx) < w:
        return seg.mean(axis=0), a
    e = np.convolve(dx, np.ones(w), "valid")
    k = int(np.argmin(e))
    return x[a + k:a + k + w].mean(axis=0), a + k + w // 2


def _trapezio_fim(y: np.ndarray, im: int, ir: int) -> int:
    """Método do trapézio (Vázquez-Seisdedos et al., 2011) para o FIM de uma onda que desce de im até ir:
    para cada candidato i, área do trapézio com vértices (im, y[im]), (i, y[i]), (ir, y[i]), (ir, y[im]);
    o fim da onda é o i de área máxima — o "joelho" onde a descida termina."""
    i = np.arange(im, ir + 1)
    return int(i[np.argmax(0.5 * (y[im] - y[i]) * (2 * ir - i - im))])


def _trapezio_inicio(y: np.ndarray, im: int, ir: int) -> int:
    """Espelho de _trapezio_fim para o INÍCIO de uma onda (ir < im; y sobe de ir até im)."""
    i = np.arange(ir, im + 1)
    return int(i[np.argmax(0.5 * (y[im] - y[i]) * (i + im - 2 * ir))])


def delinear_batimentos(sig_prep: np.ndarray, fs: float, r_amostras, alfa_faixa: float = 0.20,
                        amp_t_min: float = 0.05, versao: int = 1, qtc_max: float = 0.50,
                        folga_p_s: float = 0.20, prom_min: float = 0.10) -> pd.DataFrame:
    """Delineia cada batimento a partir do R (detector da Nicole) e do sinal de preparar_para_delinear.
    Pontos (amostras): QRSon, QRSfim (ponto J), Ton, Tpico, Tfim — pelo método do trapézio na curva d(n);
    e Ton_faixa, Tfim_faixa — pela largura da faixa da T na matriz: onde d(n) cruza
    piso + alfa·(pico − piso), indo do pico para fora (a versão "estilo Nicole", com limiar).
    T não detectável (NaN) se a corcova da T for menor que amp_t_min × a do QRS.
    Janelas de busca (relativas ao R e ao RR seguinte):
      QRS: ±120 ms;  T: de J + 40 ms até R + min(0,6 s; 0,7·RR), sem passar de R_seguinte − 120 ms.
    versao=2 (correção após a validação na QT Database, notebook 11) — três mudanças só na onda T:
      1. janela FISIOLÓGICA para o pico da T: até R + qtc_max·√RR (Bazett com QTc máximo de 500 ms) e nunca
         depois de R_seguinte − folga_p_s (200 ms). Motivo: em frequência alta a janela antiga alcançava a
         onda P do batimento SEGUINTE (sel41: marcava a P como T, erro de +190 ms);
      2. o pico da T é o máximo local MAIS PROEMINENTE de d(n), nunca a borda da janela. Proeminência = o
         quanto a corcova se destaca dos vales vizinhos; um platô (ST deslocado) não vence uma corcova;
      3. qualidade: T_confiavel = proeminência ≥ prom_min × amplitude do QRS. T achatada (sel45, sel46) fica
         marcada como não confiável — o especialista e o método discordam nelas por ambiguidade real.
    O fim da T (trapézio) também deixa de buscar além de R_seguinte − folga_p_s."""
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    x = np.asarray(sig_prep, float)
    x = x[:, None] if x.ndim == 1 else x
    r = np.sort(np.asarray(r_amostras, int))
    n = len(x)
    ms = lambda s: int(round(s * fs))
    linhas = []
    for k in range(1, len(r) - 1):
        R, Rn = r[k], r[k + 1]
        if R - ms(0.2) < 0 or Rn >= n:
            continue
        b, iso = ponto_isoeletrico(x, R, fs)
        a0, a1 = max(0, R - ms(0.25)), min(n, Rn)
        d = np.sqrt(((x[a0:a1] - b) ** 2).sum(axis=1))          # coluna da matriz no instante isoelétrico
        env = maximum_filter1d(d, max(1, ms(0.02)))             # envoltória: une as corcovas de R e S
        dT = uniform_filter1d(d, max(1, ms(0.02)))              # suavizada para a T
        loc = lambda i: i - a0
        rl = loc(R)
        reg = {"R": R, "iso": iso, "RR_s": (Rn - R) / fs}
        # QRS: início e fim (ponto J)
        s_env = np.gradient(env)
        i0 = max(0, rl - ms(0.10))
        im_on = i0 + int(np.argmax(s_env[i0:rl + 1])) if rl > i0 else rl
        ir_on = max(0, rl - ms(0.12))
        qon = _trapezio_inicio(env, im_on, ir_on) if im_on > ir_on + 1 else im_on
        i1 = min(len(env) - 1, rl + ms(0.10))
        im_off = rl + int(np.argmin(s_env[rl:i1 + 1]))
        ir_off = min(len(env) - 1, rl + ms(0.15))
        qfim = _trapezio_fim(env, im_off, ir_off) if ir_off > im_off + 1 else im_off
        reg.update({"QRSon": qon + a0, "QRSfim": qfim + a0, "amp_QRS": float(env[rl])})
        # T: pico, início, fim
        ini = qfim + ms(0.04)
        fim = min(rl + ms(min(0.6, 0.7 * (Rn - R) / fs)), loc(Rn) - ms(0.12), len(dT) - 1)
        tvals = {"Ton": np.nan, "Tpico": np.nan, "Tfim": np.nan, "Ton_faixa": np.nan, "Tfim_faixa": np.nan,
                 "amp_T": np.nan, "T_prom": np.nan, "T_confiavel": False}
        if versao == 2:
            from scipy.signal import find_peaks
            rr_s = (Rn - R) / fs
            lim_d = min(len(dT) - 1, loc(Rn) - ms(folga_p_s))
            fim2 = min(rl + ms(qtc_max * np.sqrt(rr_s)), lim_d)
            if fim2 - ini > ms(0.04):
                seg = dT[ini:fim2 + 1]
                pk, pr = find_peaks(seg, prominence=0)
                if len(pk):
                    j = int(np.argmax(pr["prominences"]))
                    tp, prom = ini + int(pk[j]), float(pr["prominences"][j])
                    k_iso = loc(iso)
                    piso = float(np.median(dT[max(0, k_iso - ms(0.01)):k_iso + ms(0.01) + 1]))
                    s_ = np.gradient(dT)
                    im_f = tp + int(np.argmin(s_[tp:min(lim_d, tp + ms(0.20)) + 1]))
                    ir_f = min(lim_d, im_f + ms(0.18))
                    tf = _trapezio_fim(dT, im_f, ir_f) if ir_f > im_f + 1 else im_f
                    im_i = ini + int(np.argmax(s_[ini:tp + 1]))
                    to = _trapezio_inicio(dT, im_i, ini) if im_i > ini + 1 else ini
                    tvals.update({"Ton": to + a0, "Tpico": tp + a0, "Tfim": tf + a0, "amp_T": float(dT[tp] - piso),
                                  "T_prom": prom, "T_confiavel": bool(prom >= prom_min * max(reg["amp_QRS"], 1e-9))})
            reg.update(tvals)
            linhas.append(reg)
            continue
        if fim - ini > ms(0.08):
            tp = ini + int(np.argmax(dT[ini:fim + 1]))
            k_iso = loc(iso)
            piso = float(np.median(dT[max(0, k_iso - ms(0.01)):k_iso + ms(0.01) + 1]))
            amp = float(dT[tp] - piso)
            if amp >= amp_t_min * max(reg["amp_QRS"], 1e-9) and ini < tp < fim:
                s = np.gradient(dT)
                lim_d = min(len(dT) - 1, loc(Rn) - ms(0.12))
                im_f = tp + int(np.argmin(s[tp:min(lim_d, tp + ms(0.20)) + 1]))
                ir_f = min(lim_d, im_f + ms(0.18))
                tf = _trapezio_fim(dT, im_f, ir_f) if ir_f > im_f + 1 else im_f
                im_i = ini + int(np.argmax(s[ini:tp + 1]))
                to = _trapezio_inicio(dT, im_i, ini) if im_i > ini + 1 else ini
                thr = piso + alfa_faixa * amp
                abaixo_antes = np.flatnonzero(dT[ini:tp] < thr)
                abaixo_depois = np.flatnonzero(dT[tp:lim_d + 1] < thr)
                tvals = {"Ton": to + a0, "Tpico": tp + a0, "Tfim": tf + a0,
                         "Ton_faixa": (ini + abaixo_antes[-1] + 1 + a0) if len(abaixo_antes) else np.nan,
                         "Tfim_faixa": (tp + abaixo_depois[0] + a0) if len(abaixo_depois) else np.nan,
                         "amp_T": amp}
        reg.update(tvals)
        linhas.append(reg)
    return pd.DataFrame(linhas)


def ler_marcas_qtdb(caminho_base: str, ext: str = "q1c") -> pd.DataFrame:
    """Marcações manuais da QT Database (.q1c = anotador 1, revisado). Cada onda é anotada como
    '(' início, símbolo do pico ('p' = P, 't' = T, 'u' = U, outro = QRS) e ')' fim — início e fim só quando o
    anotador marcou. Devolve um batimento por linha: R, QRSon, QRSfim, Pon, Ppico, Pfim, Ton, Tpico, Tfim."""
    import wfdb
    ann = wfdb.rdann(caminho_base, ext)
    s, a = list(ann.symbol), np.asarray(ann.sample)
    ondas = []
    for i, sy in enumerate(s):
        if sy in "()":
            continue
        ondas.append({"onda": {"p": "P", "t": "T", "u": "U"}.get(sy, "QRS"), "pico": a[i],
                      "ini": a[i - 1] if i > 0 and s[i - 1] == "(" else np.nan,
                      "fim": a[i + 1] if i + 1 < len(s) and s[i + 1] == ")" else np.nan})
    ondas = pd.DataFrame(ondas)
    if ondas.empty:
        return pd.DataFrame()
    qrs = ondas[ondas.onda == "QRS"].reset_index(drop=True)
    out = []
    for k, q in qrs.iterrows():
        ant = qrs.pico[k - 1] if k > 0 else -np.inf
        prox = qrs.pico[k + 1] if k + 1 < len(qrs) else np.inf
        reg = {"R": q.pico, "QRSon": q.ini, "QRSfim": q.fim}
        p = ondas[(ondas.onda == "P") & (ondas.pico > ant) & (ondas.pico < q.pico)].tail(1)
        t = ondas[(ondas.onda == "T") & (ondas.pico > q.pico) & (ondas.pico < prox)].head(1)
        for nome, w in (("P", p), ("T", t)):
            reg.update({f"{nome}on": w.ini.iloc[0] if len(w) else np.nan, f"{nome}pico": w.pico.iloc[0] if len(w) else np.nan,
                        f"{nome}fim": w.fim.iloc[0] if len(w) else np.nan})
        out.append(reg)
    return pd.DataFrame(out)


def delinear_p_st(sig_prep: np.ndarray, fs: float, de: pd.DataFrame, pr_max_s: float = 0.30,
                  folga_qrs_s: float = 0.03, prom_min_p: float = 0.04, st_ms: float = 60.0) -> pd.DataFrame:
    """Onda P e nível do ST, batimento a batimento, a partir da saída de delinear_batimentos (R, QRSon, QRSfim,
    iso, Tfim). Mesma ideia da T: d(n) = ‖x(n) − b‖, distância de cada amostra ao nível isoelétrico do PR.
    ONDA P — busca entre QRSon − pr_max_s (PR de até 300 ms) e QRSon − folga_qrs_s, e nunca antes do fim da T
      anterior + 20 ms (ou R anterior + 250 ms, se a T anterior não foi achada). Pico = máximo local MAIS
      PROEMINENTE de d(n); início e fim = método do trapézio (como a T). P_confiavel = proeminência ≥
      prom_min_p × amplitude do QRS (a P tem ~10–25 % do QRS; em fibrilação atrial não há P e a marca deve
      sair "não confiável").
    ST — ST{st_ms} = nível em J + st_ms ms menos o nível isoelétrico, POR DERIVAÇÃO (com sinal: + supra,
      − infra, em mV) e como distância ‖x(J+st) − b‖ (sem sinal, as derivações juntas). É a medida clássica
      (J + 60 ms) e NÃO depende do fim nem do início da T, só do ponto J e da linha isoelétrica."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks
    x = np.asarray(sig_prep, float)
    x = x[:, None] if x.ndim == 1 else x
    n, nc = len(x), x.shape[1]
    ms = lambda v: int(round(v * fs))
    out = de.copy()
    for c in ["Pon", "Ppico", "Pfim", "P_prom", "ST_dist"] + [f"ST_d{k + 1}" for k in range(nc)]:
        out[c] = np.nan
    out["P_confiavel"] = False
    tfim_ant = np.r_[np.nan, out["Tfim"].to_numpy(float)[:-1]] if "Tfim" in out else np.full(len(out), np.nan)
    r_ant = np.r_[np.nan, out["R"].to_numpy(float)[:-1]]
    for i, b in enumerate(out.itertuples(index=False)):
        R, qon, qfim = int(b.R), int(b.QRSon), int(b.QRSfim)
        nivel, _ = ponto_isoeletrico(x, R, fs)
        # ST
        j = qfim + ms(st_ms / 1000.0)
        if j < n:
            dv = x[j] - nivel
            out.iat[i, out.columns.get_loc("ST_dist")] = float(np.sqrt((dv ** 2).sum()))
            for k in range(nc):
                out.iat[i, out.columns.get_loc(f"ST_d{k + 1}")] = float(dv[k])
        # P
        ini = qon - ms(pr_max_s)
        lim_ant = (tfim_ant[i] + ms(0.02)) if np.isfinite(tfim_ant[i]) else ((r_ant[i] + ms(0.25)) if np.isfinite(r_ant[i]) else ini)
        ini = int(max(ini, lim_ant, 0))
        fim = qon - ms(folga_qrs_s)
        if fim - ini <= ms(0.05):
            continue
        a0 = max(0, ini - ms(0.02))
        d = np.sqrt(((x[a0:qon + 1] - nivel) ** 2).sum(axis=1))
        dP = uniform_filter1d(d, max(1, ms(0.02)))
        li, lf, lq = ini - a0, fim - a0, qon - a0
        pk, pr = find_peaks(dP[li:lf + 1], prominence=0)
        if not len(pk):
            continue
        k = int(np.argmax(pr["prominences"]))
        pp, prom = li + int(pk[k]), float(pr["prominences"][k])
        s_ = np.gradient(dP)
        im_i = li + int(np.argmax(s_[li:pp + 1]))
        p_on = _trapezio_inicio(dP, im_i, li) if im_i > li + 1 else li
        im_f = pp + int(np.argmin(s_[pp:lq + 1]))
        p_fim = _trapezio_fim(dP, im_f, lq) if lq > im_f + 1 else im_f
        amp_qrs = float(getattr(b, "amp_QRS", np.nan)) if hasattr(b, "amp_QRS") else np.nan
        out.iat[i, out.columns.get_loc("Pon")] = p_on + a0
        out.iat[i, out.columns.get_loc("Ppico")] = pp + a0
        out.iat[i, out.columns.get_loc("Pfim")] = p_fim + a0
        out.iat[i, out.columns.get_loc("P_prom")] = prom
        out.iat[i, out.columns.get_loc("P_confiavel")] = bool(np.isfinite(amp_qrs) and prom >= prom_min_p * amp_qrs)
    return out


def st_no_ponto_j(sig_prep: np.ndarray, fs: float, r: int, j: int, st_ms: float = 60.0) -> np.ndarray:
    """Nível em j + st_ms (por derivação) menos o nível isoelétrico do batimento de pico R — para medir quanto o
    erro do ponto J muda o ST (comparando o J do método com o J do especialista)."""
    x = np.asarray(sig_prep, float)
    x = x[:, None] if x.ndim == 1 else x
    nivel, _ = ponto_isoeletrico(x, int(r), fs)
    k = int(j) + int(round(st_ms / 1000.0 * fs))
    return x[k] - nivel if 0 <= k < len(x) else np.full(x.shape[1], np.nan)


def comparar_delineacao(det: pd.DataFrame, man: pd.DataFrame, fs: float, tol_r_s: float = 0.150,
                        pontos=("QRSon", "QRSfim", "Ton", "Tpico", "Tfim", "Ton_faixa", "Tfim_faixa"),
                        extras=()) -> pd.DataFrame:
    """Casa cada batimento manual com o detectado de R mais próximo (até ±tol_r_s) e devolve, por batimento,
    o erro (ms, detectado − manual) de cada ponto. Os pontos *_faixa são comparados com Ton/Tfim manuais."""
    if det.empty or man.empty:
        return pd.DataFrame()
    rd = det.R.to_numpy()
    out = []
    for _, m in man.iterrows():
        j = int(np.argmin(np.abs(rd - m.R)))
        if abs(rd[j] - m.R) > tol_r_s * fs:
            out.append({"R_man": m.R, "casou": False})
            continue
        reg = {"R_man": m.R, "casou": True}
        for x in extras:
            if x in det.columns:
                reg[x] = det.iloc[j][x]
        for p in pontos:
            ref = m.get(p.replace("_faixa", ""), np.nan)
            v = det.iloc[j].get(p, np.nan)
            reg[p] = (v - ref) / fs * 1000 if np.isfinite(ref) and np.isfinite(v) else np.nan
            reg[f"{p}_tem_ref"] = bool(np.isfinite(ref))
        out.append(reg)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# 10. ST e onda P batimento a batimento e recorrência ST × P (notebook 13)
# ---------------------------------------------------------------------------
MEDIDAS_ST_P = ("ST60_d1", "ST60_d2", "ST_dist", "COER_ST", "COER_P", "RR_ST", "DET_ST", "RR_P", "DET_P", "JRR", "RAZAO_JRR")
# Família final (notebook 14): as coerências trocadas pela versão sem tendência; o resto igual.
MEDIDAS_ST_P_FINAL = ("ST60_d1", "ST60_d2", "ST_dist", "COER_ST_semtend", "COER_P_semtend", "RR_ST", "DET_ST", "RR_P",
                      "DET_P", "JRR", "RAZAO_JRR")


def sinal_para_st(sig: np.ndarray, fs: float, iso_idx, fc_baixa: float = 40.0) -> np.ndarray:
    """Sinal para medir o ST com a técnica padrão: passa-baixas de 40 Hz e linha de base por SPLINE CÚBICA passando
    pelos pontos isoelétricos (PR) de cada batimento — o nível isoelétrico vira zero em todo batimento.
    Diferente do filtro de mediana de 600 ms, a spline só "vê" o PR, então não engole um ST deslocado."""
    from scipy.interpolate import CubicSpline
    from scipy.signal import butter, filtfilt
    x = np.asarray(sig, float)
    x = x[:, None] if x.ndim == 1 else x
    b, a = butter(3, min(fc_baixa, 0.45 * fs) / (fs / 2))
    x = filtfilt(b, a, x, axis=0)
    w = max(1, int(round(0.01 * fs)))
    k = np.unique(np.asarray([i for i in np.asarray(iso_idx, int) if w <= i < len(x) - w]))
    if len(k) < 2:
        return x - np.median(x, axis=0)
    vals = np.array([x[i - w:i + w + 1].mean(axis=0) for i in k])
    cs = CubicSpline(k, vals, axis=0)
    t = np.clip(np.arange(len(x)), k[0], k[-1])
    return x - cs(t)


def extrair_batimentos(sig: np.ndarray, fs: float, r_amostras, st_ms: float = 60.0, st_janela_s: float = 0.12,
                       p_janela_s=(0.25, 0.03)) -> tuple:
    """Para um trecho de sinal: delineia (v2) e devolve (tabela por batimento, vetores ST, vetores P).
    Tabela: R, QRSon, QRSfim, iso, ST60_d* (spline), ST60m_d* (filtro de mediana, para comparar), ST_dist.
    Vetor ST = o sinal (spline) de J até J + 120 ms, as derivações em sequência; vetor P = de QRSon − 250 ms até
    QRSon − 30 ms. São os "pontos" da recorrência entre batimentos: não precisam das bordas exatas de P e T."""
    x = np.asarray(sig, float)
    x = x[:, None] if x.ndim == 1 else x
    nc = x.shape[1]
    xp = preparar_para_delinear(x, fs)
    de = delinear_batimentos(xp, fs, r_amostras, versao=2)
    if de.empty:
        return de, np.zeros((0, 1)), np.zeros((0, 1))
    xs = sinal_para_st(x, fs, de.iso.to_numpy())
    from scipy.signal import butter as _bt, filtfilt as _ff
    _b, _a = _bt(3, min(15.0, 0.45 * fs) / (fs / 2))
    hf = xs - _ff(_b, _a, xs, axis=0)            # conteúdo acima de 15 Hz: ruído muscular/eletrodo fora do QRS
    ms = lambda v: int(round(v * fs))
    j60, lst = ms(st_ms / 1000.0), ms(st_janela_s)
    pa, pb = ms(p_janela_s[0]), ms(p_janela_s[1])
    n = len(x)
    st_s = np.full((len(de), nc), np.nan); st_m = np.full((len(de), nc), np.nan); ruido = np.full(len(de), np.nan)
    Vst = np.full((len(de), lst * nc), np.nan, np.float32); Vp = np.full((len(de), (pa - pb) * nc), np.nan, np.float32)
    for i, b in enumerate(de.itertuples(index=False)):
        J, qon, R = int(b.QRSfim), int(b.QRSon), int(b.R)
        if J + max(j60, lst) < n:
            st_s[i] = xs[J + j60]
            nivel, _ = ponto_isoeletrico(xp, R, fs)
            st_m[i] = xp[J + j60] - nivel
            Vst[i] = xs[J:J + lst].T.ravel()
        if qon - pa >= 0:
            Vp[i] = xs[qon - pa:qon - pb].T.ravel()
            ruido[i] = float(np.sqrt(np.mean(hf[qon - pa:qon - pb] ** 2)))   # RMS > 15 Hz no trecho da P (sem QRS)
    for c in range(nc):
        de[f"ST60_d{c + 1}"] = st_s[:, c]
        de[f"ST60m_d{c + 1}"] = st_m[:, c]
    de["ST_dist"] = np.sqrt(np.nansum(st_s ** 2, axis=1))
    de["ruido"] = ruido
    return de, Vst, Vp


def _det(R: np.ndarray, lmin: int = 2) -> float:
    """Determinismo: fração dos pontos recorrentes (fora da diagonal principal) que estão em linhas diagonais de
    comprimento ≥ lmin — sequências de batimentos que se repetem em ordem."""
    N = len(R)
    tot, emlinha = 0, 0
    for k in range(1, N):
        d = np.diagonal(R, k).astype(np.int8)
        s = int(d.sum())
        if s == 0:
            continue
        tot += s
        borda = np.diff(np.r_[0, d, 0])
        ini, fim = np.flatnonzero(borda == 1), np.flatnonzero(borda == -1)
        comp = fim - ini
        emlinha += int(comp[comp >= lmin].sum())
    return emlinha / tot if tot else np.nan


def coerencia(V: np.ndarray, max_pares: int = 3000, rng=None, nc: int = 1, sem_tendencia: bool = False) -> float:
    """Mediana da correlação de Pearson entre pares de batimentos (formas centradas). Ritmo sinusal: a P se repete
    (≈ 1); fibrilação atrial: cada "P" é diferente (≈ 0). Não depende de limiar nem de escala.
    sem_tendencia=True tira a reta de cada derivação (V tem nc blocos em sequência) antes de correlacionar. Motivo
    (notebook 14): como a spline zera a linha de base no ponto isoelétrico logo antes do QRS, um trecho SEM onda
    nenhuma vira uma rampa que termina em zero — e rampas iguais têm correlação alta. Na QT Database isso dava
    coerência 0,69 a um registro em FA (sel36); sem a tendência, 0,05."""
    V = np.asarray(V, float)
    V = V[np.isfinite(V).all(axis=1)]
    if len(V) < 10:
        return np.nan
    if sem_tendencia:
        from scipy.signal import detrend
        L = V.shape[1] // max(int(nc), 1)
        V = np.hstack([detrend(V[:, k * L:(k + 1) * L], axis=1) for k in range(max(int(nc), 1))])
    rng = np.random.default_rng(0) if rng is None else rng
    Z = V - V.mean(axis=1, keepdims=True)
    nz = np.linalg.norm(Z, axis=1)
    ok = nz > 1e-9
    Z = Z[ok] / nz[ok, None]
    if len(Z) < 10:
        return np.nan
    i = rng.integers(0, len(Z), max_pares); j = rng.integers(0, len(Z), max_pares)
    m = i != j
    return float(np.median((Z[i[m]] * Z[j[m]]).sum(axis=1)))


def medidas_janela(Vst: np.ndarray, Vp: np.ndarray, eps_st: float, eps_p: float, max_bat: int = 300, nc: int = 1) -> dict:
    """Recorrência entre batimentos numa janela: RR (taxa de recorrência), DET, coerência, e a recorrência CONJUNTA
    ST × P: JRR = fração de pares de batimentos recorrentes nos dois ao mesmo tempo; RAZAO_JRR = JRR ÷ (RR_ST·RR_P)
    (1 = ST e P recorrem de forma independente; > 1 = recorrem juntos, sinal de um fator comum — postura, eletrodo,
    tônus autonômico; < 1 = mudam em momentos diferentes). COER_*_semtend = coerência sem a tendência linear de cada
    derivação (ver coerencia); nc = número de derivações nos vetores."""
    from scipy.spatial.distance import cdist
    ok = np.isfinite(Vst).all(axis=1) & np.isfinite(Vp).all(axis=1)
    Vst, Vp = Vst[ok], Vp[ok]
    out = {k: np.nan for k in ("COER_ST", "COER_P", "COER_ST_semtend", "COER_P_semtend", "RR_ST", "DET_ST", "RR_P",
                               "DET_P", "JRR", "RAZAO_JRR")}
    if len(Vst) < 30:
        return out
    if len(Vst) > max_bat:                       # amostra uniforme, mantendo a ordem no tempo (para o DET)
        idx = np.linspace(0, len(Vst) - 1, max_bat).round().astype(int)
        Vst, Vp = Vst[idx], Vp[idx]
    off = ~np.eye(len(Vst), dtype=bool)
    Rst = cdist(Vst, Vst) <= eps_st
    Rp = cdist(Vp, Vp) <= eps_p
    rr_st, rr_p = Rst[off].mean(), Rp[off].mean()
    jrr = (Rst & Rp)[off].mean()
    out.update({"COER_ST": coerencia(Vst), "COER_P": coerencia(Vp),
                "COER_ST_semtend": coerencia(Vst, nc=nc, sem_tendencia=True),
                "COER_P_semtend": coerencia(Vp, nc=nc, sem_tendencia=True), "RR_ST": rr_st, "RR_P": rr_p,
                "DET_ST": _det(Rst), "DET_P": _det(Rp), "JRR": jrr,
                "RAZAO_JRR": jrr / (rr_st * rr_p) if rr_st > 0 and rr_p > 0 else np.nan})
    return out


def eps_do_basal(V: np.ndarray, quantil: float = 0.10, max_bat: int = 600, seed: int = 0) -> float:
    """Raio de recorrência de UM registro, fixado no período basal: quantil 10 % das distâncias entre batimentos
    basais. Fixar no basal (e não janela a janela) é o que deixa a taxa de recorrência variar ao longo do tempo."""
    from scipy.spatial.distance import pdist
    V = np.asarray(V, float)
    V = V[np.isfinite(V).all(axis=1)]
    if len(V) < 30:
        return np.nan
    if len(V) > max_bat:
        V = V[np.random.default_rng(seed).choice(len(V), max_bat, replace=False)]
    return float(np.quantile(pdist(V), quantil))


def validade_batimentos(R_amostras, marcas, fs: float) -> np.ndarray:
    """Batimento válido = os dois RR em volta dele passam no filtro de ectópicos de 20 % (mesma regra do notebook 13)."""
    t_rr, rr, v_rr = serie_rr_detectada(marcas, fs)
    pos = {int(m): k for k, m in enumerate(np.sort(marcas))}
    k = np.array([pos.get(int(x), -1) for x in R_amostras])
    valido = (k >= 1) & (k < len(v_rr))
    valido[valido] = v_rr[k[valido] - 1] & v_rr[np.minimum(k[valido], len(v_rr) - 1)]
    return valido


def alternancia(Vst: np.ndarray, consecutivo: np.ndarray) -> float:
    """Índice de alternância do ST: mediana da distância entre batimentos vizinhos (k, k+1) dividida pela mediana entre
    batimentos a dois de distância (k, k+2), só em trincas consecutivas e válidas. > 1 = padrão ABAB (cada batimento
    parece mais com o de dois atrás que com o anterior), o que acontece na alternância da onda T."""
    V = np.asarray(Vst, float)
    c = np.asarray(consecutivo, bool)
    ok = c[:-2] & c[1:-1] & c[2:] if len(c) >= 3 else np.array([], bool)
    if ok.sum() < 20:
        return np.nan
    i = np.flatnonzero(ok)
    d1 = np.linalg.norm(V[i] - V[i + 1], axis=1)
    d2 = np.linalg.norm(V[i] - V[i + 2], axis=1)
    m2 = np.median(d2)
    return float(np.median(d1) / m2) if m2 > 0 else np.nan


def tabela_janelas(de: pd.DataFrame, Vs: np.ndarray, Vp: np.ndarray, valido: np.ndarray, t_evento: float, fs: float,
                   base: str, registro: str, win_s: float = 300, horas: float = 24, min_bat: int = 100,
                   gate: float = 0.80, eps_q: float = 0.10, basal_min: float = -120, extras: bool = True) -> pd.DataFrame:
    """Mesmas janelas e medidas do notebook 13, com os parâmetros abertos (para o estudo de sensibilidade) e, se
    extras=True, medidas para explicações alternativas: FC e DP do RR na janela, ruído (RMS > 15 Hz fora do QRS),
    fração de batimentos válidos e índice de alternância do ST."""
    t_b = de.R.to_numpy() / fs
    basal = valido & (t_b <= t_evento + basal_min * 60)
    if basal.sum() < 30:
        basal = valido & (t_b <= t_b.min() + 1800)
    eps_st, eps_p = eps_do_basal(Vs[basal], eps_q), eps_do_basal(Vp[basal], eps_q)
    ini_s = max(t_b.min() if len(t_b) else 0, t_evento - horas * 3600)
    rr_all = np.r_[np.nan, np.diff(t_b) * 1000]
    consec = np.r_[False, np.diff(de.R.to_numpy()) < 2.0 * fs] & valido
    linhas = []
    for s0 in np.arange(t_evento - win_s, ini_s - 1e-9, -win_s)[::-1]:
        m = (t_b >= s0) & (t_b < s0 + win_s)
        n_tot, mv = int(m.sum()), m & valido
        reg = {"base": base.upper(), "registro": registro, "t_inicio_s": s0, "min_ate_evento": (s0 + win_s - t_evento) / 60,
               "n_bat": n_tot, "frac_valida": mv.sum() / n_tot if n_tot else np.nan}
        reg["passed"] = bool(n_tot and mv.sum() >= min_bat and mv.sum() / n_tot >= gate)
        if extras and n_tot:
            reg["ruido"] = float(np.nanmedian(de.ruido.to_numpy()[m])) if "ruido" in de else np.nan
        if reg["passed"]:
            d = de[mv]
            for c in [x for x in de.columns if x.startswith("ST60") or x == "ST_dist"]:
                reg[c] = float(np.nanmedian(d[c]))
            nc = max(1, sum(1 for x in de.columns if x.startswith("ST60_d")))
            reg.update(medidas_janela(Vs[mv], Vp[mv], eps_st, eps_p, nc=nc))
            if extras:
                rr = rr_all[mv & np.r_[False, consec[1:]]]
                rr = rr[np.isfinite(rr)]
                reg["FC"] = 60000.0 / rr.mean() if len(rr) else np.nan
                reg["RR_dp"] = float(rr.std(ddof=1)) if len(rr) > 2 else np.nan
                reg["ALT_ST"] = alternancia(Vs[m], consec[m])
        linhas.append(reg)
    return pd.DataFrame(linhas)
