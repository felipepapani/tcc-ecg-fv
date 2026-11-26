# Matriz de distância do ECG antes da fibrilação ventricular

Código, resultados e ferramenta web do Trabalho de Conclusão de Curso de **Felipe Garcia Papani**
(Engenharia Biomédica, UFPE, 2026), orientado pelo Prof. Fernando José Ribeiro Sales.

- **Ferramenta web:** https://felipepapani.github.io/tcc-ecg-fv/ (detecção de QRS e delineação no navegador, com arquivos EDF, WFDB ou CSV; o ECG não sai do computador; veja [Ferramenta web](#ferramenta-web))
- **DOI (Zenodo):** _será gerado na primeira versão publicada_
- **Trabalho anterior:** código do artigo do CBEB 2026 em [felipepapani/Paper-CBEB](https://github.com/felipepapani/Paper-CBEB) ([doi:10.5281/zenodo.21689324](https://doi.org/10.5281/zenodo.21689324))

## O que o trabalho faz

Três etapas, cada uma avaliada com critério fixado antes da comparação:

| Etapa | Pergunta | Resultado |
| --- | --- | --- |
| 1. Detecção de QRS | O método da matriz de distância (Charron) funciona em Holter de 24 h? | Não atingiu o critério (F1 de 91,1 % e 87,8 %, contra 99 %). Os batimentos da análise vêm do detector `xqrs` (WFDB). |
| 2. Delineação | A distância de cada amostra ao nível isoelétrico, com o método do trapézio, marca as bordas das ondas? | Marca; nenhum ponto atinge a tolerância do CSE. O erro do ponto J muda o ST60 em 0,004 a 0,006 mV. |
| 3. Antes da FV | O segmento ST e a onda P mudam na última hora antes da fibrilação ventricular? | De 11 medidas, uma se sustenta: o ST60 na derivação 2 varia +0,029 mV nos 12 pacientes (12 de 12) e não nos 18 controles; g = 1,17, p de Holm = 0,010. Estudo exploratório. |

## Estrutura

```
codigo/
  tcc_lib.py                  biblioteca: leitura, filtros, matriz de distância, detecção, delineação, medidas, estatística, figuras
  gerar_nb10.py … gerar_nb15.py  geram os cadernos abaixo (a fonte de verdade dos cadernos)
notebooks/
  10_qrs_matriz_sddb.ipynb      etapa 1: método de Charron, variantes e xqrs contra as anotações da SDDB
  11_delineacao_t_qtdb.ipynb    etapa 2: QRS, ponto J e onda T contra a QT Database (versões 1 e 2)
  12_onda_p_st_qtdb.ipynb       onda P, erro do ST60 e validação da coerência (sinusal × fibrilação atrial)
  13_recorrencia_st_p.ipynb     etapa 3: medidas por janela de 5 min na SDDB e na NSRDB
  14_figuras_numeros_st_p.ipynb figuras finais e tabela única de números (numeros_tcc.csv)
  15_sensibilidade_st_p.ipynb   sensibilidade a parâmetros e ao basal
analises_complementares/        contrastes de tempo, outra linha de base, modelo dentro do registro, delta de Cliff
resultados/
  csv/                          todos os números citados no texto (numeros_tcc.csv é a tabela-resumo)
  figuras/                      figuras do TCC
docs/index.html                 ferramenta web (publicada pelo GitHub Pages)
testes/                         testes dos leitores de arquivo da ferramenta web (EDF, WFDB, CSV)
```

## Ferramenta web

A página [felipepapani.github.io/tcc-ecg-fv](https://felipepapani.github.io/tcc-ecg-fv/) roda a detecção de QRS pela
matriz de distância e a delineação do QRS e da onda T no navegador. O arquivo é lido localmente e não é enviado a
nenhum servidor. É uma ferramenta de pesquisa, não de diagnóstico.

### Formatos aceitos

| Formato | O que selecionar | Detalhes |
| --- | --- | --- |
| **EDF, EDF+ ou BDF** | um arquivo `.edf` ou `.bdf` | Padrão de exportação da maioria dos Holter e polígrafos. A página usa os canais com a maior taxa de amostragem (o ECG) e ignora anotações e canais lentos (oximetria, por exemplo). Unidades µV, mV e V são convertidas para mV. |
| **WFDB (PhysioNet)** | o `.hea` e o `.dat` do registro, juntos | Formatos 16, 24, 32, 61, 80 e 212 (o da SDDB e da QTDB). Unidades do cabeçalho convertidas para mV. Registros de vários segmentos e os formatos 310/311 não são aceitos. |
| **CSV ou texto** | um `.csv`, `.tsv` ou `.txt` | Uma linha por amostra e uma coluna por derivação. Separador vírgula, ponto e vírgula, tabulação ou espaços; com ponto e vírgula ou tabulação, a vírgula decimal é aceita. A linha de nomes é opcional. Amostragem e unidade (mV, µV, V) são escolhidas na página; uma 1ª coluna de tempo é reconhecida sozinha, e a amostragem sai dela. Linhas de texto antes dos números são ignoradas, e a amostragem (“Sample Rate”) e a unidade (µV) escritas nelas são lidas. |

Os arquivos podem ser escolhidos no botão ou arrastados para a página. O tutorial “Como preparar e enviar o arquivo”,
na própria página, mostra o passo a passo e oferece um CSV de exemplo. Cada trecho analisado tem até 60 min; um Holter
de 24 h abre inteiro e é analisado em partes.

### Testes dos leitores

`testes/gerar_casos.py` escreve arquivos em todos os formatos com as bibliotecas de referência (`wfdb` e `pyedflib`) e
relê o sinal esperado com elas; `testes/testar_leitores.mjs` lê os mesmos arquivos com o código da página e compara
amostra a amostra. Inclui casos que devem ser recusados (formato não aceito, `.dat` faltando, arquivo inválido).

```bash
pip install wfdb pyedflib numpy
python testes/gerar_casos.py
node testes/testar_leitores.mjs     # Node 18 ou mais novo
```

Com registros reais do PhysioNet, a leitura também coincidiu amostra a amostra com a das bibliotecas: `sel30` da QTDB
(WFDB formato 212, 2 derivações, 15 min) e `r01.edf` da ADFECGDB (EDF+, 5 canais a 1000 Hz, com anotações). Para repetir
com outros arquivos, gere um `casos.json` no mesmo formato e rode `node testes/testar_leitores.mjs <pasta>`.

## Dados

Os sinais **não são redistribuídos**. Os cadernos baixam do PhysioNet:

| Base | Uso | Endereço |
| --- | --- | --- |
| Sudden Cardiac Death Holter Database (SDDB) | pacientes | https://physionet.org/content/sddb/1.0.0/ |
| MIT-BIH Normal Sinus Rhythm Database (NSRDB) | controles | https://physionet.org/content/nsrdb/1.0.0/ |
| QT Database (QTDB) | validação da delineação | https://physionet.org/content/qtdb/1.0.0/ |

Cite as bases como pede cada página do PhysioNet, junto com Goldberger et al. (2000), *Circulation* 101(23):e215–e220.

## Como rodar

**No Google Colab (como foi feito):** coloque `codigo/` e `notebooks/` em `MyDrive/Pesquisa/TCC/` e rode os cadernos na ordem 10 → 15.

**No computador:**

```bash
pip install -r requirements.txt
export TCC_BASE="$(pwd)"        # os cadernos procuram a biblioteca em $TCC_BASE/codigo
jupyter lab notebooks/
```

As análises complementares rodam direto sobre `resultados/csv/janelas_final.csv`, sem baixar sinais:

```bash
python analises_complementares/gera.py
```

Os cadernos 13 a 15 baixam e processam 30 registros de Holter de 24 h; contam com algumas horas de execução.

## Decisões fixadas antes da comparação

- Detector: F1 de pelo menos 99 % fora e dentro da última hora, tolerância de ±150 ms.
- Delineação: desvio-padrão do erro dentro da tolerância do CSE (limite de 2σ).
- Comparação: efeito = mediana da última hora − mediana do basal (antes de −2 h), por registro; Mann–Whitney entre grupos,
  g de Hedges com intervalo por bootstrap, correção de Holm sobre as 11 medidas.

O que foi decidido depois de ver os dados (troca do detector, versão 2 da onda T, correção da coerência, análises
complementares) está declarado no texto do TCC.

## Como citar

Use o arquivo [`CITATION.cff`](CITATION.cff) (o GitHub mostra o botão "Cite this repository") ou o DOI do Zenodo.

## Licença

Código sob licença MIT ([LICENSE](LICENSE)). Os resultados em `resultados/` podem ser reutilizados com citação.

---

**English summary.** Code, results and a browser tool for an undergraduate thesis (Biomedical Engineering, UFPE, 2026).
The ECG distance-matrix QRS detector of Charron is evaluated on 24-h Holter recordings (it misses beats: F1 91.1 % / 87.8 %),
a single column of that matrix (distance to the isoelectric level) is used with the trapezium method to delineate J point,
T wave and P wave against the QT Database, and ST-segment and P-wave measures are compared between the last hour and
baseline in 12 SDDB patients and 18 NSRDB controls. One of 11 measures holds: ST level 60 ms after J in lead 2
(+0.029 mV in patients, 12/12; Hedges g = 1.17; Holm p = 0.010). Exploratory study; data are not redistributed.
