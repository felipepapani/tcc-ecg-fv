# Contexto para o Claude Code

Repositório público do TCC de Felipe Garcia Papani (Engenharia Biomédica, UFPE; defesa em dezembro de 2026;
orientador Prof. Fernando José Ribeiro Sales). O texto do TCC (LaTeX) fica no repositório privado `tcc-ecg-fv-texto`.

## Como trabalhar com o Felipe

- Responder em português, com profundidade, com analogias e explicando o porquê.
- Nunca inventar resultado. Todo número vem de `resultados/csv/` ou de rodar o código. Se algo não foi testado, dizer.
- Figuras em português do Brasil, com vírgula decimal (`T.estilo_figuras()` já configura).

## Mapa do código

- `codigo/tcc_lib.py` concentra tudo. Funções principais: detecção pelo método de Charron e variantes (`detectar_qrs_matriz`, `detectar_qrs_matriz_movel`); delineação
  pela distância ao nível isoelétrico (`d(n)`) com o método do trapézio (`delinear_batimentos`, versões 1 e 2 da onda T; `delinear_p_st` para a onda P); medidas por
  janela (`medidas_janela`, `tabela_janelas`, `coerencia(..., sem_tendencia=True)`); estatística e figuras.
  `MEDIDAS_ST_P_FINAL` lista as 11 medidas da família de Holm.
- Os cadernos são gerados por `codigo/gerar_nbXX.py`. Para mudar um caderno, edite o gerador e gere de novo.
- Os cadernos procuram a biblioteca em `$TCC_BASE/codigo` (no Colab, `MyDrive/Pesquisa/TCC`).
- `analises_complementares/*.py` rodam só sobre `resultados/csv/janelas_final.csv` e regravam os `compl_*.csv`
  e a figura de trajetórias individuais. Foram conferidos: reproduzem os arquivos atuais byte a byte.

## Números de referência (não mudar sem rodar de novo)

- Detecção (±150 ms, fora | última hora): Charron F1 91,1 | 87,8; melhor variante 96,7 | 93,0; xqrs 99,5 | 99,1.
- Delineação (DP do erro, 18 registros não vistos): J 20,0 ms; fim da T 66,4 ms; P início 43,2, P fim 49,2 (todos os registros).
- ST60 na derivação 2: SDDB +0,029 mV (12 de 12), NSRDB +0,004; g = 1,17 [0,66; 1,86]; p de Holm = 0,010.
- QTDB: 24 registros (sel30–sel52 da SDDB + sel17152); 6 olhados no desenho da versão 2, 18 não vistos.

## Pendências deste repositório

- Gerar o DOI no Zenodo (GitHub → release) e colocar no README e no texto do TCC (metodos.tex, marcador PENDENTE).
- A página web (`docs/index.html`) agora usa a versão 2 da onda T por padrão (v1 selecionável), conferida contra
  `tcc_lib.py` por `testes/testar_delineacao.mjs`. Falta mudar no texto do TCC as frases que dizem que a página usa a versão 1.
- Opcional: repetir o sorteio dos pseudo-eventos muitas vezes; reamostrar a SDDB para 128 Hz.
