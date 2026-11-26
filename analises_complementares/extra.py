import sys, numpy as np, pandas as pd
from pathlib import Path; RAIZ=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(RAIZ/'codigo')); import tcc_lib as T
from scipy.stats import wilcoxon, mannwhitneyu
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',200)
D=str(RAIZ/'resultados'/'csv')+'/'
j=pd.read_csv(D+'janelas_final.csv'); p=j[j.passed]
# confere com efeitos_final
e=T.efeito_ultima_hora(j,-120,(-60,0),['ST60_d1','ST60_d2','ST60m_d1','ST60m_d2','FC','ruido','frac_valida','COER_ST_semtend'])
print(e.columns.tolist()[:12])
S,N=e[e.base=='SDDB'],e[e.base=='NSRDB']
print('g ST60_d2',T.hedges_g_boot(S.ST60_d2_delta,N.ST60_d2_delta))
print('g ST60m_d2 (mediana)',T.hedges_g_boot(S.ST60m_d2_delta,N.ST60m_d2_delta), 'wilc S',wilcoxon(S.ST60m_d2_delta.dropna()).pvalue,'N',wilcoxon(N.ST60m_d2_delta.dropna()).pvalue, S.ST60m_d2_delta.median(), N.ST60m_d2_delta.median())
print('g ST60m_d1',T.hedges_g_boot(S.ST60m_d1_delta,N.ST60m_d1_delta), S.ST60m_d1_delta.median())
# por registro
rows=[]
for (b,r),g in j.groupby(['base','registro']):
    gp=g[g.passed]
    bas=gp[gp.min_ate_evento<=-120]; ter=gp[(gp.min_ate_evento>-60)&(gp.min_ate_evento<=0)]
    rows.append(dict(base=b,registro=r,horas_antes=-g.min_ate_evento.min()/60,t_evento_h=(g.t_inicio_s.max()+300)/3600,n_jan=len(g),aprov=g.passed.mean(),n_basal=len(bas),n_term=len(ter),
      ST_basal=bas.ST60_d2.median(),ST_term=ter.ST60_d2.median(),FC_basal=bas.FC.median(),FC_term=ter.FC.median()))
pr=pd.DataFrame(rows); pr['delta']=np.where((pr.n_basal>=3)&(pr.n_term>=3),pr.ST_term-pr.ST_basal,np.nan)
print(pr.round(3).to_string())
pr.to_csv('por_registro.csv',index=False)
