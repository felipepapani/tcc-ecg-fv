import sys, numpy as np, pandas as pd
from pathlib import Path; RAIZ=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(RAIZ/'codigo')); import tcc_lib as T
from scipy.stats import wilcoxon, mannwhitneyu
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
D=str(RAIZ/'resultados'/'csv')+'/'
j=pd.read_csv(D+'janelas_final.csv'); p=j[j.passed].copy()
def med(g,lo,hi,c='ST60_d2',nmin=3):
    x=g[(g.min_ate_evento>lo)&(g.min_ate_evento<=hi)][c].dropna(); return x.median() if len(x)>=nmin else np.nan
rows=[]
for (b,r),g in p.groupby(['base','registro']):
    d=dict(base=b,registro=r)
    for c in ('ST60_d2','ST60_d1'):
        t=med(g,-60,0,c); m=med(g,-180,-120,c); m2=med(g,-120,-60,c); e=med(g,-1e9,-180,c); bas=med(g,-1e9,-120,c)
        d.update({c+'_tardio':t-m, c+'_precoce':m-e, c+'_pen':m2-bas, c+'_princ':t-bas})
    # modelo dentro do registro
    gb=g[g.min_ate_evento<=-120].dropna(subset=['ST60_d2','FC','ruido']); gt=g[(g.min_ate_evento>-60)].dropna(subset=['ST60_d2','FC','ruido'])
    if len(gb)>=12 and len(gt)>=3:
        X=np.c_[np.ones(len(gb)),gb.FC,gb.ruido]; beta=np.linalg.lstsq(X,gb.ST60_d2,rcond=None)[0]
        res=gt.ST60_d2-np.c_[np.ones(len(gt)),gt.FC,gt.ruido]@beta; d['resid']=np.median(res)
    rows.append(d)
q=pd.DataFrame(rows)
def rel(c,txt):
    S=q[q.base=='SDDB'][c].dropna(); N=q[q.base=='NSRDB'][c].dropna()
    h=T.hedges_g_boot(S,N)
    print(f"{txt:42s} SDDB n={len(S)} med={S.median():+.3f} pos={int((S>0).sum())} pW={wilcoxon(S).pvalue if len(S)>=6 else np.nan:.4f} | NSRDB n={len(N)} med={N.median():+.3f} pW={wilcoxon(N).pvalue:.3f} | g={h['g']:.2f} [{h['ic_lo']:.2f};{h['ic_hi']:.2f}] pMW={h['p_mw']:.4f}")
for c in ('ST60_d2','ST60_d1'):
    rel(c+'_princ','principal (ult. hora - basal<-2h) '+c)
    rel(c+'_tardio','tardio (ult. hora - [-3h,-2h]) '+c)
    rel(c+'_pen','placebo ([-2h,-1h] - basal<-2h) '+c)
    rel(c+'_precoce','precoce ([-3h,-2h] - antes de -3h) '+c)
rel('resid','residuo do modelo FC+ruido no basal')
print(q[q.base=='SDDB'].round(3).to_string())
q.to_csv('placebos.csv',index=False)
# Cliff delta e retirada de controles
e=T.efeito_ultima_hora(j,-120,(-60,0),['ST60_d2']); S=e[e.base=='SDDB'].ST60_d2_delta.dropna().values; N=e[e.base=='NSRDB'].ST60_d2_delta.dropna().values
cl=np.mean([np.sign(a-b) for a in S for b in N]); print('Cliff',round(cl,3),'P(sup)',round(np.mean([a>b for a in S for b in N]),3))
gs=[T.hedges_g_boot(S,np.delete(N,i),n_boot=2000) for i in range(len(N))]; print('LOO controles g',min(x['g'] for x in gs),max(x['g'] for x in gs),'ic_lo min',min(x['ic_lo'] for x in gs))
gs=[T.hedges_g_boot(np.delete(S,i),N,n_boot=2000) for i in range(len(S))]; print('LOO pacientes g',min(x['g'] for x in gs),max(x['g'] for x in gs),'ic_lo min',min(x['ic_lo'] for x in gs))
rng=np.random.default_rng(0)
for nm,v in (('SDDB',S),('NSRDB',N)):
    b=[np.median(rng.choice(v,len(v))) for _ in range(10000)]; print(nm,'mediana',np.median(v).round(4),'IC',np.percentile(b,[2.5,97.5]).round(4),'min',v.min().round(4),'ordenado',np.sort(v).round(3))
d=pd.read_csv(D+'qrs_diagnostico_perdidos.csv'); print(len(d)); print(d.perdido.value_counts().head()); print((d.dono_do_maximo.value_counts(normalize=True)*100).round(1)); print(d.razao.describe().round(3)); print(d.registro.nunique())
s=pd.read_csv(D+'sensibilidade.csv'); print(s.pivot(index='config',columns='medida',values='g').round(2)); print(s.pivot(index='config',columns='medida',values='ic_lo').round(2))
print(pd.read_csv(D+'qrs_erro_rr.csv').to_string())
