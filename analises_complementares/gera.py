import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path; RAIZ=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(RAIZ/'codigo')); import tcc_lib as T
from scipy.stats import wilcoxon
T.estilo_figuras(base=15)
D=str(RAIZ/'resultados'/'csv')+'/'; F=str(RAIZ/'resultados'/'figuras')+'/'
j=pd.read_csv(D+'janelas_final.csv'); p=j[j.passed].copy()
n=lambda x,k=3: T.num(x,k)
def med(g,lo,hi,c,nmin=3):
    x=g[(g.min_ate_evento>lo)&(g.min_ate_evento<=hi)][c].dropna(); return x.median() if len(x)>=nmin else np.nan
# 1. tabela por registro
rows=[]
for (b,r),g in j.groupby(['base','registro']):
    gp=g[g.passed]; bas=gp[gp.min_ate_evento<=-120]; ter=gp[(gp.min_ate_evento>-60)]
    ok=len(bas)>=3 and len(ter)>=3
    rows.append({'base':b,'registro':r,'horas de registro antes do evento':round(-g.min_ate_evento.min()/60,1),'janelas':len(g),'% aprovadas':round(100*g.passed.mean()),
      'janelas no basal':len(bas),'janelas na última hora':len(ter),'ST60 d2 basal (mV)':bas.ST60_d2.median(),'ST60 d2 última hora (mV)':ter.ST60_d2.median() if len(ter) else np.nan,
      'variação (mV)':(ter.ST60_d2.median()-bas.ST60_d2.median()) if ok else np.nan,'FC basal (bpm)':bas.FC.median(),'FC última hora (bpm)':ter.FC.median() if len(ter) else np.nan,'entra na análise':'sim' if ok else 'não'})
pr=pd.DataFrame(rows); pr.to_csv(D+'compl_por_registro.csv',index=False)
# 2. contrastes
q=[]
for (b,r),g in p.groupby(['base','registro']):
    d=dict(base=b,registro=r)
    for c in ('ST60_d2','ST60_d1','ST60m_d2','ST60m_d1'):
        t=med(g,-60,0,c); m=med(g,-180,-120,c); m2=med(g,-120,-60,c); e=med(g,-1e9,-180,c); bas=med(g,-1e9,-120,c)
        d.update({c+'|última hora − basal (antes de −2 h)':t-bas, c+'|última hora − faixa de −3 a −2 h':t-m, c+'|faixa de −2 a −1 h − basal (antes de −2 h)':m2-bas, c+'|faixa de −3 a −2 h − antes de −3 h':m-e})
    gb=g[g.min_ate_evento<=-120].dropna(subset=['ST60_d2','FC','ruido']); gt=g[g.min_ate_evento>-60].dropna(subset=['ST60_d2','FC','ruido'])
    if len(gb)>=12 and len(gt)>=3:
        beta=np.linalg.lstsq(np.c_[np.ones(len(gb)),gb.FC,gb.ruido],gb.ST60_d2,rcond=None)[0]
        d['ST60_d2|resíduo na última hora do modelo ajustado no basal (FC e ruído)']=np.median(gt.ST60_d2-np.c_[np.ones(len(gt)),gt.FC,gt.ruido]@beta)
    q.append(d)
q=pd.DataFrame(q); NOME={'ST60_d2':'ST60 na derivação 2 (spline)','ST60_d1':'ST60 na derivação 1 (spline)','ST60m_d2':'ST60 na derivação 2 (linha de base por filtro de mediana)','ST60m_d1':'ST60 na derivação 1 (linha de base por filtro de mediana)'}
out=[]
for c in [x for x in q.columns if '|' in x]:
    med_,contr=c.split('|')
    if med_.startswith('ST60m') and 'basal (antes' not in contr.split(' − ')[1]+'' and not contr.startswith('última hora − basal'): continue
    if med_.startswith('ST60m') and not contr.startswith('última hora − basal'): continue
    S=q[q.base=='SDDB'][c].dropna(); N=q[q.base=='NSRDB'][c].dropna(); h=T.hedges_g_boot(S,N)
    out.append({'medida':NOME[med_],'contraste':contr,'n SDDB':len(S),'mediana SDDB (mV)':n(S.median()),'positivos SDDB':f'{int((S>0).sum())}/{len(S)}','p Wilcoxon SDDB':n(wilcoxon(S).pvalue,4) if len(S)>=6 else '',
       'n NSRDB':len(N),'mediana NSRDB (mV)':n(N.median()),'positivos NSRDB':f'{int((N>0).sum())}/{len(N)}','p Wilcoxon NSRDB':n(wilcoxon(N).pvalue,3),'g':n(h['g'],2),'IC 95 % do g':T.ic(h['ic_lo'],h['ic_hi'],2),'p Mann–Whitney':n(h['p_mw'],4)})
pd.DataFrame(out).to_csv(D+'compl_contrastes.csv',index=False); print(pd.DataFrame(out).to_string())
# 3. robustez extra do ST60 d2
e=T.efeito_ultima_hora(j,-120,(-60,0),['ST60_d2']); S=e[e.base=='SDDB'].ST60_d2_delta.dropna().values; N=e[e.base=='NSRDB'].ST60_d2_delta.dropna().values
rng=np.random.default_rng(0); ex=[]
ex.append(('Delta de Cliff (SDDB × NSRDB)',n(np.mean([np.sign(a-b) for a in S for b in N]),2)))
ex.append(('Probabilidade de um paciente variar mais que um controle',n(100*np.mean([a>b for a in S for b in N]),1)+' %'))
gs=[T.hedges_g_boot(S,np.delete(N,i),n_boot=2000) for i in range(len(N))]; ex.append(('Retirando um controle por vez: g mínimo a máximo; menor limite inferior do IC',f"{n(min(x['g'] for x in gs),2)} a {n(max(x['g'] for x in gs),2)}; {n(min(x['ic_lo'] for x in gs),2)}"))
gs=[T.hedges_g_boot(np.delete(S,i),N,n_boot=2000) for i in range(len(S))]; ex.append(('Retirando um paciente por vez (12 retiradas): g mínimo a máximo; menor limite inferior do IC',f"{n(min(x['g'] for x in gs),2)} a {n(max(x['g'] for x in gs),2)}; {n(min(x['ic_lo'] for x in gs),2)}"))
for nm,v in (('SDDB',S),('NSRDB',N)):
    b=[np.median(rng.choice(v,len(v))) for _ in range(10000)]; lo,hi=np.percentile(b,[2.5,97.5]); ex.append((f'Mediana da variação, {nm}, com IC 95 % por bootstrap (mV)',f'{n(np.median(v))} {T.ic(lo,hi,3)}'))
    ex.append((f'Menor e maior variação, {nm} (mV)',f'{n(v.min())} e {n(v.max())}'))
ex.append(('Registro da SDDB fora da análise','46 (nenhuma janela aprovada na última hora; 41 % de janelas aprovadas no registro)'))
d=pd.read_csv(D+'qrs_diagnostico_perdidos.csv'); vc=d.dono_do_maximo.value_counts(normalize=True)*100
ex.append(('Diagnóstico dos batimentos perdidos pelo método de Charron: total analisado (7 registros anotados)',str(len(d))))
for k,v in vc.items(): ex.append((f'Batimentos perdidos em janelas cujo máximo da soma das colunas pertence a: {k}',n(v,1)+' %'))
pd.DataFrame(ex,columns=['item','valor']).to_csv(D+'compl_robustez.csv',index=False); print(pd.DataFrame(ex,columns=['item','valor']).to_string())
# 4. figura: trajetórias individuais (variação em relação ao basal, por hora, registros com >= 6 h... todos)
fig,axs=plt.subplots(1,2,figsize=(14,5.6),sharey=True)
for a,(b,cor,tit) in zip(axs,[('NSRDB','tab:blue','Controles (NSRDB)'),('SDDB','tab:red','Antes da FV (SDDB)')]):
    M=[]
    for r,g in p[p.base==b].groupby('registro'):
        bas=med(g,-1e9,-120,'ST60_d2')
        if np.isnan(bas) or np.isnan(med(g,-60,0,'ST60_d2')): continue
        hs=[(h,med(g,60*h,60*(h+1),'ST60_d2')-bas) for h in range(-6,0)]; hs=[(h+0.5,v) for h,v in hs if not np.isnan(v)]
        a.plot(*zip(*hs),'-o',color=cor,alpha=0.45,lw=1.5,ms=4); M.append(dict(hs))
    hh=[h+0.5 for h in range(-6,0)]; mm=[np.median([m[h] for m in M if h in m]) for h in hh]
    a.plot(hh,mm,'-o',color='k',lw=3,ms=7,label='Mediana entre registros'); a.axhline(0,color='0.5',ls=':')
    a.set_title(tit); a.set_xlabel('Horas até o evento'); T.negrito_ticks(a)
axs[0].set_ylabel('ST60 na derivação 2 menos o basal\ndo próprio registro (mV)'); axs[0].legend(loc='upper left')
plt.tight_layout(); plt.savefig(F+'fig_compl_trajetorias_individuais.png',dpi=170); plt.close()
