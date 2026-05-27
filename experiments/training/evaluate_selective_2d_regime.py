import importlib.util, numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
spec=importlib.util.spec_from_file_location('erc','experiments/training/evaluate_regime_calibration.py')
erc=importlib.util.module_from_spec(spec); spec.loader.exec_module(erc)
ytr,yv,ptr,pv=erc.load_predictions(); dtr,dv=erc.load_dates(); mf=erc.build_market_features(dtr,dv,ytr,yv)
rvtr=mf['train']['vol10']; rvv=mf['val']['vol10']; qtr=mf['train']['q905']; qv=mf['val']['q905']
rve,mke,grid=erc.fit_2d_scales(ytr,ptr,rvtr,qtr); p2tr=erc.apply_2d_scales(ptr,rvtr,qtr,rve,mke,grid); p2v=erc.apply_2d_scales(pv,rvv,qv,rve,mke,grid)
# risk features
Xtr=np.column_stack([p2tr,np.abs(p2tr),rvtr,qtr,rvtr*qtr,np.abs(p2tr)/(rvtr+1e-6)]).astype('float32')
Xv=np.column_stack([p2v,np.abs(p2v),rvv,qv,rvv*qv,np.abs(p2v)/(rvv+1e-6)]).astype('float32')
errtr=np.abs(ytr-p2tr); tailtr=(errtr>.035).astype(int)
sc=StandardScaler(); log=LogisticRegression(max_iter=1000,class_weight='balanced',C=.5,random_state=43).fit(sc.fit_transform(Xtr),tailtr)
hgb=HistGradientBoostingClassifier(max_iter=120,learning_rate=.035,max_leaf_nodes=10,l2_regularization=.1,class_weight='balanced',random_state=43).fit(Xtr,tailtr)
rtr=.5*log.predict_proba(sc.transform(Xtr))[:,1]+.5*hgb.predict_proba(Xtr)[:,1]
rvpred=.5*log.predict_proba(sc.transform(Xv))[:,1]+.5*hgb.predict_proba(Xv)[:,1]
errv=np.abs(yv-p2v); tailv=(errv>.035).astype(int)
print('risk auc/ap',roc_auc_score(tailv,rvpred),average_precision_score(tailv,rvpred))
def robust(v): return float(np.quantile(np.abs(v),.5)+.5*np.quantile(np.abs(v),.9))
def rs(y,p): return 1-robust(y-p)/robust(y)
rows=[]
for cov in [.5,.6,.7,.75,.8,.85,.9,.95,1.0]:
    thr=np.quantile(rvpred,cov)
    keep=rvpred<=thr
    y=yv[keep]; p=p2v[keep]; e=np.abs(y-p)
    rows.append(dict(coverage=keep.mean(),rel_score=rs(y,p),q90=float(np.quantile(e,.9)),q95=float(np.quantile(e,.95)),share_gt035=float((e>.035).mean()),share_gt050=float((e>.05).mean()),DA=float((np.sign(y)==np.sign(p)).mean())))
df=pd.DataFrame(rows); print(df.to_string(index=False))
out=Path('data/processed/assets/data_info_vn/history/training_runs/reports/selective_2d_regime_20260527'); gold=Path('gold/vn_transition_pressure_20260512/plots/selective_2d_regime_20260527')
out.mkdir(parents=True,exist_ok=True); gold.mkdir(parents=True,exist_ok=True); df.to_csv(out/'coverage_metrics.csv',index=False); df.to_csv(gold/'coverage_metrics.csv',index=False)
# plot
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax1=plt.subplots(figsize=(10,5)); x=df.coverage*100
ax1.plot(x,df.q90*100,marker='o',label='Q90(|E|)',color='#1a73e8'); ax1.plot(x,df.q95*100,marker='o',label='Q95(|E|)',color='#174ea6')
ax1.axhline(3.5,color='red',ls='--',label='3.5% tail line'); ax1.set_xlabel('Coverage kept: lowest-risk predictions (%)'); ax1.set_ylabel('Abs error quantile (%)'); ax1.grid(alpha=.3)
ax2=ax1.twinx(); ax2.plot(x,df.rel_score,marker='s',color='#34a853',label='rel_score'); ax2.set_ylabel('rel_score',color='#34a853')
ln1,lb1=ax1.get_legend_handles_labels(); ln2,lb2=ax2.get_legend_handles_labels(); ax1.legend(ln1+ln2,lb1+lb2,loc='upper left')
ax1.set_title('Selective 2D-Regime Prediction: Tail Reduction vs Coverage',fontweight='bold')
fig.tight_layout(); fig.savefig(out/'selective_coverage.png',dpi=130,bbox_inches='tight'); fig.savefig(gold/'selective_coverage.png',dpi=130,bbox_inches='tight'); plt.close(fig)
# summary
(out/'summary.md').write_text('# Selective 2D-Regime Tail Model\n\nHoldout/test not used.\n\n'+df.round(6).to_markdown(index=False),encoding='utf-8')
(gold/'summary.md').write_text((out/'summary.md').read_text(),encoding='utf-8')
print({'out':str(out),'gold':str(gold)})
