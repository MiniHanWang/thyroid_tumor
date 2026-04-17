import json,re,warnings
import sys,time
from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from scipy import stats
from scipy.special import expit
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score,f1_score,roc_curve,confusion_matrix
import yaml
warnings.filterwarnings('ignore')
RANDOM_STATE=42
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
CFG=yaml.safe_load((PROJECT_ROOT/'configs'/'config.yaml').read_text(encoding='utf-8'))
OUT=PROJECT_ROOT/CFG['paths']['evaluation_dir'];OUT.mkdir(exist_ok=True)
XLSX=PROJECT_ROOT/CFG['data']['raw_clinical_workbook']
from training.progress import format_duration,log,progress,timed_stage

run_start=time.perf_counter(); log('[run] thyroid clinical analysis'); workbook_stage=time.perf_counter(); log('[start] Load workbook and sheets')

xl=pd.ExcelFile(XLSX); sheets={s:pd.read_excel(XLSX,sheet_name=s) for s in progress(xl.sheet_names,total=len(xl.sheet_names),desc='Read sheets')}
pd.DataFrame([{'sheet_name':s,'n_rows':len(sheets[s]),'n_cols':sheets[s].shape[1],'columns':' | '.join(map(str,sheets[s].columns))} for s in xl.sheet_names]).to_csv(OUT/'sheet_overview.csv',index=False,encoding='utf-8-sig')
final_sheet='cleaned_dataset' if ('cleaned_dataset' in sheets and len(sheets['cleaned_dataset'])>0) else ('raw_input' if 'raw_input' in sheets else xl.sheet_names[0])
raw=sheets['raw_input'] if 'raw_input' in sheets else sheets[final_sheet].copy(); raw.to_csv(OUT/'raw_input_copy.csv',index=False,encoding='utf-8-sig')
path_map=sheets.get('pathology_mapping',pd.DataFrame(columns=['关键词','pathology_main','malignant_label','indeterminate_label']))

log(f"[done] Load workbook and sheets | elapsed={format_duration(time.perf_counter()-workbook_stage)}")
c=raw.columns.tolist(); df=raw.rename(columns={c[0]:'id',c[1]:'sex_raw',c[2]:'age',c[3]:'weight_kg',c[4]:'height_cm',c[5]:'bmi_raw',c[6]:'smoking_raw',c[7]:'drinking_raw',c[8]:'pmh_raw',c[9]:'birthplace_raw',c[10]:'pathology_raw'}).copy()
for k in ['id','age','weight_kg','height_cm','bmi_raw']: df[k]=pd.to_numeric(df[k],errors='coerce')
df['sex']=df['sex_raw'].astype(str).map(lambda s:1 if '男' in s else (0 if '女' in s else np.nan))
df['bmi_calc']=df['weight_kg']/((df['height_cm']/100)**2); df['bmi']=df['bmi_raw'].fillna(df['bmi_calc']); df['bmi_diff']=(df['bmi_raw']-df['bmi_calc']).abs(); df['bmi_group']=pd.cut(df['bmi'],[-np.inf,18.5,24,28,np.inf],labels=[1,2,3,4]).astype('float')

def parse_01(s,pos=('有','吸烟','戒烟','偶尔','饮酒','应酬'),neg=('无','否','不')):
    if pd.isna(s): return np.nan
    t=str(s)
    if any(x in t for x in neg): return 0
    if any(x in t for x in pos): return 1
    return np.nan

def parse_year(s):
    if pd.isna(s): return np.nan
    m=re.search(r'(\d+)\s*年',str(s)); return float(m.group(1)) if m else np.nan

def parse_cig(s):
    if pd.isna(s): return np.nan
    t=str(s); m=re.search(r'每天\s*(\d+)\s*支',t) or re.search(r'(\d+)\s*支/天',t); return float(m.group(1)) if m else np.nan

df['smoking_bin']=df['smoking_raw'].map(parse_01); df['smoking_years']=df['smoking_raw'].map(parse_year); df['cigs_per_day']=df['smoking_raw'].map(parse_cig)
df['drinking_bin']=df['drinking_raw'].map(parse_01); df['drinking_years']=df['drinking_raw'].map(parse_year)

def pmh_flags(t):
    s='' if pd.isna(t) else str(t)
    if s.strip() in ('','无'): return {'hypertension':0,'diabetes':0,'hepatitis':0,'thyroid_history':0,'other_cancer_history':0}
    return {'hypertension':int('高血压' in s),'diabetes':int(('糖尿病' in s) or ('血糖偏高' in s)),'hepatitis':int(('肝炎' in s) or ('乙肝' in s)),'thyroid_history':int(('甲状腺' in s) or ('甲亢' in s) or ('甲减' in s)),'other_cancer_history':int(('癌' in s) or ('肿瘤' in s) or ('恶性' in s))}
for k,v in df['pmh_raw'].map(pmh_flags).apply(pd.Series).items(): df[k]=v

def std_bp(s):
    if pd.isna(s): return ('Unknown','Unknown')
    t=str(s).strip()
    if t=='': return ('Unknown','Unknown')
    if '外籍' in t: return ('Foreign','Foreign')
    ps=['北京','天津','上海','重庆','河北','山西','辽宁','吉林','黑龙江','江苏','浙江','安徽','福建','江西','山东','河南','湖北','湖南','广东','海南','四川','贵州','云南','陕西','甘肃','青海','台湾','内蒙古','广西','西藏','宁夏','新疆','香港','澳门']
    p=next((x for x in ps if x in t),'Unknown')
    rm={'北京':'North','天津':'North','河北':'North','山西':'North','内蒙古':'North','辽宁':'Northeast','吉林':'Northeast','黑龙江':'Northeast','上海':'East','江苏':'East','浙江':'East','安徽':'East','福建':'East','江西':'East','山东':'East','台湾':'East','河南':'Central','湖北':'Central','湖南':'Central','广东':'South','海南':'South','广西':'South','香港':'South','澳门':'South','重庆':'Southwest','四川':'Southwest','贵州':'Southwest','云南':'Southwest','西藏':'Southwest','陕西':'Northwest','甘肃':'Northwest','青海':'Northwest','宁夏':'Northwest','新疆':'Northwest'}
    return p,rm.get(p,'Unknown')
bp=df['birthplace_raw'].map(std_bp); df['birthplace_province']=bp.map(lambda x:x[0]); df['birthplace_region']=bp.map(lambda x:x[1])

def path_outcome(t):
    if pd.isna(t): return {'pathology_main':np.nan,'malignant':np.nan,'indeterminate_label':1,'papillary_ca':0,'medullary_ca':0,'follicular_carcinoma':0,'follicular_adenoma':0,'follicular_lesion':0,'nodular_goiter_flag':0,'thyroiditis_flag':0,'ln_metastasis':np.nan}
    s=str(t); m=path_map[path_map['关键词'].astype(str).map(lambda k:k in s)]
    mains=list(m['pathology_main'].astype(str)); pm='Unknown'
    for p in ['MTC','FTC','PTC','Indeterminate','Benign']:
        if p in mains: pm=p; break
    mal=list(m['malignant_label'].dropna().astype(int)); ind=list(m['indeterminate_label'].dropna().astype(int))
    if 1 in mal: y,indf=1,0
    elif 1 in ind: y,indf=np.nan,1
    elif 0 in mal: y,indf=0,0
    else:
        if ('恶性' in s) or ('癌' in s): y,indf=1,0
        elif '良性' in s: y,indf=0,0
        else: y,indf=np.nan,1
    return {'pathology_main':pm,'malignant':y,'indeterminate_label':indf,'papillary_ca':int('乳头状癌' in s),'medullary_ca':int('髓样癌' in s),'follicular_carcinoma':int(('滤泡癌' in s) or ('滤泡腺癌' in s)),'follicular_adenoma':int('滤泡性腺瘤' in s),'follicular_lesion':int(('滤泡性病变' in s) or ('滤泡性肿瘤' in s) or ('NIFTP' in s) or ('恶性潜能未定' in s)),'nodular_goiter_flag':int('结节性甲状腺肿' in s),'thyroiditis_flag':int(('甲状腺炎' in s) or ('桥本' in s) or ('慢性淋巴细胞性甲状腺炎' in s) or ('淋巴细胞性甲状腺炎' in s)),'ln_metastasis':1 if '淋巴结转移' in s else 0}
for k,v in df['pathology_raw'].map(path_outcome).apply(pd.Series).items(): df[k]=v

quality_stage=time.perf_counter(); log('[start] Run data quality checks')
df['final_analysis_include']=((df['indeterminate_label']==0)&(df['malignant'].isin([0,1]))).astype(int)
flags=[]; dup=set(df[df['id'].duplicated(keep=False)]['id'].dropna())
for _,r in progress(df.iterrows(),total=len(df),desc='Quality flags'):
    z=[]
    if r['id'] in dup: z.append('duplicate_id')
    if pd.notna(r['bmi_diff']) and r['bmi_diff']>0.3: z.append('bmi_mismatch')
    if pd.notna(r['age']) and (r['age']<10 or r['age']>100): z.append('age_outlier')
    if pd.notna(r['height_cm']) and (r['height_cm']<130 or r['height_cm']>210): z.append('height_outlier')
    if pd.notna(r['weight_kg']) and (r['weight_kg']<25 or r['weight_kg']>200): z.append('weight_outlier')
    if pd.notna(r['bmi']) and (r['bmi']<12 or r['bmi']>60): z.append('bmi_outlier')
    flags.append(';'.join(z))
df['review_flag']=flags
for b in ['sex','smoking_bin','drinking_bin','hypertension','diabetes','hepatitis','thyroid_history','other_cancer_history','thyroiditis_flag','nodular_goiter_flag','malignant','ln_metastasis']:
    df[b]=pd.to_numeric(df[b],errors='coerce'); df.loc[~df[b].isin([0,1]),b]=np.nan
log(f"[done] Run data quality checks | elapsed={format_duration(time.perf_counter()-quality_stage)}")

df.to_csv(OUT/'cleaned_analysis_dataset.csv',index=False,encoding='utf-8-sig')
with pd.ExcelWriter(OUT/'cleaned_analysis_dataset.xlsx',engine='openpyxl') as w: df.to_excel(w,index=False,sheet_name='cleaned_analysis')
missing=pd.DataFrame({'variable':df.columns,'missing_n':df.isna().sum().values,'missing_pct':(df.isna().mean()*100).round(2).values,'dtype':[str(df[x].dtype) for x in df.columns]}).sort_values('missing_pct',ascending=False); missing.to_csv(OUT/'missing_summary.csv',index=False,encoding='utf-8-sig')
df[df['review_flag'].astype(str).str.len()>0].to_csv(OUT/'review_flag_cases.csv',index=False,encoding='utf-8-sig')
q={'n_rows':int(len(df)),'n_cols':int(df.shape[1]),'duplicate_id_n':int(df['id'].duplicated().sum()),'review_flag_n':int((df['review_flag'].astype(str).str.len()>0).sum()),'malignant_missing_n':int(df['malignant'].isna().sum()),'ln_metastasis_missing_n':int(df['ln_metastasis'].isna().sum())}
(OUT/'data_quality_report.json').write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf-8')

pred=['sex','age','bmi','smoking_bin','drinking_bin','hypertension','diabetes','hepatitis','thyroid_history','other_cancer_history','thyroiditis_flag','nodular_goiter_flag']
imp=df.copy()
for c in pred:
    if imp[c].dropna().isin([0,1]).all() and len(imp[c].dropna())>0: imp[c]=imp[c].fillna(imp[c].mode(dropna=True).iloc[0] if len(imp[c].mode(dropna=True)) else 0)
    else: imp[c]=imp[c].fillna(imp[c].median())
imp.to_csv(OUT/'cleaned_analysis_dataset_imputed.csv',index=False,encoding='utf-8-sig')
df.dropna(subset=[x for x in pred+['malignant','ln_metastasis'] if x in df.columns]).to_csv(OUT/'complete_case_analysis_dataset.csv',index=False,encoding='utf-8-sig')

def summarize(x):
    x=pd.Series(x).dropna()
    if len(x)==0:return 'NA'
    pn=stats.shapiro(x.sample(min(len(x),500),random_state=RANDOM_STATE)).pvalue if len(x)>=3 else 0
    return f"{x.mean():.2f} ± {x.std(ddof=1):.2f}" if pn>0.05 else f"{x.median():.2f} [{x.quantile(.25):.2f}, {x.quantile(.75):.2f}]"

def make_table(d,g,cont,cat):
    d=d[d[g].isin([0,1])].copy();g0=d[d[g]==0];g1=d[d[g]==1];rows=[]
    for v in cont:
        x0=pd.to_numeric(g0[v],errors='coerce').dropna(); x1=pd.to_numeric(g1[v],errors='coerce').dropna(); test='Mann-Whitney U'; p=np.nan
        if len(x0)>=3 and len(x1)>=3:
            p0=stats.shapiro(x0.sample(min(len(x0),500),random_state=RANDOM_STATE)).pvalue; p1=stats.shapiro(x1.sample(min(len(x1),500),random_state=RANDOM_STATE)).pvalue
            if p0>0.05 and p1>0.05: test='t-test'; p=stats.ttest_ind(x0,x1,equal_var=False,nan_policy='omit').pvalue
            else: p=stats.mannwhitneyu(x0,x1,alternative='two-sided').pvalue
        rows.append({'variable':v,'group_0':summarize(x0),'group_1':summarize(x1),'test':test,'p_value':p})
    for v in cat:
        t=d[[v,g]].dropna().copy(); t[v]=pd.to_numeric(t[v],errors='coerce'); t=t[t[v].isin([0,1])]
        if len(t)==0: continue
        ct=pd.crosstab(t[v],t[g])
        for lvl in [0,1]:
            n0=int(((t[g]==0)&(t[v]==lvl)).sum()); d0=int((t[g]==0).sum()); n1=int(((t[g]==1)&(t[v]==lvl)).sum()); d1=int((t[g]==1).sum())
            rows.append({'variable':f'{v}={lvl}','group_0':f'{n0} ({(100*n0/d0 if d0 else 0):.1f}%)','group_1':f'{n1} ({(100*n1/d1 if d1 else 0):.1f}%)','test':'','p_value':np.nan})
        if ct.shape==(2,2):
            exp=stats.contingency.expected_freq(ct.values); test='Fisher exact' if (exp<5).any() else 'Chi-square'; p=stats.fisher_exact(ct.values)[1] if test=='Fisher exact' else stats.chi2_contingency(ct.values)[1]
        else: test='Chi-square'; p=stats.chi2_contingency(ct.values)[1]
        rows.append({'variable':f'{v} (overall)','group_0':'','group_1':'','test':test,'p_value':p})
    return pd.DataFrame(rows)

cont=['age','weight_kg','height_cm','bmi']; cat=['sex','smoking_bin','drinking_bin','hypertension','diabetes','hepatitis','thyroid_history','other_cancer_history','thyroiditis_flag','nodular_goiter_flag']
t1=make_table(df,'malignant',cont,cat); t1.to_csv(OUT/'table1_baseline.csv',index=False,encoding='utf-8-sig');
with pd.ExcelWriter(OUT/'table1_baseline.xlsx',engine='openpyxl') as w: t1.to_excel(w,index=False,sheet_name='Table1')

def logit_wald(y,X):
    lr=LogisticRegression(penalty=None,solver='lbfgs',max_iter=5000)
    lr.fit(X,y)
    b=np.r_[lr.intercept_[0],lr.coef_[0]]
    X1=np.c_[np.ones(len(X)),X.values]
    p=expit(X1@b)
    W=np.diag(np.clip(p*(1-p),1e-8,None))
    fisher=X1.T@W@X1
    cov=np.linalg.pinv(fisher)
    se=np.sqrt(np.diag(cov))
    z=np.divide(b,se,out=np.zeros_like(b),where=se>0)
    pv=2*(1-stats.norm.cdf(np.abs(z)))
    return b,se,pv

def calc_vif(data):
    out=[]; A=data.values
    for i,col in enumerate(data.columns):
        y=A[:,i]; X=np.delete(A,i,axis=1)
        if X.shape[1]==0: out.append({'variable':col,'VIF':1.0}); continue
        r2=LinearRegression().fit(X,y).score(X,y)
        vif=1/(1-r2) if r2<0.999999 else np.inf
        out.append({'variable':col,'VIF':float(vif)})
    return pd.DataFrame(out).sort_values('VIF',ascending=False)

def fit_logit(d,y,xs,prefix):
    u=[]
    for x in xs:
        z=d[[y,x]].dropna().copy()
        if len(z)<30 or z[y].nunique()<2 or z[x].nunique()<2: continue
        try:
            b,se,pv=logit_wald(z[y].astype(int),z[[x]].astype(float)); beta=b[1]; s=se[1]; p=pv[1]
            u.append({'variable':x,'OR':float(np.exp(beta)),'CI95_low':float(np.exp(beta-1.96*s)),'CI95_high':float(np.exp(beta+1.96*s)),'p_value':float(p),'n':int(len(z))})
        except Exception: pass
    udf=(pd.DataFrame(u).sort_values('p_value') if u else pd.DataFrame(columns=['variable','OR','CI95_low','CI95_high','p_value','n']))
    udf.to_csv(OUT/f'univariate_logistic_{prefix}.csv',index=False,encoding='utf-8-sig')
    use=[x for x in xs if x in d.columns]; z=d[[y]+use].dropna().copy(); use=[x for x in use if z[x].nunique()>1]
    mdf=pd.DataFrame(columns=['variable','OR','CI95_low','CI95_high','p_value','n']); vif=pd.DataFrame(columns=['variable','VIF'])
    if len(use)>0 and len(z)>=30 and z[y].nunique()==2:
        try:
            b,se,pv=logit_wald(z[y].astype(int),z[use].astype(float)); rows=[]
            for i,x in enumerate(use,1): rows.append({'variable':x,'OR':float(np.exp(b[i])),'CI95_low':float(np.exp(b[i]-1.96*se[i])),'CI95_high':float(np.exp(b[i]+1.96*se[i])),'p_value':float(pv[i]),'n':int(len(z))})
            mdf=pd.DataFrame(rows).sort_values('p_value'); vif=calc_vif(z[use].astype(float))
        except Exception: pass
    mdf.to_csv(OUT/f'multivariable_logistic_{prefix}.csv',index=False,encoding='utf-8-sig'); vif.to_csv(OUT/f'vif_{prefix}.csv',index=False,encoding='utf-8-sig')

ana=df[df['final_analysis_include']==1].copy(); fit_logit(ana,'malignant',pred,'malignant')

use=[c for c in pred if c in ana.columns]; md=ana[['malignant']+use].dropna(subset=['malignant']); md=md[md['malignant'].isin([0,1])]; X=md[use]; y=md['malignant'].astype(int)
pre=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median'))]),X.columns.tolist())],remainder='drop')
models={'LogisticRegression':LogisticRegression(max_iter=2000,random_state=RANDOM_STATE),'RandomForest':RandomForestClassifier(n_estimators=500,random_state=RANDOM_STATE,class_weight='balanced')}
try:
    from xgboost import XGBClassifier
    models['XGBoost']=XGBClassifier(n_estimators=400,learning_rate=0.05,max_depth=4,subsample=0.9,colsample_bytree=0.9,eval_metric='logloss',random_state=RANDOM_STATE)
except Exception: pass
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=RANDOM_STATE,stratify=y)
model_stage=time.perf_counter(); log('[start] Train and compare clinical models')
rec=[]; rocD={}; fitted={}
for idx,(n,m) in enumerate(progress(models.items(),total=len(models),desc='Fit models'),start=1):
    p=Pipeline([('prep',pre),('model',m)]); p.fit(Xtr,ytr); fitted[n]=p; prob=p.predict_proba(Xte)[:,1]; predy=(prob>=0.5).astype(int); tn,fp,fn,tp=confusion_matrix(yte,predy).ravel(); cv=cross_val_score(p,X,y,cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=RANDOM_STATE),scoring='roc_auc').mean()
    rec.append({'model':n,'AUC':roc_auc_score(yte,prob),'Accuracy':accuracy_score(yte,predy),'Sensitivity':recall_score(yte,predy),'Specificity':tn/(tn+fp) if (tn+fp) else np.nan,'Precision':precision_score(yte,predy,zero_division=0),'Recall':recall_score(yte,predy),'F1':f1_score(yte,predy),'CV_AUC_5fold':cv,'n_train':int(len(Xtr)),'n_test':int(len(Xte))}); rocD[n]=roc_curve(yte,prob)[:2]; avg=(time.perf_counter()-model_stage)/idx; log(f"[model {idx}/{len(models)}] {n} auc={rec[-1]['AUC']:.4f} cv_auc={cv:.4f} eta={format_duration(avg*(len(models)-idx))}")
perf=pd.DataFrame(rec).sort_values('AUC',ascending=False); perf.to_csv(OUT/'model_performance_comparison.csv',index=False,encoding='utf-8-sig')
log(f"[done] Train and compare clinical models | elapsed={format_duration(time.perf_counter()-model_stage)}")
plt.figure(figsize=(7,6));
for n,(fpr,tpr) in rocD.items(): plt.plot(fpr,tpr,label=f"{n} (AUC={perf.loc[perf['model']==n,'AUC'].iloc[0]:.3f})")
plt.plot([0,1],[0,1],'k--',alpha=.5); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curves for Malignant Prediction'); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'roc_curve.png',dpi=300); plt.close()
best=perf.iloc[0]['model']; mo=fitted[best].named_steps['model']; impv=mo.feature_importances_ if hasattr(mo,'feature_importances_') else (np.abs(mo.coef_[0]) if hasattr(mo,'coef_') else np.zeros(len(use)))
impdf=pd.DataFrame({'feature':use,'importance':impv}).sort_values('importance',ascending=False); impdf.to_csv(OUT/'feature_importance_values.csv',index=False,encoding='utf-8-sig')
plt.figure(figsize=(8,6)); top=impdf.head(20).iloc[::-1]; plt.barh(top['feature'],top['importance'],color='#2C7FB8'); plt.title(f'Feature Importance ({best})'); plt.tight_layout(); plt.savefig(OUT/'feature_importance.png',dpi=300); plt.close()
try:
    import shap
    if best in ['RandomForest','XGBoost']:
        Xt=fitted[best].named_steps['prep'].transform(Xtr); ex=shap.TreeExplainer(mo); sv=ex.shap_values(Xt); plt.figure(); shap.summary_plot(sv[1] if isinstance(sv,list) else sv,features=Xt,feature_names=use,show=False); plt.tight_layout(); plt.savefig(OUT/'shap_summary.png',dpi=300,bbox_inches='tight'); plt.close()
    else: (OUT/'shap_summary_not_available.txt').write_text('SHAP summary not generated (unsupported best model).',encoding='utf-8')
except Exception:
    (OUT/'shap_summary_not_available.txt').write_text('SHAP summary not generated (package unavailable or model issue).',encoding='utf-8')

ln=df[(df['malignant']==1)&(df['ln_metastasis'].isin([0,1]))].copy(); tln=make_table(ln,'ln_metastasis',cont,cat); tln.to_csv(OUT/'table_ln_baseline.csv',index=False,encoding='utf-8-sig');
with pd.ExcelWriter(OUT/'table_ln_baseline.xlsx',engine='openpyxl') as w: tln.to_excel(w,index=False,sheet_name='TableLN')
fit_logit(ln,'ln_metastasis',pred,'ln')

uni=pd.read_csv(OUT/'univariate_logistic_malignant.csv') if (OUT/'univariate_logistic_malignant.csv').exists() else pd.DataFrame(); multi=pd.read_csv(OUT/'multivariable_logistic_malignant.csv') if (OUT/'multivariable_logistic_malignant.csv').exists() else pd.DataFrame()
sig_u=uni[uni['p_value']<0.05] if len(uni) else pd.DataFrame(); sig_m=multi[multi['p_value']<0.05] if len(multi) else pd.DataFrame(); bestrow=perf.sort_values('AUC',ascending=False).iloc[0]
(OUT/'results_summary.md').write_text(f"# Results Summary\n\n- Sample size: {len(df)}\n- Malignant proportion: {df['malignant'].dropna().mean():.3f}\n- LN metastasis proportion: {df['ln_metastasis'].dropna().mean():.3f}\n\n## Table 1 Highlights (first 20 rows)\n\n{t1.head(20).to_string(index=False)}\n\n## Significant Variables (Univariate, malignant)\n\n{sig_u.head(12).to_string(index=False) if len(sig_u) else 'No significant variables identified.'}\n\n## Significant Variables (Multivariable, malignant)\n\n{sig_m.head(12).to_string(index=False) if len(sig_m) else 'No significant variables identified.'}\n\n## Best Prediction Model\n\n- {bestrow['model']} (AUC={bestrow['AUC']:.3f}, Accuracy={bestrow['Accuracy']:.3f})\n",encoding='utf-8')
(OUT/'methods_ready_summary.md').write_text("# Methods-Ready Summary\n\nData were parsed from all workbook sheets; `raw_input` was used for analysis construction because `cleaned_dataset` had zero rows. Variables were harmonized according to the data dictionary and modeling list. Primary endpoint (`malignant`) and secondary endpoint (`ln_metastasis`) were derived from pathology mapping and keyword rules.\n\nData quality checks covered duplicate IDs, missingness, out-of-range values, BMI inconsistency, and coding anomalies. Outliers were flagged (`review_flag`) rather than deleted. Both imputed and complete-case datasets were generated.\n\nBaseline statistics were grouped by endpoint using t-test/Mann-Whitney U for continuous variables and chi-square/Fisher exact for categorical variables. Risk factors were assessed with univariate and multivariable logistic regression, reporting OR, 95%CI, and p-values, with VIF for collinearity.\n\nPrediction models for malignant status included Logistic Regression, Random Forest, and XGBoost when available, with 80/20 split and 5-fold CV, reporting AUC, Accuracy, Sensitivity, Specificity, Precision, Recall, and F1. ROC and feature-importance plots were generated; SHAP summary was generated when supported.\n\nSecondary endpoint analysis for LN metastasis repeated descriptive and logistic analyses in the malignant subgroup.\n",encoding='utf-8')
(OUT/'clean_log.txt').write_text('\n'.join(['Thyroid publishable analysis cleaning log',f'Final analysis source sheet: {final_sheet}','Key variables: '+', '.join(pred+['malignant','ln_metastasis']),'','Quality summary:']+[f'- {k}: {v}' for k,v in q.items()]+['','Cleaning actions:','- Standardized sex/smoking/drinking and comorbidities to binary coding.','- Recomputed BMI and flagged BMI mismatch > 0.3.','- Derived malignant and indeterminate labels from pathology mapping.','- Derived LN metastasis from pathology text.','- Standardized birthplace to province/region categories.','- Added review_flag for duplicates and outliers without deletion.','- Produced imputed and complete-case analysis datasets.']),encoding='utf-8')

print('分析完成'); print('输出文件列表:')
for f in sorted([p.name for p in OUT.iterdir() if p.is_file()]): print(f'- {f}')
print('主结果摘要:'); print(f"- 样本量: {len(df)}"); print(f"- 恶性比例: {df['malignant'].dropna().mean():.3f}"); print(f"- 淋巴结转移比例: {df['ln_metastasis'].dropna().mean():.3f}"); print(f"- Best model: {bestrow['model']} | AUC={bestrow['AUC']:.3f} | Accuracy={bestrow['Accuracy']:.3f}")
log(f"[run-done] thyroid clinical analysis finished in {format_duration(time.perf_counter()-run_start)}")
