"""Read-only experiment audit; outputs are written beside this script. Standard library only."""
from pathlib import Path
import csv, json, re, statistics
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
TASKS={7:'task07_keypoint_linear',8:'task08_keypoint_linear',9:'task09_linear'}
KEYS={7:['ap_epsilon_MAE','lat_epsilon_MAE','ap_epsilon_dec','lat_epsilon_dec'],8:['MAE_mm','MRE_mm','SDR_2mm_percent','SDR_3mm_percent','SDR_4mm_percent','SDR_5mm_percent'],9:['IDR','IRA']}
def read(p): return json.loads(p.read_text())
bad_lines=[]
def rows(p):
 result=[]
 for i,s in enumerate(p.read_text().splitlines(),1):
  if not s.strip(): continue
  try: result.append(json.loads(s))
  except json.JSONDecodeError as e: bad_lines.append({'file':str(p.relative_to(ROOT)),'line':i,'error':str(e)})
 return result
def writecsv(name, data):
 if not data: return
 fields=list(dict.fromkeys(k for d in data for k in d))
 with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
records=[]; index={}; curves=[]
for task,folder in TASKS.items():
 for mode in ['linear','lora']:
  for p in sorted((ROOT/'outputs/downstream'/folder.replace('linear',mode)).rglob('test_metrics.json')):
   d=read(p); sp=p.with_name('training_summary.json'); summary=read(sp) if sp.exists() else {}; val=summary.get('best_val_metrics',{})
   name=str(p.parent.relative_to(ROOT/'outputs/downstream'))
   r={'task':task,'run':name,'split':d.get('split'),'best_epoch':val.get('epoch'),'stopped_epoch':summary.get('stopped_epoch'),'max_epochs':summary.get('max_epochs')}
   r.update({k:d.get(k) for k in KEYS[task]}); records.append(r)
   index[name]=(p,d,val)
   mp=p.with_name('metrics.jsonl')
   if mp.exists():
    rr=[x for x in rows(mp) if 'train_loss' in x]
    for x in rr: curves.append({'task':task,'run':name,**{k:v for k,v in x.items() if not isinstance(v,(list,dict))}})
writecsv('all_test_metrics.csv',records);writecsv('downstream_curves.csv',curves)
# Match each numeric table row against complete metric vectors, allowing manual truncation to five decimals.
text=(ROOT/'docs/整个预训练过程.md').read_text();task=None; audit=[]
for lineno,line in enumerate(text.splitlines(),1):
 if re.search(r'#{3,}.*(?:任务|下游任务)\s*([789])',line): task=int(re.search(r'(?:任务|下游任务)\s*([789])',line).group(1))
 if not task or not line.startswith('|'): continue
 cells=[re.sub(r'<[^>]+>|\*','',s).strip() for s in line.strip().strip('|').split('|')]
 if len(cells)!=len(KEYS[task])+1:continue
 try: nums=[float(s) for s in cells[1:]]
 except ValueError:continue
 if cells[0] in ['DVFNet','LSLD-Net','VertClassific']:continue
 matches=[]
 for name,(p,d,v) in index.items():
  if not name.startswith(TASKS[task].replace('linear','')):continue
  for split,dd in [('test',d),('val',v)]:
   if all(isinstance(dd.get(k),(int,float)) and abs(dd[k]-n)<10**(-len(cell.split('.')[-1]))+1e-10 for k,n,cell in zip(KEYS[task],nums,cells[1:])):
    matches.append((name,split,d))
 if matches:
  for name,split,d in matches:
   audit.append({'line':lineno,'task':task,'label':cells[0],'matched_split':split,'run':name,'reported':json.dumps(nums),'correct_test':json.dumps([d[k] for k in KEYS[task]])})
 else:audit.append({'line':lineno,'task':task,'label':cells[0],'matched_split':'NO_FULL_VECTOR_MATCH','reported':json.dumps(nums)})
writecsv('manual_table_audit.csv',audit)
up=[]
for p in sorted((ROOT/'outputs/upstream/pretrain').glob('*/training_metrics.json')):
 rr=rows(p); dedup={x['iteration']:x for x in rr}; ordered=sorted(dedup.values(),key=lambda x:x['iteration'])
 for target in [1000,20000,75712,151424,227137,302849,378562]:
  if target>ordered[-1]['iteration']+20:continue
  window=[x for x in ordered if target-1000<=x['iteration']<=target]
  if not window:continue
  d={'run':p.parent.name,'target_step':target,'samples':len(window),'duplicate_iterations_in_file':len(rr)-len(dedup)}
  for k in ['total_loss','dino_local_crops_loss','dino_global_crops_loss','ibot_loss','koleo_loss','lr','wd','mom']:
   vv=[x[k] for x in window if isinstance(x.get(k),(int,float))]
   if vv:d[k]=statistics.fmean(vv)
  up.append(d)
writecsv('pretrain_window_means.csv',up)
writecsv('malformed_log_lines.csv',bad_lines)
lines=['# 原始 JSON 统一测试集指标','', '由 audit.py 自动生成；checkpoint 由验证集选取。run 名称保留原样，不能据目录名推断 epoch。','']
for task in TASKS:
 lines+=['## 任务 '+str(task),'','| run | '+' | '.join(KEYS[task])+' |','|---|'+'---:|'*len(KEYS[task])]
 for r in records:
  if r['task']==task:lines+=['| '+r['run']+' | '+' | '.join(f'{r[k]:.5f}' for k in KEYS[task])+' |']
 lines+=['']
(OUT/'corrected_test_tables.md').write_text('\n'.join(lines))
print('test runs',len(records),'table rows',len(audit),'val-matched',sum(x['matched_split']=='val' for x in audit),'unmatched',sum(x['matched_split']=='NO_FULL_VECTOR_MATCH' for x in audit))
for x in audit:
 if x['matched_split']=='NO_FULL_VECTOR_MATCH':print(x)
