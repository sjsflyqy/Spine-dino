"""Rebuild aggregate tables from per-tensor statistics, excluding non-parameter buffers."""
import csv,math,collections
from pathlib import Path
P=Path(__file__).resolve().parent
def read(name):return list(csv.DictReader((P/name).open(encoding='utf-8-sig')))
def write(name,rr):
 with (P/name).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def aggregate(rr):
 n=sum(int(x['elements']) for x in rr);s0=sum(float(x['reference_l2'])**2 for x in rr);sd=sum(float(x['delta_l2'])**2 for x in rr)
 s1=sum((float(x['norm_ratio'])*float(x['reference_l2']))**2 if x['norm_ratio'] else float(x['delta_l2'])**2 for x in rr)
 dot=sum(float(x['cosine'])*float(x['norm_ratio'])*float(x['reference_l2'])**2 if x['cosine'] else 0. for x in rr)
 return dict(elements=n,exact_changed_percent=sum(float(x['exact_changed_percent'])*int(x['elements']) for x in rr)/n,reference_l2=math.sqrt(s0),delta_l2=math.sqrt(sd),relative_l2_percent=100*math.sqrt(sd/s0) if s0 else None,cosine=dot/math.sqrt(s0*s1) if s0*s1 else None,norm_ratio=math.sqrt(s1/s0) if s0 else None,mean_abs_delta=sum(float(x['mean_abs_delta'])*int(x['elements']) for x in rr)/n,rms_delta=math.sqrt(sd/n),max_abs_delta=max(float(x['max_abs_delta']) for x in rr))
tensors=[x for x in read('tensors.csv') if not x['tensor'].endswith('.bias_mask') and not x['tensor'].startswith('rope_embed.')]
summary=read('summary.csv');layers=[]
for r in summary:
 rr=[x for x in tensors if x['model']==r['model'] and x['checkpoint']==r['checkpoint']];r.update(aggregate(rr));groups=collections.defaultdict(list)
 for t in rr:
  k=t['tensor'];g='.'.join(k.split('.')[:2]) if k.startswith('blocks.') else k.split('.')[0];groups[g].append(t)
 for g,tt in groups.items():
  a=aggregate(tt);layers.append({'model':r['model'],'checkpoint':r['checkpoint'],'group':g,**a,'share_of_total_squared_delta_percent':100*a['delta_l2']**2/r['delta_l2']**2})
 assert abs((r['relative_l2_percent']/100)**2-(1+r['norm_ratio']**2-2*r['norm_ratio']*r['cosine']))<1e-10
write('summary.csv',summary);write('layers.csv',layers);write('tensors.csv',tensors)
lines=['# 预训练teacher相对各自原始权重的参数变化','', '相对L2 = 100 × ||θ_teacher − θ_original||₂ / ||θ_original||₂。仅backbone参数；排除RoPE、attention bias_mask等buffer。','', '| 模型 | checkpoint | 约数据遍数 | 相对L2变化% | 参数余弦 | 新/旧范数 | 精确变化元素% |','|---|---|---:|---:|---:|---:|---:|']
for r in summary:lines.append(f"| {r['model']} | {r['checkpoint']} | {float(r['approx_data_passes']):.2f} | {r['relative_l2_percent']:.4f} | {r['cosine']:.6f} | {r['norm_ratio']:.4f} | {r['exact_changed_percent']:.4f} |")
(P/'results.md').write_text('\n'.join(lines)+'\n')
print('Verified',len(summary),'checkpoints; aggregate norm/cosine identities passed.')
for model in dict.fromkeys(r['model'] for r in summary):
 r=[r for r in summary if r['model']==model][-1];print('FINAL',r)
 ll=[x for x in layers if x['model']==model and x['checkpoint']==r['checkpoint']];print('TOP REL',sorted([(x['group'],x['relative_l2_percent'],x['share_of_total_squared_delta_percent']) for x in ll],key=lambda x:x[1] or 0,reverse=True)[:5])
