"""Compare frozen teacher backbones to original initialization; CPU, no model execution."""
from pathlib import Path
import sys,json,csv,math,re,collections
import torch
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
torch.set_num_threads(2)
def load(p):return torch.load(ROOT/p,map_location='cpu',weights_only=False,mmap=True)
def identity(a,b):
 return {'missing':sorted(set(a)-set(b)),'extra':sorted(set(b)-set(a)),'unequal':[k for k in a if k in b and not torch.equal(a[k],b[k])],'tensor_count':len(a)}
def empty():return dict(n=0,changed=0,ss0=0.,ss1=0.,ssd=0.,dot=0.,sad=0.,max_abs=0.)
def add(a,b):
 for k in a:a[k]=max(a[k],b[k]) if k=='max_abs' else a[k]+b[k]
def finish(a):
 n=a['n'];return {'elements':n,'exact_changed_percent':100*a['changed']/n,'reference_l2':math.sqrt(a['ss0']),'delta_l2':math.sqrt(a['ssd']),'relative_l2_percent':100*math.sqrt(a['ssd']/a['ss0']) if a['ss0'] else None,'cosine':a['dot']/math.sqrt(a['ss0']*a['ss1']) if a['ss0']*a['ss1'] else None,'norm_ratio':math.sqrt(a['ss1']/a['ss0']) if a['ss0'] else None,'mean_abs_delta':a['sad']/n,'rms_delta':math.sqrt(a['ssd']/n),'max_abs_delta':a['max_abs']}
def write(name,rr):
 with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
configs=[('DINOv2','dinov2_vitb14_lvd142m_100pass_plan','dinov2_vitb14_dinov2_format.pth',None),('RAD-DINO','rad_dino_init_rad_crops_100pass_plan','rad_dino_dinov2_format.pth','spine_rad_dino_vitb14_teacher_20000.pth'),('MAIRA-2','rad_dino_maira2_init_rad_crops_100pass_plan','rad_dino_maira2_dinov2_format.pth','spine_rad_dino_maira2_vitb14_teacher_20000.pth'),('DINOv3','dinov3_vitb16_lvd1689m_100pass_plan','dinov3_vitb16_lvd1689m_training_init.pth','spine_dinov3_vitb16_teacher_20000.pth')]
summary=[];layers=[];tensors=[];checks=[]
for family,folder,initial,short in configs:
 raw=load('weights/initialization/'+initial);base=raw.get('model',raw.get('teacher'))
 if family=='DINOv2':original=load('weights/base/dinov2/dinov2_vitb14_pretrain.pth')
 elif family=='DINOv3':original=load('weights/base/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
 else:
  sys.path.insert(0,str(ROOT/'upstream/dinov2-main'))
  from load_rad_dino import convert_rad_dino_to_dinov2
  from safetensors.torch import load_file
  source='rad-dino-hf' if family=='RAD-DINO' else 'rad-dino-maira-2-hf'
  original=convert_rad_dino_to_dinov2(load_file(str(ROOT/'weights/base'/source/'model.safetensors')))
 check={'model':family,**identity(original,base)};checks.append(check);print('INITIALIZATION',check,flush=True)
 assert not check['missing'] and not check['extra'] and not check['unequal']
 paths=sorted((ROOT/'weights/pretrained'/folder).glob('*.pth'),key=lambda p:int(re.search(r'(\d+)\.pth',p.name).group(1)))
 if short:paths.insert(0,ROOT/'weights/pretrained'/short)
 for p in paths:
  state=load(p.relative_to(ROOT));assert set(base)==set(state),(family,p)
  totals=empty();groups=collections.defaultdict(empty);buffers=[]
  for k,a in base.items():
   b=state[k];assert a.shape==b.shape,(k,a.shape,b.shape)
   # DINOv3 RoPE periods and attention bias masks are buffers.
   if k.startswith('rope_embed.') or k.endswith('.bias_mask') or not a.is_floating_point():
    buffers.append({'key':k,'equal':torch.equal(a,b)});continue
   a=a.double();b=b.double();assert torch.isfinite(a).all() and torch.isfinite(b).all()
   d=b-a; z=dict(n=a.numel(),changed=(a!=b).sum().item(),ss0=(a*a).sum().item(),ss1=(b*b).sum().item(),ssd=(d*d).sum().item(),dot=(a*b).sum().item(),sad=d.abs().sum().item(),max_abs=d.abs().max().item())
   add(totals,z);parts=k.split('.');group='.'.join(parts[:2]) if k.startswith('blocks.') else parts[0];add(groups[group],z)
   tensors.append({'model':family,'checkpoint':p.name,'tensor':k,**finish(z)})
  step=int(re.search(r'(\d+)\.pth',p.name).group(1));batch=4 if '20000' in p.name else 8
  row={'model':family,'checkpoint':p.name,'weight_path':str(p.relative_to(ROOT)),'steps_in_filename':step,'global_batch':batch,'approx_data_passes':step*batch/60570,**finish(totals)};summary.append(row)
  for group,z in groups.items():layers.append({'model':family,'checkpoint':p.name,'group':group,**finish(z),'share_of_total_squared_delta_percent':100*z['ssd']/totals['ssd']})
  print('RESULT',family,p.name,'relative_l2_percent',round(row['relative_l2_percent'],4),'cosine',round(row['cosine'],6),'excluded_buffers',buffers,flush=True)
write('summary.csv',summary);write('layers.csv',layers);write('tensors.csv',tensors)
(OUT/'initialization_checks.json').write_text(json.dumps(checks,indent=2))
lines=['# 预训练teacher相对各自原始权重的参数变化','', '整体相对L2 = 100 × ||θ_teacher − θ_original||₂ / ||θ_original||₂。排除DINOv3 RoPE与attention bias_mask buffer；包含backbone参数，未包含SSL head。','', '| 模型 | checkpoint | 约数据遍数 | 相对L2变化% | 参数余弦 | 新/旧范数 | 精确变化元素% |','|---|---|---:|---:|---:|---:|---:|']
for r in summary:lines.append(f"| {r['model']} | {r['checkpoint']} | {r['approx_data_passes']:.2f} | {r['relative_l2_percent']:.4f} | {r['cosine']:.6f} | {r['norm_ratio']:.4f} | {r['exact_changed_percent']:.4f} |")
(OUT/'results.md').write_text('\n'.join(lines)+'\n')
