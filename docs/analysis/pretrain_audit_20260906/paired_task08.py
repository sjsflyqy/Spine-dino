"""Patient-cluster paired bootstrap of task08 MRE; uncertainty is conditional on trained checkpoints."""
import json,re,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
def grouped(p):
 d={}
 for x in json.loads(p.read_text()):
  key=re.sub(r'^(gq|gs)','',x['id'],flags=re.I);d.setdefault(key,[]).extend(x['radial_error_mm'])
 return {k:(sum(v),len(v)) for k,v in d.items()}
base=ROOT/'outputs/downstream/task08_keypoint_linear';a=grouped(base/'dinov2/test_predictions.json');out=[]
for p in sorted((base/'dinov2_vitb14_lvd142m_100pass_plan').rglob('test_predictions.json')):
 b=grouped(p);assert a.keys()==b.keys(); keys=sorted(a);rng=random.Random(42);values=[]
 def delta(kk):return sum(b[k][0]-a[k][0] for k in kk)/sum(a[k][1] for k in kk)
 for _ in range(2000):values.append(delta(rng.choices(keys,k=len(keys))))
 values.sort();out.append({'run':p.parent.name,'patients':len(keys),'delta_MRE_mm_new_minus_official':delta(keys),'paired_cluster_bootstrap_95_percent':[values[50],values[1949]]})
(OUT/'task08_paired_bootstrap.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
