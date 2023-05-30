# Mask2Former_hpo

Mask2Former: https://github.com/facebookresearch/Mask2Former

## Example

```bash
nohup python hpo.py --retrain_infer --scale base --space lr --surrogate_type gp --acq_optimizer_type random_scipy --num-gpus 8 --max_runs 25 >/dev/null 2>&1 &
```
