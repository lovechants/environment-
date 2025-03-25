## Testing KAN layers with event sequencing. 

[Code](https://github.com/KindXiaoming/pykan) adapated from PyKAN (original paper)

---

## Files 

`vis_util` - some visual utility

`model`    - where the model + layers are 

`main`     - driver script

---

## Example commands

```bash
python main.py --model_type=multiscale   

python main.py --model_type=multiscale --batch_size=50 --hidden_dim=128 --num_layers=5 --spline_points=8 --embedding_dim=64 --spline_order=4

python main.py --hidden_dim=128 --num_layers=5 --spline_points=8
```

---

## Sample Output 

```
Sequence 455, Prefix length 11:
  Prefix: ['E', 'A', 'B', 'D', 'C', 'E', 'B', 'C', 'E', 'E', 'B']
  Predicted next: C (p=0.4155)
  Actual next: C
  Correct: True
  Top 3 probabilities:
    C: 0.4155
    E: 0.2659
    D: 0.1432

Sequence 117:
  Original: ['E', 'E', 'C', 'A', 'C', 'C', 'E', 'A', 'C', 'B', 'C', 'E', 'C', 'C', 'C']
  Prefix: ['E', 'E', 'C', 'A', 'C']
  Generated: ['E', 'E', 'C', 'A', 'C', 'D', 'E', 'C', 'A', 'D', 'C', 'E', 'E', 'E', 'C']
  Accuracy: 0.4000
```

---

## TODO 

- [ ] Model Code Optimization 
- [ ] Hyper Parameter Tuning 
- [ ] Clean up code 
- [ ] Output Results 
- [ ] Interpretability Notes 
