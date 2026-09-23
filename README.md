# COMP60272 AI Security Coursework

Adversarial robustness exercises against `mnist-net_256x2`, a 2-hidden-layer
(256 units each) MLP classifier for MNIST digits, exported as ONNX.

## Files

| File | Description |
| --- | --- |
| [code_1.py](code_1.py) | Exercise 1 — finds a minimal L∞ adversarial example for `mnist_img_9.csv` using an FGSM gradient direction followed by a binary search on the step size, decided against the `onnxruntime` backend. Writes `solution_1.csv`. |
| [code_2.py](code_2.py) | Exercise 2 — certifies/falsifies robustness at increasing L∞ radii by generating `.vnnlib` queries and running them through [abCROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN)'s branch-and-bound verifier, binary-searching for the largest certified epsilon. Writes `solution_2.csv` and `query_2.vnnlib`. |
| [mnist-net_256x2.onnx](mnist-net_256x2.onnx) | Target model under evaluation. |
| [mnist_img_9.csv](mnist_img_9.csv) | Input image (a digit `9`), flattened 28x28 pixel values in `[0, 1]`. |

Generated artifacts (solutions, vnnlib queries, ab-CROWN run configs/output,
compiled specs, notes, and source archives) are intentionally excluded from
version control — see [.gitignore](.gitignore).

## Requirements

- Python 3.10+
- `numpy`, `torch`, `onnx`, `onnxruntime`, `onnx2pytorch` (for Exercise 1)
- `pyyaml` and a local checkout of [abCROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN) (for Exercise 2)

## Usage

```bash
# Exercise 1: craft a minimal adversarial example
python code_1.py

# Exercise 2: certify robustness via ab-CROWN
python code_2.py --abcrown_py ../abCROWN/complete_verifier/abcrown.py
```
