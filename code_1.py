"""
Exercise 1 - COMP60272 AI Security
Minimal L-infinity adversarial example for mnist-net_256x2.

Method: a single FGSM direction (sign of the input gradient at x), followed by
a binary search on the step size eps for the smallest perturbation that flips
the prediction.

The misclassification oracle is onnxruntime, i.e. the same backend used to
grade the submission. onnx2pytorch is used *only* to obtain the input gradient.
Deciding with PyTorch while being graded with onnxruntime is what produced the
earlier near-boundary answer: the two backends disagreed by ~1e-7 in the logits
right at the decision boundary, so the saved image scored f(x') = f(x).

To make the answer robust to that kind of numerical noise, the search does not
stop at the first eps that flips the class. It continues until the top-1/top-2
logit gap reaches MARGIN_TARGET, which costs only a few percent in eps and
stays well inside the 10% tolerance on the reported minimal eps.
"""

import warnings

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from onnx2pytorch import ConvertModel

ONNX_PATH = "mnist-net_256x2.onnx"
IMG_PATH = "mnist_img_9.csv"
OUT_PATH = "solution_1.csv"

MAX_EPS = 1.0
EPS_START = 1e-4
BSEARCH_STEPS = 60      # float64 binary search converges long before this
MARGIN_TARGET = 0.05    # required top1-top2 logit gap at the final eps
MARGIN_STEP = 1.002     # multiplicative eps increase while growing the margin

SHOW_PLOTS = False

torch.manual_seed(0)
np.random.seed(0)


def load_image(path: str) -> np.ndarray:
    img = np.loadtxt(path, delimiter=",").astype(np.float32)
    assert img.shape == (28, 28), f"expected (28,28), got {img.shape}"
    return np.clip(img, 0.0, 1.0)


def ort_logits(session, input_name: str, img_flat: np.ndarray) -> np.ndarray:
    """Logits from onnxruntime. This is the authoritative classifier."""
    x = np.asarray(img_flat, dtype=np.float32).reshape(1, 784, 1)
    return session.run(None, {input_name: x})[0].ravel()


def main():
    onnx_model = onnx.load(ONNX_PATH)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pytorch_model = ConvertModel(onnx_model)
    pytorch_model.eval()

    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    img = load_image(IMG_PATH)
    x = torch.from_numpy(img).view(1, 784)

    # Clean prediction, decided by onnxruntime.
    clean_logits = ort_logits(session, input_name, img)
    c = int(np.argmax(clean_logits))

    with torch.no_grad():
        c_torch = int(pytorch_model(x).argmax(dim=1).item())
    if c_torch != c:
        print(f"warning: backends disagree on the clean image "
              f"(onnxruntime={c}, onnx2pytorch={c_torch}); trusting onnxruntime")

    print(f"Original class c = f(x): {c}")

    # FGSM direction. The gradient is taken at x, so it does not depend on eps
    # and is computed once.
    x_var = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(pytorch_model(x_var), torch.tensor([c], dtype=torch.long))
    pytorch_model.zero_grad()
    loss.backward()
    direction = x_var.grad.sign().detach()

    def candidate(eps: float) -> torch.Tensor:
        return (x + eps * direction).clamp(0.0, 1.0).detach()

    def evaluate(eps: float):
        """Return (predicted class, top1-top2 gap) for this eps, via onnxruntime."""
        logits = ort_logits(session, input_name, candidate(eps).numpy())
        pred = int(np.argmax(logits))
        top2 = np.sort(logits)[::-1]
        return pred, float(top2[0] - top2[1])

    # Phase 1: bracket the boundary by doubling eps.
    eps_hi = EPS_START
    while eps_hi <= MAX_EPS and evaluate(eps_hi)[0] == c:
        eps_hi *= 2.0

    if eps_hi > MAX_EPS:
        raise SystemExit(f"no misclassification found up to eps = {MAX_EPS}")

    # Phase 2: binary search for the smallest eps that flips the class.
    eps_lo = 0.0
    for _ in range(BSEARCH_STEPS):
        eps_mid = (eps_lo + eps_hi) / 2.0
        if evaluate(eps_mid)[0] != c:
            eps_hi = eps_mid
        else:
            eps_lo = eps_mid

    eps_min = eps_hi          # smallest eps observed to misclassify
    pred_min, margin_min = evaluate(eps_min)
    print(f"Minimal eps (boundary):  {eps_min:.9f}  "
          f"-> f(x')={pred_min}, margin={margin_min:.3e}")

    # Phase 3: step off the boundary so the result survives backend noise.
    eps_star = eps_min
    while True:
        pred, margin = evaluate(eps_star)
        if pred != c and margin >= MARGIN_TARGET:
            break
        eps_star *= MARGIN_STEP
        if eps_star > MAX_EPS:
            raise SystemExit(f"could not reach margin {MARGIN_TARGET} within eps <= {MAX_EPS}")

    x_adv = candidate(eps_star)
    img_adv = x_adv.view(28, 28).numpy()
    np.savetxt(OUT_PATH, img_adv, delimiter=",")

    # Verify what actually landed on disk: np.savetxt round-trips through text,
    # and the grader reads the file, not the in-memory array.
    reloaded = np.loadtxt(OUT_PATH, delimiter=",").astype(np.float32)
    logits_adv = ort_logits(session, input_name, reloaded)
    pred_adv = int(np.argmax(logits_adv))
    top2 = np.sort(logits_adv)[::-1]
    delta = float(np.abs(reloaded - img).max())
    rel_err = abs(delta - eps_min) / eps_min

    assert reloaded.shape == (28, 28), f"bad shape written: {reloaded.shape}"
    assert reloaded.min() >= 0.0 and reloaded.max() <= 1.0, "written image left [0,1]"
    assert pred_adv != c, f"written image is not adversarial: f(x')={pred_adv}"

    print()
    print(f"Saved {OUT_PATH}")
    print(f"  1.a  range [{reloaded.min():.4f}, {reloaded.max():.4f}]  -> in [0,1]")
    print(f"  1.b  f(x) = {c}, f(x') = {pred_adv}  -> misclassified")
    print(f"  1.c  delta = {delta:.6f}, eps_min = {eps_min:.6f}, rel.err = {100 * rel_err:.3f}%")
    print(f"       top1-top2 margin = {top2[0] - top2[1]:.6f}")

    if SHOW_PLOTS:
        import matplotlib.pyplot as plt

        for title, data in (
            ("Input image x", img),
            (f"Adversarial x' (pred={pred_adv}, eps={eps_star:.6f})", img_adv),
            ("Perturbation (x' - x)", img_adv - img),
        ):
            plt.figure()
            plt.imshow(data, cmap="gray")
            plt.title(title)
            plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
