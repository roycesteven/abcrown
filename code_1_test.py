import onnx
import torch
import numpy as np
import torch.nn.functional as F
from onnx2pytorch import ConvertModel
import onnxruntime as ort

torch.manual_seed(0)
np.random.seed(0)

# Load model in both frameworks
onnx_model = onnx.load("mnist-net_256x2.onnx")
pytorch_model = ConvertModel(onnx_model)
pytorch_model.eval()

# ONNX runtime session (used for verification, matches grader)
session = ort.InferenceSession("mnist-net_256x2.onnx")
ort_input_name = session.get_inputs()[0].name


def ort_predict(img_flat):
    """Predict using ONNX runtime (same as grader)."""
    logits = session.run(None, {ort_input_name: img_flat.reshape(1, 784, 1)})[0]
    return int(np.argmax(logits[0]))


# Load original image
img = np.loadtxt("solution_1.csv", delimiter=",").astype(np.float32)
assert img.shape == (28, 28)
img = np.clip(img, 0.0, 1.0)

x = torch.from_numpy(img).view(1, 784)

# Get original class using ONNX runtime (authoritative)
c = ort_predict(img.flatten())
print("Original class c = f(x):", c)
exit()
y_c = torch.tensor([c], dtype=torch.long)


def fgsm_candidate(x_in, eps):
    x_var = x_in.clone().detach().requires_grad_(True)
    logits = pytorch_model(x_var)
    loss = F.cross_entropy(logits, y_c)
    pytorch_model.zero_grad()
    loss.backward()
    x_prime = x_var + eps * x_var.grad.sign()
    x_prime = x_prime.clamp(0.0, 1.0).detach()
    return x_prime


def ort_misclassified(x_prime):
    """Check misclassification using ONNX runtime (matches grader)."""
    return ort_predict(x_prime.numpy().flatten()) != c


# Phase 1: Find initial eps that causes misclassification (verified via ORT)
MAX_EPS = 1.0
eps_hi = 1e-4
x_hi = None

while eps_hi <= MAX_EPS:
    cand = fgsm_candidate(x, eps_hi)
    if ort_misclassified(cand):
        x_hi = cand
        break
    eps_hi *= 2.0

if x_hi is None:
    print("No misclassification found up to eps =", MAX_EPS)
else:
    # Phase 2: Binary search for minimum eps (verified via ORT)
    eps_lo = 0.0
    best_eps = eps_hi
    best_xprime = x_hi

    for _ in range(1000):
        eps_mid = (eps_lo + eps_hi) / 2.0
        cand = fgsm_candidate(x, eps_mid)

        if ort_misclassified(cand):
            best_eps = eps_mid
            best_xprime = cand
            eps_hi = eps_mid
        else:
            eps_lo = eps_mid

    eps_star = best_eps
    x_star = best_xprime

    # Final verification with ONNX runtime
    x_adv = fgsm_candidate(x, eps_star)
    pred_adv = ort_predict(x_adv.numpy().flatten())

    print(f"c = f(x): {c}")
    print(f"f(x'): {pred_adv}")
    print(f"Misclassification: {pred_adv != c}")
    print(f"Approx minimal eps*: {eps_star}")

    # Save adversarial image
    img_adv = x_adv.detach().view(28, 28).numpy()
    np.savetxt("solution_1.csv", img_adv, delimiter=",")
    print("Saved solution_1.csv")