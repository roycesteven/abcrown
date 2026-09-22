
import onnx

import subprocess
from io import StringIO
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from onnx2pytorch import ConvertModel

torch.manual_seed(0)
np.random.seed(0)
onnx_model = onnx.load("mnist-net_256x2.onnx")
pytorch_model = ConvertModel(onnx_model)
pytorch_model.eval()
print(pytorch_model)


img = np.loadtxt("solution_1.csv", delimiter=",").astype(np.float32)

assert img.shape == (28, 28), f"expected (28,28), got {img.shape}"

img = np.clip(img, 0.0, 1.0)

plt.imshow(img, cmap="gray")
plt.title("Input image x (from CSV)")
plt.axis("off")
plt.show()

x = torch.from_numpy(img).view(1, 784)

with torch.no_grad():
    logits = pytorch_model(x)                  
    c = int(logits.argmax(dim=1).item())       

print("Logits shape:", tuple(logits.shape))
print("Original class c = f(x):", c)

exit()

y_c = torch.tensor([c], dtype=torch.long)

def fgsm_candidate(x_in: torch.Tensor, eps: float) -> torch.Tensor:

    x_var = x_in.clone().detach().requires_grad_(True)

    logits = pytorch_model(x_var)

    loss = F.cross_entropy(logits, y_c)

    pytorch_model.zero_grad()

    loss.backward()

    x_prime = x_var + eps * x_var.grad.sign()

    x_prime = x_prime.clamp(0.0, 1.0).detach()

    return x_prime


def predicted_class(x_in: torch.Tensor) -> int:
    with torch.no_grad():
        return int(pytorch_model(x_in).argmax(dim=1).item())

def misclassified(x_prime: torch.Tensor) -> bool:
    return predicted_class(x_prime) != c

MAX_EPS = 1.0        
BSEARCH_STEPS = 1000   

eps_hi = 1e-4
x_hi = None

while eps_hi <= MAX_EPS:
    cand = fgsm_candidate(x, eps_hi)
    if misclassified(cand):
        x_hi = cand
        break
    eps_hi *= 2.0

if x_hi is None:
    print("tidak ditemukan sampai dengan eps =", MAX_EPS)
else:
    eps_lo = 0.0
    best_eps = eps_hi
    best_xprime = x_hi

    for _ in range(BSEARCH_STEPS):
        eps_mid = (eps_lo + eps_hi) / 2.0
        cand = fgsm_candidate(x, eps_mid)

        if misclassified(cand):
            best_eps = eps_mid
            best_xprime = cand
            eps_hi = eps_mid
        else:
            eps_lo = eps_mid

    eps_star = best_eps  
    x_star = best_xprime 

    x_adv = fgsm_candidate(x, eps_star)

    pred_adv = predicted_class(x_adv)

    print("c = f(x):", c)
    print("f(x'):", pred_adv)
    print("Misclassification (f(x') != f(x)):", pred_adv != c)
    print("Approx minimal eps*:", eps_star)

    img_adv = x_adv.view(28, 28).cpu().numpy()
    perturb = (x_adv - x).view(28, 28).cpu().numpy()

    plt.figure()
    plt.imshow(img_adv, cmap="gray")
    plt.title(f"Adversarial image x' (pred={pred_adv}, eps={eps_star:.6f})")
    plt.axis("off")
    plt.show()


    plt.figure()
    plt.imshow(perturb, cmap="gray")
    plt.title("Perturbation (x' - x)")
    plt.axis("off")
    plt.show()

    img_adv = x_adv.detach().cpu().view(28, 28).numpy()

    # np.savetxt("solution_1.csv", img_adv, delimiter=",")