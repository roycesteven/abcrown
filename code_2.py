import os
import csv
import time
import json
import pickle
import argparse
import subprocess
import numpy as np
import yaml
import onnxruntime as ort
import sys


ONNX_PATH_DEFAULT = "mnist-net_256x2.onnx"
IMG_CSV_PATH_DEFAULT = "mnist_img_9.csv"

ABCROWN_PY_DEFAULT = "../abCROWN/complete_verifier/abcrown.py"

TIMEOUT_DEFAULT = 300
MAX_STEPS_DEFAULT = 30
TARGET_RATIO_DEFAULT = 1+1e-9


STD_CONF = """
general:
  device: cuda
  seed: 100
  results_file: out.txt
  save_output: true
  output_file: out.pkl
  root_path: .
model:
  onnx_path: null
  input_shape: null
specification:
  type: lp
  vnnlib_path: null
  vnnlib_path_prefix: ''
bab:
  timeout: 60
  branching:
    method: naive
    input_split:
      enable: True
      reorder_bab: True
"""

def load_image_flat(csv_path: str) -> np.ndarray:
    img = np.loadtxt(csv_path, delimiter=",").astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    x_flat = img.reshape(-1)
    return x_flat

def predict_class_with_onnxruntime(onnx_path: str, x_flat: np.ndarray) -> int:
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    inp_name = inp.name
    ishape = inp.shape  
    dims = [(-1 if s is None else int(s)) for s in ishape]
    print(len(dims))

    need = 1
    for s in dims[1:]:
        need *= s

    x = x_flat.reshape(dims).astype(np.float32)
    logits = sess.run(None, {inp_name: x})[0]  
    return int(np.argmax(logits, axis=1)[0])


def write_vnnlib_counterexample_query(x_flat: np.ndarray, eps: float, c: int, out_path: str):

    d = x_flat.shape[0]      
    num_classes = 10

    x_f64 = x_flat.astype(np.float64)
    lbs = np.maximum(0.0, x_f64 - eps)
    ubs = np.minimum(1.0, x_f64 + eps)

    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(d):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")
        for j in range(num_classes):
            f.write(f"(declare-const Y_{j} Real)\n")
        f.write("\n; Input constraints:\n")
        for i in range(d):
            f.write(f"(assert (>= X_{i} {lbs[i]}))\n")
            f.write(f"(assert (<= X_{i} {ubs[i]}))\n")

        f.write("\n; Output constraints: misclassification exists\n")
        f.write("(assert (or\n")
        for j in range(num_classes):
            if j == c:
              continue
            f.write(f"  (and (>= Y_{j} Y_{c}))\n")
        f.write("))\n")


def load_base_config(base_config_path: str | None) -> dict:
    if base_config_path:
        with open(base_config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return yaml.safe_load(STD_CONF)


def run_abcrown_with_config(
    abcrown_py: str,
    base_conf: dict,
    onnx_path: str,
    input_shape: list,
    vnnlib_file: str,
    timeout_s: int,
    device: str,
) -> str:

    conf = json.loads(json.dumps(base_conf))
    print(timeout_s)
    conf.setdefault("general", {})
    conf.setdefault("model", {})
    conf.setdefault("specification", {})
    conf.setdefault("bab", {})

    conf["general"]["device"] = device
    conf["bab"]["timeout"] = int(timeout_s)

    conf["model"]["onnx_path"] = onnx_path
    conf["model"]["input_shape"] = input_shape

    vnn_dir = os.path.dirname(os.path.abspath(vnnlib_file))
    vnn_name = os.path.basename(vnnlib_file)
    conf["specification"]["vnnlib_path_prefix"] = vnn_dir + "/" if vnn_dir else ""
    conf["specification"]["vnnlib_path"] = vnn_name

    stamp = f"{time.time():.6f}".replace(".", "_")
    out_pkl = f"out_{stamp}.pkl"
    out_txt = f"out_{stamp}.txt"
    conf["general"]["output_file"] = out_pkl
    conf["general"]["results_file"] = out_txt
    conf["general"]["save_output"] = True

    conf_yaml = os.path.join(vnn_dir, f"conf_{stamp}.yaml")
    with open(conf_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(conf, f)

    cmd = ["python", abcrown_py, "--config", conf_yaml]
    proc = subprocess.run(
        cmd,
        cwd=vnn_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )



    out_pkl_path = os.path.join(vnn_dir, out_pkl)


    with open(out_pkl_path, "rb") as f:
        result_dict = pickle.load(f)

    res = result_dict.get("results", None)

    for p in [conf_yaml, out_pkl_path, os.path.join(vnn_dir, out_txt)]:
        try:
            os.remove(p)
        except OSError:
            pass

    if isinstance(res, str):
        r = res.strip().lower()
        return r

def is_robust_status(status: str) -> bool:
    return status == "unsat"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=ONNX_PATH_DEFAULT)
    ap.add_argument("--img", default=IMG_CSV_PATH_DEFAULT)
    ap.add_argument("--abcrown_py", default=ABCROWN_PY_DEFAULT)
    ap.add_argument("--base_config", "--config", dest="base_config", default=None)
    ap.add_argument("--timeout", type=int, default=TIMEOUT_DEFAULT)
    ap.add_argument("--device", default="cuda")  
    ap.add_argument("--max_steps", type=int, default=MAX_STEPS_DEFAULT)
    ap.add_argument("--target_ratio", type=float, default=TARGET_RATIO_DEFAULT)
    ap.add_argument("--input_shape", default=",-1,784,1") 

    _sys_argv_backup = sys.argv
    sys.argv = [''] 
    args = ap.parse_args([]) 
    sys.argv = _sys_argv_backup 

    shape_parts = [p for p in args.input_shape.split(",") if p.strip() != ""]
    input_shape = [int(p) for p in shape_parts]

    x_flat = load_image_flat(args.img)
    c = predict_class_with_onnxruntime(args.onnx, x_flat)
    print(f"class = {c}")

    base_conf = load_base_config(args.base_config)

    history = []  
    lo = 0.036
    hi = 0.036075009012222284

    while True:
        tmp_vnn = "tmp_query.vnnlib"
        write_vnnlib_counterexample_query(x_flat, hi, c, tmp_vnn)
        status = run_abcrown_with_config(
            abcrown_py=args.abcrown_py,
            base_conf=base_conf,
            onnx_path=os.path.abspath(args.onnx),
            input_shape=input_shape,
            vnnlib_file=os.path.abspath(tmp_vnn),
            timeout_s=args.timeout,
            device=args.device,
        )
        robust = is_robust_status(status)
        if status == "unsat":
            label = "robust"
        elif status == "sat":
            label = "unsafe"
        else:
            label = "unknown"
        history.append((hi, label))
        print(f"[bracket] eps={hi} -> {status} => {label}")

        if not robust:
            break

        lo = hi
        hi *= 2.0
        if hi >= 1.0:
            hi = 1.0
            break

    for step in range(args.max_steps):
        if lo > 0 and (hi / lo) <= args.target_ratio:
            print(f"Stopping early: hi/lo = {hi/lo} <= {args.target_ratio}")
            break

        mid = (lo + hi) / 2.0
        tmp_vnn = "tmp_query.vnnlib"
        write_vnnlib_counterexample_query(x_flat, mid, c, tmp_vnn)
        status = run_abcrown_with_config(
            abcrown_py=args.abcrown_py,
            base_conf=base_conf,
            onnx_path=os.path.abspath(args.onnx),
            input_shape=input_shape,
            vnnlib_file=os.path.abspath(tmp_vnn),
            timeout_s=args.timeout,
            device=args.device,
        )
        robust = is_robust_status(status)
        if status == "unsat":
            label = "robust"
        elif status == "sat":
            label = "unsafe"
        else:
            label = "unknown"
        history.append((mid, label))
        print(f"[search {step+1:02d}] eps={mid} -> {status} => {label}")

        if robust:
            lo = mid
        else:
            hi = mid
    

    best_eps = lo

    with open("solution_2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for e, lab in history:
            w.writerow([f"{e}", lab])

    write_vnnlib_counterexample_query(x_flat, best_eps, c, "query_2.vnnlib")
    print(f"Best eps = {best_eps}")

if __name__ == "__main__":
    main()