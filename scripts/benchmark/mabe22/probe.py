"""Print the raw structure of every small MABe22 npy (keys, shapes, dtypes). Run inside a job."""
import sys, numpy as np, glob, os
D = "dataset/benchmark/mabe22/raw"
def desc(o, ind=0, name="", depth=0):
    p = " " * ind
    if isinstance(o, np.ndarray):
        extra = ""
        if o.dtype == object: extra = f" (object; first elem type {type(o.flat[0]).__name__})"
        print(f"{p}{name}: ndarray {o.shape} {o.dtype}{extra}")
        if o.dtype == object and o.ndim == 0: desc(o.item(), ind + 2, "item()", depth + 1)
    elif isinstance(o, dict):
        print(f"{p}{name}: dict {len(o)} keys: {list(o.keys())[:12]}")
        if depth < 3:
            for k in list(o.keys())[:6]: desc(o[k], ind + 2, repr(k), depth + 1)
    elif isinstance(o, (list, tuple)):
        print(f"{p}{name}: {type(o).__name__} len {len(o)}")
        if o and depth < 3: desc(o[0], ind + 2, "[0]", depth + 1)
    else:
        print(f"{p}{name}: {type(o).__name__} {str(o)[:100]}")
for f in sorted(glob.glob(f"{D}/*.npy")):
    if sys.argv[1:] and not any(a in f for a in sys.argv[1:]): continue
    print("==", os.path.basename(f), os.path.getsize(f))
    try: a = np.load(f, allow_pickle=True, mmap_mode='r')
    except ValueError: a = np.load(f, allow_pickle=True)
    desc(a, 2, "root")
