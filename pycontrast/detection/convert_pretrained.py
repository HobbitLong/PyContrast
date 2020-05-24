import pickle as pkl
import torch
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convert Models')
    parser.add_argument('input', metavar='I',
                        help='input model path')
    parser.add_argument('output', metavar='O',
                        help='output path')
    parser.add_argument('--ema', action='store_true',
                        help='using ema model')
    args = parser.parse_args()

    obj = torch.load(args.input, map_location="cpu")
    if args.ema:
        obj = obj["model_ema"]
        prefix = "encoder."
    else:
        obj = obj["model"]
        prefix = "module.encoder."

    newmodel = {}
    for k, v in obj.items():
        if not k.startswith(prefix):
            continue
        old_k = k
        k = k.replace(prefix, "")
        if "layer" not in k:
            k = "stem." + k
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        print(old_k, "->", k)
        newmodel[k] = v.numpy()

    res = {"model": newmodel, "__author__": "Yonglong", "matching_heuristics": True}

    with open(args.output, "wb") as f:
        pkl.dump(res, f)
