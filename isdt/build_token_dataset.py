import os
import argparse
import torch


def load_tokens_pt(path):
    obj = torch.load(path, map_location="cpu")
    # obj: {"codes": list([N,3]), "key_idx": list([M]), "S_key": list([M,3])}
    assert "S_key" in obj, f"Missing S_key in {path}"
    return obj


def pad_stack_skey(skey_list, L=32):
    """
    skey_list: list of tensors, each [Mi, 3] (Mi<=L typically)
    return:
      tokens: [G, L, 3] long
      mask:   [G, L] bool (True for valid)
    """
    G = len(skey_list)
    tokens = torch.zeros(G, L, 3, dtype=torch.long)
    mask = torch.zeros(G, L, dtype=torch.bool)

    for i, sk in enumerate(skey_list):
        sk = sk.long()
        Mi = min(sk.size(0), L)
        tokens[i, :Mi] = sk[:Mi]
        mask[i, :Mi] = True
    return tokens, mask


def triple_to_single_id(tokens, K=256):
    """
    tokens: [G,L,3] with values in [0, K-1]
    map (m,t,p) -> id in [0, K^3 - 1]
    """
    m = tokens[..., 0]
    t = tokens[..., 1]
    p = tokens[..., 2]
    return (m * (K * K) + t * K + p).long()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", required=True, help="folder containing tokens_train/dev/test.pt")
    ap.add_argument("--top_m", type=int, default=32, help="pad length L (should match ISDT top_m)")
    ap.add_argument("--K", type=int, default=256, help="codebook size")
    ap.add_argument("--make_single_id", action="store_true", help="also save single token ids [G,L]")
    args = ap.parse_args()

    for split in ["train", "dev", "test"]:
        in_path = os.path.join(args.save_dir, f"tokens_{split}.pt")
        out_path = os.path.join(args.save_dir, f"token_dataset_{split}.pt")

        obj = load_tokens_pt(in_path)
        skey_list = obj["S_key"]  # list of [M,3]

        tokens, mask = pad_stack_skey(skey_list, L=args.top_m)

        out = {
            "tokens": tokens,       # [G,L,3]
            "attn_mask": mask,      # [G,L]
        }

        if args.make_single_id:
            out["token_ids"] = triple_to_single_id(tokens, K=args.K)  # [G,L]

        torch.save(out, out_path)
        print(f"[SAVED] {out_path} | tokens={tuple(tokens.shape)} mask={tuple(mask.shape)}")


if __name__ == "__main__":
    main()
