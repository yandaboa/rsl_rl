"""Acting vs update-path log-prob parity for ActorCriticEpisodeContext (CPU, no Isaac).

Mirrors what PPO actually does: collect W steps through the KV ring (``act`` -> sample -> log_prob),
then re-infer the same window with ``forward_window(prefix=...)`` built exactly like
``EpisodeContextRolloutStorage.context_slice`` does, and compare the two distributions with the
parameters UNCHANGED. Any nonzero clip fraction here is a reconstruction error, not a policy update.
"""

from __future__ import annotations

import argparse
import math
import sys
import torch

REPO = "/home/yandabao/UWLab-patrick-private/.claude/worktrees/worktree-local-exploration"
sys.path.insert(0, f"{REPO}/_rsl_rl")

import torch.nn.functional as F  # noqa: E402

from rsl_rl.modules.actor_critic_episode_context import (  # noqa: E402
    ActorCriticEpisodeContext,
    EpisodeContextPrefix,
)

# ---- TF32 emulation -------------------------------------------------------------------------------
# scripts/reinforcement_learning/rsl_rl/train.py:252 sets torch.backends.cuda.matmul.allow_tf32 = True,
# so every GEMM on the cluster rounds its INPUTS to a 10-bit mantissa (fp32 accumulate). Emulated here by
# rounding the inputs of F.linear / SDPA; that is the dominant term, reduction order is not modelled.
_F_LINEAR, _F_SDPA = F.linear, F.scaled_dot_product_attention


def _tf32(x: torch.Tensor) -> torch.Tensor:
    if x is None or x.dtype != torch.float32:
        return x
    bits = x.contiguous().view(torch.int32)
    return ((bits + 0x1000) & ~0x1FFF).view(torch.float32)


def _linear_tf32(x, w, b=None):
    return _F_LINEAR(_tf32(x), _tf32(w), b)


def _sdpa_tf32(q, k, v, *a, **kw):
    return _F_SDPA(_tf32(q), _tf32(k), _tf32(v), *a, **kw)


def enable_tf32_emulation(on: bool) -> None:
    F.linear = _linear_tf32 if on else _F_LINEAR
    F.scaled_dot_product_attention = _sdpa_tf32 if on else _F_SDPA

CKPT_L160 = "/home/yandabao/UWLab-patrick-private/pulled_ckpts/asteroid_pc160/iter1_ep175_gate0.772.pt"
CKPT_L16 = "/home/yandabao/UWLab-patrick-private/init_weights/episode_context/bc_pcctx16_0615_L16_ep021.pt"
OBS_GROUPS = {"policy": ["policy"], "critic": ["critic"]}


def build_policy(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    hp = ck["hparams"]
    pk = dict(hp["policy_kwargs"])
    obs_dim = int(hp["num_actor_obs"])
    critic_dim = int(hp.get("num_critic_obs", obs_dim))
    num_actions = int(hp["num_actions"])
    obs_groups = hp.get("obs_groups", OBS_GROUPS) or OBS_GROUPS
    example = {"policy": torch.zeros(1, obs_dim), "critic": torch.zeros(1, critic_dim)}
    policy = ActorCriticEpisodeContext(example, obs_groups, num_actions, **pk).float()
    policy.load_state_dict(ck["model_state_dict"], strict=True)
    policy.eval()  # freezes EmpiricalNormalization (mirrors defer_obs_normalization at update time)
    norm = policy.actor_obs_normalizer
    mean = norm._mean.clone().float()
    std = norm._std.clone().float()
    return policy, obs_dim, num_actions, mean, std, pk


def stats(x: torch.Tensor) -> tuple[float, float]:
    x = x.reshape(-1)
    return float(x.abs().mean()), float(x.abs().max())


def run_adam_probe(tag, policy, recs, ring, frame_obs, frame_pos, W, P):
    """How big is the ratio after ONE Adam step? Adam's element-wise step is ~lr, so a Rademacher
    perturbation of magnitude lr is a fair stand-in for one update at that learning rate."""
    named = [(n, p) for n, p in policy.named_parameters()]
    backup = [p.detach().clone() for _, p in named]
    g = torch.Generator().manual_seed(1234)
    print(f"\n[{tag}] one-Adam-step probe (Rademacher |dtheta| = lr on every parameter):")
    for lr in (1e-5, 3e-5, 1e-4, 3e-4):
        with torch.no_grad():
            for (_, prm), b in zip(named, backup):
                prm.copy_(b + lr * (torch.randint(0, 2, prm.shape, generator=g).float() * 2 - 1))
        # Only the LAST rollout: the ring no longer holds the prefix frames of the earlier ones.
        r = recs[-1]
        total = W * len(recs)
        gidx = torch.arange(total - W - P, total)
        slots = torch.remainder(gidx, ring)
        positions = frame_pos[slots]
        prefix = EpisodeContextPrefix(
            obs=frame_obs[slots[:P]], positions=positions[:P], window_positions=positions[P:]
        )
        with torch.no_grad():
            h = policy.forward_window(r["raw"], prefix=prefix)
            policy._update_distribution(h)
            lp = policy.get_actions_log_prob(r["act"])
        rr = torch.exp(lp - r["lp"]).reshape(-1)
        print(
            f"    lr={lr:.0e}: ratio mean {rr.mean():.4f} std {rr.std():.4f}"
            f"  clipfrac@0.2 {float(((rr-1).abs()>0.2).float().mean()):.4f}"
            f"  |dlogp| mean {(rr.log()).abs().mean():.4f} max {(rr.log()).abs().max():.4f}"
        )
    with torch.no_grad():
        for (_, prm), b in zip(named, backup):
            prm.copy_(b)


def run(tag, policy, obs_dim, num_actions, mean, std, W, total_steps, ep_lo, ep_hi, seed, obs_mode, extra_prefix,
        adam_probe=False):
    torch.manual_seed(seed)
    B = 8
    T = policy.max_episode_length
    span = policy.context_span
    P = policy.context_prefix_length
    P_ext = min(T - 1, extra_prefix) if extra_prefix else P
    ring = W + max(P, P_ext) + 1
    num_rollouts = math.ceil(total_steps / W)

    frame_obs = torch.zeros(ring, B, obs_dim)      # normalized, as the storage keeps them
    frame_pos = torch.zeros(ring, B, dtype=torch.long)
    ep_step = torch.zeros(B, dtype=torch.long)
    total = 0

    ep_len = torch.randint(ep_lo, ep_hi + 1, (B,))
    policy.reset(None)

    recs = []  # per-rollout dicts of acting-time tensors
    ar_state = torch.randn(B, obs_dim)

    for _ in range(num_rollouts):
        raw_obs = torch.zeros(W, B, obs_dim)
        acts = torch.zeros(W, B, num_actions)
        lp_a = torch.zeros(W, B)
        mu_a = torch.zeros(W, B, num_actions)
        sd_a = torch.zeros(W, B, num_actions)
        h_a = torch.zeros(W, B, policy.d_model)
        pos_a = torch.zeros(W, B, dtype=torch.long)

        for w in range(W):
            if obs_mode == "ar1":
                ar_state = 0.95 * ar_state + math.sqrt(1 - 0.95**2) * torch.randn(B, obs_dim)
                z = ar_state
            else:
                z = torch.randn(B, obs_dim)
            obs = mean + std * z
            with torch.no_grad():
                a = policy.act(obs)                       # ppo.act(): sample through the KV ring
                lp = policy.get_actions_log_prob(a)
                mu, sd = policy.action_mean.clone(), policy.action_std.clone()
                h = policy._last_hidden.clone()
                nobs = policy.actor_obs_normalizer(obs)

            raw_obs[w], acts[w], lp_a[w], mu_a[w], sd_a[w], h_a[w] = obs, a, lp, mu, sd, h
            pos_a[w] = ep_step

            slot = total % ring
            frame_obs[slot] = nobs
            frame_pos[slot] = ep_step
            total += 1

            dones = (ep_step + 1) >= ep_len
            ep_step = (ep_step + 1) * (~dones).long()
            ep_len = torch.where(dones, torch.randint(ep_lo, ep_hi + 1, (B,)), ep_len)
            with torch.no_grad():
                policy.reset(dones)

        recs.append(dict(raw=raw_obs, act=acts, lp=lp_a, mu=mu_a, sd=sd_a, h=h_a, pos=pos_a))

        # ---- update path: [prefix | window] exactly as context_slice() builds it ----
        first = total - W
        out = {}
        for name, p_len in (("storage", P), ("ext", P_ext)):
            if name == "ext" and P_ext == P:
                continue
            g = torch.arange(first - p_len, first + W)
            slots = torch.remainder(g, ring)
            positions = frame_pos[slots]
            pre_obs = frame_obs[slots[:p_len]]
            prefix = EpisodeContextPrefix(
                obs=pre_obs, positions=positions[:p_len], window_positions=positions[p_len:]
            )
            with torch.no_grad():
                h_u = policy.forward_window(raw_obs, prefix=prefix)
                policy._update_distribution(h_u)
                lp_u = policy.get_actions_log_prob(acts)
                out[name] = (h_u, policy.action_mean.clone(), policy.action_std.clone(), lp_u)
            # sanity: the storage's window positions must equal the acting-time episode steps
            assert torch.equal(positions[p_len:], pos_a), f"{tag}: window positions != acting episode steps"
        recs[-1]["upd"] = out

    if adam_probe:
        run_adam_probe(tag, policy, recs, ring, frame_obs, frame_pos, W, P)
    sd_all = torch.cat([r["sd"].reshape(-1) for r in recs])
    print(f"\n[{tag}] acting action std: mean {sd_all.mean():.4f} min {sd_all.min():.4f} max {sd_all.max():.4f}")
    return summarize(tag, recs, P, P_ext, span, T, W)


def summarize(tag, recs, P, P_ext, span, T, W):
    rows = {}
    for name in ("storage", "ext"):
        if name not in recs[0]["upd"]:
            continue
        dmu, dlsd, dlp, ratio, pos, dh = [], [], [], [], [], []
        for r in recs:
            h_u, mu_u, sd_u, lp_u = r["upd"][name]
            dmu.append((r["mu"] - mu_u) / r["sd"])
            dlsd.append(torch.log(sd_u) - torch.log(r["sd"]))
            d = lp_u - r["lp"]
            dlp.append(d)
            ratio.append(torch.exp(d))
            pos.append(r["pos"])
            dh.append((r["h"] - h_u).abs().amax(-1) / r["h"].abs().amax(-1))
        dmu = torch.cat([x.reshape(-1, x.shape[-1]) for x in dmu])
        dlsd = torch.cat([x.reshape(-1, x.shape[-1]) for x in dlsd])
        dlp = torch.cat([x.reshape(-1) for x in dlp])
        ratio = torch.cat([x.reshape(-1) for x in ratio])
        pos = torch.cat([x.reshape(-1) for x in pos])
        dh = torch.cat([x.reshape(-1) for x in dh])
        clip = (ratio - 1).abs() > 0.2
        rows[name] = dict(
            n=dlp.numel(),
            dmu=stats(dmu), dlsd=stats(dlsd), dlp=stats(dlp), dh=stats(dh),
            clip=float(clip.float().mean()),
            pos=pos, dlp_v=dlp, ratio=ratio, dmu_v=dmu, dh_v=dh,
        )

    print(f"\n{'='*100}\n{tag}   (P_storage={P}, P_ext={P_ext}, span={span}, T={T}, W={W})\n{'='*100}")
    for name, r in rows.items():
        print(
            f"[{name:8s}] n={r['n']:6d}  |dmu|/std mean {r['dmu'][0]:.3e} max {r['dmu'][1]:.3e} |"
            f"  |dlog sd| mean {r['dlsd'][0]:.3e} max {r['dlsd'][1]:.3e} |"
            f"  |dlogp| mean {r['dlp'][0]:.3e} max {r['dlp'][1]:.3e} |"
            f"  rel|dh| mean {r['dh'][0]:.3e} max {r['dh'][1]:.3e} |"
            f"  CLIPFRAC(|r-1|>0.2) {r['clip']:.4f}"
        )
        rr = r["ratio"]
        q = torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0])
        print(
            "           ratio quantiles [0,1,50,99,100]%: "
            + " ".join(f"{v:.6f}" for v in torch.quantile(rr, q))
            + "   clipfrac@0.05/0.1/0.2: "
            + " ".join(f"{float(((rr-1).abs()>t).float().mean()):.4f}" for t in (0.05, 0.1, 0.2))
        )
        p = r["pos"]
        bins = [("early <32", p < 32), ("mid 32-128", (p >= 32) & (p <= 128)), ("late >128", p > 128)]
        for label, m in bins:
            if int(m.sum()) == 0:
                continue
            d, rr, dm, dhh = r["dlp_v"][m], r["ratio"][m], r["dmu_v"][m], r["dh_v"][m]
            print(
                f"           {label:12s} n={int(m.sum()):6d}  |dlogp| mean {d.abs().mean():.3e}"
                f" max {d.abs().max():.3e}  clip {float(((rr-1).abs()>0.2).float().mean()):.4f}"
                f"  |dmu|/std mean {dm.abs().mean():.3e}  rel|dh| mean {dhh.mean():.3e}"
            )
        # distance-from-episode-start buckets (fine grained near the prefix boundary)
        edges = [0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 10**6]
        line = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p >= lo) & (p < hi)
            if int(m.sum()) == 0:
                continue
            line.append(f"{lo}-{hi-1}:{r['dlp_v'][m].abs().mean():.2e}/{float(((r['ratio'][m]-1).abs()>0.2).float().mean()):.2f}")
        print("           by ep-step |dlogp|mean/clip: " + "  ".join(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obs_mode", default="iid", choices=["iid", "ar1", "both"])
    ap.add_argument("--adam_probe", action="store_true", help="also probe the ratio after one Adam step")
    ap.add_argument("--tf32", action="store_true", help="emulate the cluster's allow_tf32=True GEMMs")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    enable_tf32_emulation(args.tf32)
    print(f"[cfg] tf32 emulation: {args.tf32}")
    modes = ["iid", "ar1"] if args.obs_mode == "both" else [args.obs_mode]
    verdict = []

    for mode in modes:
        for label, ckpt, lo, hi in (
            ("L=160 T=160 (asteroid_pc160 iter1)", CKPT_L160, 40, 160),
            ("L=160 T=160 OVERLONG eps 170-240", CKPT_L160, 170, 240),
            ("L=16  T=80  (bc_pcctx16_0615)", CKPT_L16, 40, 80),
        ):
            policy, obs_dim, na, mean, std, pk = build_policy(ckpt)
            rows = run(
                f"{label}  obs={mode}", policy, obs_dim, na, mean, std,
                args.window, args.steps, lo, hi, args.seed, mode,
                extra_prefix=policy.max_episode_length - 1,
                adam_probe=args.adam_probe,
            )
            for name, r in rows.items():
                verdict.append((f"{label} [{mode}/{name}]", r["clip"], r["dlp"][1], r["dlp"][0], r["dmu"][1]))

    print("\n" + "=" * 100)
    print("VERDICT: zero-weight-change PPO ratio (parameters IDENTICAL between acting and update)")
    print("=" * 100)
    print(f"{'config':52s} {'clipfrac>0.2':>13s} {'max|dlogp|':>12s} {'mean|dlogp|':>12s} {'max|dmu|/std':>13s}")
    for name, clip, mx, mn, dmu in verdict:
        print(f"{name:52s} {clip:13.4f} {mx:12.3e} {mn:12.3e} {dmu:13.3e}")


if __name__ == "__main__":
    main()
