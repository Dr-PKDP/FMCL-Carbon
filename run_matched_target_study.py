"""
run_matched_target_study.py -- matched-target (balanced-accuracy) energy-to-
convergence comparison between the compact and large model on one domain.

WHY THIS EXISTS
----------------
Table 5/6 price every configuration at a *fixed* round budget (T) and report
whatever accuracy that budget reaches. That is an honest campaign-cost
accounting, but it leaves one question unanswered: when the compact model
and the large model are both run until they reach the SAME quality bar,
rather than the same round count, how does the energy/carbon cost of getting
there actually compare? §V-C's "capacity rescues convergence" finding is
about whether a bar is reached at all under a fixed budget; this script asks
the different, matched-target question directly, within a single
architecture (FMCL) so the comparison is not confounded by federated-vs-
centralised algorithmic differences the way a three-architecture time-to-
target race would be (see the paper's Discussion of why that broader
experiment was not attempted).

TARGET METRIC: BALANCED ACCURACY, NOT RAW ACCURACY
-----------------------------------------------------
MIT-BIH and Icentia11k are majority-class-dominated (89.5% and 71.1% of
examples belong to one class). A model that always predicts the majority
class already scores ~89.5%/71.1% raw accuracy while being clinically
useless. Table 6's raw-accuracy margins over that majority-class baseline
are the right lens for "did training move the needle at all", but they are
the wrong target metric for a matched-quality experiment, since a model
could clear the compact-model margin (see Table 6) while still being a
near-trivial classifier on the minority classes that matter clinically.
This script accordingly evaluates and stops on BALANCED accuracy (mean
per-class recall), not raw accuracy. balanced_accuracy() is new in this
script; fl_harness.evaluate() is untouched, and every other script in this
repo continues to use it exactly as before.

TARGET THRESHOLD IS A GUESS, NOT A CALIBRATED VALUE
-----------------------------------------------------
No prior run in this repo recorded balanced accuracy, only raw accuracy, so
there is no existing data point to calibrate --target against. The default
below (0.35 for MIT-BIH, 0.30 for Icentia11k) is a placeholder chosen to sit
clearly above chance-level balanced accuracy (1/n_classes: 0.20 for MIT-
BIH's 5 classes, 0.25 for Icentia11k's 4) without assuming either model can
reach it. Run once with --probe_only first (see below) to see what balanced
accuracy each model actually reaches within --max_rounds, then set --target
to a value both models have some chance of reaching before treating a
"did not reach target" result as meaningful rather than as evidence the
threshold was simply set too high.

USAGE
-----
    # Recommended first step: see what balanced accuracy each model reaches
    # within a generous round budget, with no early stopping, to calibrate
    # a sensible --target before spending seeds on the real comparison.
    python run_matched_target_study.py --data_dir PATH --domain mitbih \\
        --probe_only --max_rounds 400 --n_seeds 1

    # Matched-target comparison once --target is calibrated:
    python run_matched_target_study.py --data_dir PATH --domain mitbih \\
        --target 0.40 --max_rounds 400 --n_seeds 5 --out_dir results

Depends only on files already in this repo: data_mitbih.py,
data_icentia11k.py, data_femnist.py (model classes), fl_harness.py,
engine.py. Does not modify any of them.
"""
import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F

from fl_harness import (
    ClientState, Candidate, POLICIES, TIER_ENERGY_MULT, TRAIN_FLOP_MULTIPLIER,
    local_train, fedavg, count_forward_flops, model_bytes,
    topk_sparsify, sparse_payload_bytes,
)
from data_femnist import ResNet1D_PTB, TinyCNN_PTB
from engine import Params, account_all


def balanced_accuracy(model, state, x_eval, y_eval, n_classes,
                       device=torch.device("cpu")):
    """
    Mean per-class recall, computed on WHATEVER (x_eval, y_eval) is passed
    in. This function does no subsampling of its own -- the caller is
    responsible for fixing the evaluation set once and reusing it across
    every call within a run.

    Redrawing a random subsample on every call would make the trajectory
    bounce from sampling noise rather than the model's actual state: with
    MIT-BIH/Icentia11k's severe class imbalance, which rare classes land
    in a given draw varies call to call, and balanced accuracy divides by
    however many classes are present in that draw. A fixed evaluation set
    removes this: the only thing that can change balanced accuracy between
    calls is the model itself.

    Classes absent from x_eval/y_eval altogether (not just from one draw --
    genuinely absent from the fixed evaluation set) are excluded from the
    mean, not scored as 0, since that reflects a gap in the evaluation
    pool itself, not a model failure on that class.
    """
    model.to(device)
    model.load_state_dict(state)
    model.eval()
    batch = 64
    preds = []
    with torch.no_grad():
        for i in range(0, x_eval.shape[0], batch):
            xb = x_eval[i:i + batch].to(device, non_blocking=True)
            preds.append(model(xb).argmax(1).cpu())
    pred = torch.cat(preds)
    recalls = []
    for c in range(n_classes):
        mask = (y_eval == c)
        if mask.sum() == 0:
            continue
        recalls.append((pred[mask] == c).float().mean().item())
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    raw_acc = (pred == y_eval).float().mean().item()
    return macro_recall, raw_acc, len(recalls)


def run_to_target(clients, model_fn, input_shape, n_classes,
                   target, max_rounds, K, mu, policy_name, seed,
                   epochs=2, lr=0.01, batch_size=32, sparsify_frac=0.3,
                   participation_p=0.1, device=None,
                   max_batches_per_client=None, eval_every=1,
                   progress_every=25, progress_tag="", stability_window=5):
    """
    Adapted from study_runner.fl_run(), with two changes: evaluation uses
    balanced_accuracy() every eval_every rounds (default every round, not
    every 5th, so the stopping point is not overshot by several rounds'
    worth of unnecessary training), and the loop stops once balanced
    accuracy has held at or above target for stability_window consecutive
    evaluations, not the instant it is first touched. Everything else --
    client sampling, selection policy, local FedProx training,
    sparsification, aggregation -- is identical to fl_run(), so results
    from this script and from study_runner.py are produced by the same
    underlying training mechanics and differ only in the stopping rule.

    stability_window exists because real trajectories bounce round to
    round under partial participation. Stopping on the first single round
    that touches target risks capturing a noisy upward blip rather than
    genuine learning, particularly for a model hovering near target rather
    than clearing it comfortably. Requiring stability_window consecutive
    evaluations at or above target trades a few extra rounds of training
    for a stopping decision that reflects the model's actual state rather
    than one lucky round.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(seed)
    torch.manual_seed(seed)
    t_start = time.time()
    model = model_fn()
    global_state = {k: v.clone() for k, v in model.state_dict().items()}

    n_clients = len(clients)
    tiers = [["low", "medium", "high"][i % 3] for i in range(n_clients)]

    Xs, ys = [], []
    for c in clients:
        X = c["X"] if isinstance(c["X"], torch.Tensor) else torch.from_numpy(c["X"])
        y = c["y"] if isinstance(c["y"], torch.Tensor) else torch.from_numpy(c["y"])
        Xs.append(X); ys.append(y)
    x_all = torch.cat(Xs[:min(len(Xs), 200)])
    y_all = torch.cat(ys[:min(len(ys), 200)])

    # Fixed evaluation set, drawn once for the whole run and reused across
    # every call, so the only thing that can change balanced accuracy
    # between evaluations is the model itself.
    #
    # A uniform random draw over the pool is not sufficient on its own.
    # MIT-BIH's rarest AAMI classes (particularly Q, whose constituent
    # symbols are mostly associated with the paced records this loader
    # excludes by default, leaving very few residual examples) can be
    # missed entirely by a uniform draw, purely by chance. When that
    # happens the balanced-accuracy denominator silently drops (5 classes
    # -> 4), and a model that has merely learned to always predict the
    # majority class -- requiring no real learning -- lands on an
    # artificially inflated floor (1/4 = 0.25 instead of the honest
    # 1/5 = 0.20) that then looks stable, since the same degenerate eval
    # set is reused every round. The stability window cannot catch this on
    # its own: it only checks whether a score holds steady, not whether
    # the evaluation set behind that score is itself sound.
    #
    # Stratifying avoids this: every class actually present anywhere in
    # the pool is guaranteed meaningful representation in the fixed eval
    # set, rather than leaving that to chance. Up to per_class_cap
    # examples of each class are included (every available example, if
    # fewer exist than the cap); the remaining eval budget is filled with
    # a random draw from whatever is left. A missing class is then
    # possible only if that class has zero examples in the entire pool,
    # not merely few.
    n_pool = x_all.shape[0]
    max_eval = 2000
    if n_pool > max_eval:
        strat_rng = random.Random(seed)
        per_class_cap = max(50, max_eval // (2 * n_classes))
        stratified_idx = []
        for c in range(n_classes):
            class_idx = (y_all == c).nonzero(as_tuple=True)[0].tolist()
            strat_rng.shuffle(class_idx)
            stratified_idx.extend(class_idx[:per_class_cap])
        stratified_set = set(stratified_idx)
        remaining_budget = max_eval - len(stratified_idx)
        if remaining_budget > 0:
            remaining_pool = [i for i in range(n_pool) if i not in stratified_set]
            strat_rng.shuffle(remaining_pool)
            stratified_idx.extend(remaining_pool[:remaining_budget])
        eval_idx = torch.tensor(stratified_idx, dtype=torch.long)
        x_eval, y_eval = x_all[eval_idx], y_all[eval_idx]
        classes_in_eval = sorted(set(y_eval.tolist()))
        if len(classes_in_eval) < n_classes:
            missing = sorted(set(range(n_classes)) - set(classes_in_eval))
            print(f"  WARNING [{progress_tag}]: classes {missing} have ZERO examples "
                  f"in the entire data pool, not just the eval draw -- balanced "
                  f"accuracy for this run is out of {len(classes_in_eval)} classes, "
                  f"not {n_classes}.", flush=True)
    else:
        x_eval, y_eval = x_all, y_all

    cstates = [ClientState(cid=i, x=Xs[i], y=ys[i], device_tier=tiers[i])
               for i in range(n_clients)]

    dense_bytes = model_bytes(global_state)
    history = []
    fresh_evals = []  # only genuinely fresh evaluations, never carried-forward
                       # repeats -- kept separate from history so the stability
                       # window below counts real confirmations even if
                       # eval_every is ever set above 1 in the future
    total_client_rounds = 0
    total_flops = 0
    total_bytes = 0
    reached_round = None
    reached_balanced_acc = None

    for t in range(max_rounds):
        available = [c for c in cstates if rng.random() < participation_p]
        if len(available) < 2:
            available = cstates[:2]

        cands = []
        for c in available:
            with torch.no_grad():
                model.load_state_dict(global_state)
                model.to(device)
                if c.x.shape[0] == 0:
                    du = 0.0
                else:
                    xb = c.x[:min(8, c.x.shape[0])].to(device, non_blocking=True)
                    out = model(xb)
                    yb = c.y[:min(8, c.y.shape[0])].to(device, non_blocking=True)
                    du = float(F.cross_entropy(out, yb))
            su = 1.0 / TIER_ENERGY_MULT[c.device_tier]
            me = TIER_ENERGY_MULT[c.device_tier]
            lab = int(c.y[0].item()) if c.y.shape[0] > 0 else -1
            cands.append(Candidate(cid=c.cid, data_utility=du, system_utility=su,
                                   marginal_energy_J=me, grid_intensity=0.156,
                                   label=lab))

        k = min(K, len(cands))
        selected_ids = set(POLICIES[policy_name](cands, k, rng))
        selected = [c for c in available if c.cid in selected_ids]

        states, weights = [], []
        for c in selected:
            if c.x.shape[0] == 0:
                continue
            new_state, _, fwd_fl, _ = local_train(
                model, c, global_state, mu=mu, epochs=epochs,
                lr=lr, batch_size=batch_size, device=device,
                max_batches_per_client=max_batches_per_client)
            _, nnz = topk_sparsify(new_state, sparsify_frac)
            states.append(new_state)
            weights.append(int(c.x.shape[0]))
            n_batches = max(1, c.x.shape[0] // batch_size) * epochs
            total_flops += fwd_fl * TRAIN_FLOP_MULTIPLIER * n_batches
            total_bytes += sparse_payload_bytes(nnz) * 2
            total_client_rounds += 1

        if states:
            global_state = fedavg(states, weights)

        if t % eval_every == 0 or t == max_rounds - 1:
            bal_acc, raw_acc, n_classes_seen = balanced_accuracy(
                model, global_state, x_eval, y_eval, n_classes, device=device)
            fresh_evals.append({"round": t, "balanced_acc": bal_acc})
        else:
            bal_acc, raw_acc = history[-1]["balanced_acc"], history[-1]["raw_acc"]
            n_classes_seen = history[-1]["n_classes_seen"]

        history.append({"round": t, "balanced_acc": bal_acc, "raw_acc": raw_acc,
                        "n_classes_seen": n_classes_seen, "n_selected": len(selected)})

        if progress_every and (t % progress_every == 0 or t == max_rounds - 1):
            elapsed = time.time() - t_start
            rate = (t + 1) / elapsed if elapsed > 0 else 0.0
            eta = (max_rounds - t - 1) / rate if rate > 0 else float("nan")
            print(f"  [{progress_tag}] round {t:4d}/{max_rounds}  "
                  f"balanced_acc={bal_acc:.3f}  raw_acc={raw_acc:.3f}  "
                  f"elapsed={elapsed:6.0f}s  ~{rate:.2f} rounds/s  "
                  f"ETA={eta:6.0f}s", flush=True)

        if reached_round is None and bal_acc >= target:
            recent = [e["balanced_acc"] for e in fresh_evals[-stability_window:]]
            if len(recent) >= stability_window and all(v >= target for v in recent):
                reached_round = fresh_evals[-stability_window]["round"]  # the round the
                                                                           # qualifying
                                                                           # streak actually
                                                                           # started
                reached_balanced_acc = min(recent)  # the weakest round in the qualifying
                                                       # streak, not the peak, so this
                                                       # number is not flattered by
                                                       # whichever round happened to be
                                                       # highest within the streak
                break

    fpc = total_flops / max(total_client_rounds, 1)
    upload_mb = (total_bytes / max(total_client_rounds, 1) / 2) / 1e6
    rounds_used = (reached_round + 1) if reached_round is not None else max_rounds

    p_eng = Params(
        rounds=rounds_used,
        clients_per_round=K,
        flops_per_client_round=fpc,
        upload_MB=upload_mb,
        download_MB=upload_mb,
    )
    lifecycle_to_target = {
        arch: {k: float(v) for k, v in r.items()}
        for arch, r in account_all(p_eng).items()
    } if reached_round is not None else None

    return {
        "history": history,
        "reached_target": reached_round is not None,
        "rounds_to_target": (reached_round + 1) if reached_round is not None else None,
        "balanced_acc_at_target": reached_balanced_acc,
        "final_balanced_acc": history[-1]["balanced_acc"],
        "final_raw_acc": history[-1]["raw_acc"],
        "lifecycle_to_target_kg_CO2e": (
            {k: v["GWP100_kg_CO2e"] for k, v in lifecycle_to_target.items()}
            if lifecycle_to_target else None
        ),
        "measured": {"flops_per_client_round": fpc, "upload_MB": upload_mb,
                     "total_client_rounds": total_client_rounds,
                     "dense_model_MB": dense_bytes / 1e6},
        "config": {"target": target, "max_rounds": max_rounds, "K": K, "mu": mu,
                   "policy": policy_name, "seed": seed, "epochs": epochs, "lr": lr},
    }


DOMAIN_BUILDERS = {}


def _build_mitbih(data_dir, model):
    from data_mitbih import load_mitbih
    meta, clients = load_mitbih(data_dir)
    ModelClass = ResNet1D_PTB if model == "resnet" else TinyCNN_PTB
    model_fn = lambda: ModelClass(n_leads=2, n_classes=5)
    return meta, clients, model_fn, (2, meta["window_len"]), 5


def _build_icentia11k(data_dir, model):
    from data_icentia11k import load_icentia11k
    meta, clients = load_icentia11k(data_dir)
    ModelClass = ResNet1D_PTB if model == "resnet" else TinyCNN_PTB
    model_fn = lambda: ModelClass(n_leads=1, n_classes=4)
    return meta, clients, model_fn, (1, meta["window_len"]), 4


DOMAIN_BUILDERS["mitbih"] = _build_mitbih
DOMAIN_BUILDERS["icentia11k"] = _build_icentia11k

# Per-domain lr defaults matching run_mitbih_study.py / run_icentia11k_study.py's
# own documented settings -- see those files' docstrings for why these
# differ by domain. Overridable with --lr.
DEFAULT_LR = {"mitbih": 0.001, "icentia11k": 0.01}
DEFAULT_TARGET = {"mitbih": 0.35, "icentia11k": 0.30}
DEFAULT_K = {"mitbih": 6, "icentia11k": 30}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--domain", choices=["mitbih", "icentia11k"], required=True)
    parser.add_argument("--out_dir", default="results_matched_target")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--model", choices=["tiny", "resnet", "both"], default="both",
                        help="Run only one model, or both sequentially (default). "
                             "Use --model tiny and --model resnet in two separate "
                             "processes, each with its own CUDA_VISIBLE_DEVICES, to "
                             "run them in parallel on separate GPUs instead of "
                             "sequentially within one process.")
    parser.add_argument("--max_rounds", type=int, default=400,
                        help="Ceiling so a non-converging model does not run "
                             "forever. Table 6 used 100 rounds for both domains; "
                             "this defaults far higher since reaching a quality "
                             "target, rather than exhausting a fixed budget, is "
                             "the actual stopping condition here.")
    parser.add_argument("--K", type=int, default=None,
                        help="Defaults to this domain's value in run_*_study.py "
                             "(6 for mitbih, 30 for icentia11k) unless overridden.")
    parser.add_argument("--target", type=float, default=None,
                        help="Balanced-accuracy stopping target. Defaults to a "
                             "placeholder (0.35 mitbih / 0.30 icentia11k) -- see "
                             "module docstring. Recalibrate using --probe_only "
                             "before trusting a 'did not reach target' result.")
    parser.add_argument("--probe_only", action="store_true",
                        help="Ignore --target (run the full --max_rounds with no "
                             "early stopping) and report the balanced-accuracy "
                             "trajectory for both models, to calibrate a real "
                             "--target before spending seeds on the matched run.")
    parser.add_argument("--policy", default="energy_aware",
                        choices=["random", "oort_lite", "energy_aware",
                                 "carbon_aware", "class_balanced"])
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument("--stability_window", type=int, default=5,
                        help="Balanced accuracy must hold at or above --target "
                             "for this many CONSECUTIVE evaluations before the "
                             "run is considered to have reached it, rather than "
                             "stopping on a single round that happens to touch "
                             "target. Real trajectories are noisy round to "
                             "round; a lower value risks stopping on a lucky "
                             "blip rather than genuine, sustained learning.")
    parser.add_argument("--lr", type=float, default=None,
                        help="Defaults to this domain's confirmed-stable value "
                             "(see DEFAULT_LR) unless overridden.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Local training batch size. This is the RECOMMENDED "
                             "speedup for Icentia11k, not --max_batches_per_client "
                             "below: a larger batch size processes the same total "
                             "per-client data in fewer, larger batches, reducing "
                             "per-batch overhead without changing how much local "
                             "data is used per round. Matches the same "
                             "recommendation already documented in "
                             "run_icentia11k_study.py for the same reason.")
    parser.add_argument("--max_batches_per_client", type=int, default=None,
                        help="NOT RECOMMENDED for the primary study comparison -- "
                             "caps batches drawn per client per round, meaning a "
                             "client trains on only a SUBSET of its own local data "
                             "rather than the full local epoch every other domain "
                             "(PTB-XL, FEMNIST, MIT-BIH) uses in this repo. This "
                             "changes the experimental protocol, not just the "
                             "speed, and would make an Icentia11k matched-target "
                             "result incomparable to MIT-BIH's. Matches the exact "
                             "warning already documented in run_icentia11k_study.py "
                             "for the same parameter -- try --batch_size first, "
                             "which has no such tradeoff. Default None (no cap, "
                             "full local epochs, matching every other domain).")
    args = parser.parse_args()

    K = args.K if args.K is not None else DEFAULT_K[args.domain]
    lr = args.lr if args.lr is not None else DEFAULT_LR[args.domain]
    target = args.target if args.target is not None else DEFAULT_TARGET[args.domain]
    if args.target is None and not args.probe_only:
        print(f"  NOTE: --target not set, using uncalibrated placeholder "
              f"{target:.2f}. Run with --probe_only first to check this is "
              f"reachable before trusting a 'did not reach' result.")

    meta, clients, model_fn_base, input_shape, n_classes = DOMAIN_BUILDERS[args.domain](
        args.data_dir, model="tiny")
    print(f"Domain: {args.domain}  Clients: {len(clients)}  "
          f"Input shape: {input_shape}  Classes: {n_classes}")

    os.makedirs(args.out_dir, exist_ok=True)

    models_to_run = ("tiny", "resnet") if args.model == "both" else (args.model,)
    for model_name in models_to_run:
        _, _, model_fn, _, _ = DOMAIN_BUILDERS[args.domain](args.data_dir, model=model_name)
        n_params = sum(p.numel() for p in model_fn().parameters())
        print(f"\n=== Model: {model_name} ({n_params} parameters) ===")
        for seed in range(args.n_seeds):
            fname = f"{args.domain}_{model_name}_seed{seed:02d}.json"
            fpath = os.path.join(args.out_dir, fname)
            if os.path.exists(fpath):
                print(f"  seed {seed:2d}  [already complete, skipping]")
                continue
            t0 = time.time()
            max_r = args.max_rounds if not args.probe_only else args.max_rounds
            result = run_to_target(
                clients, model_fn, input_shape, n_classes,
                target=(1.01 if args.probe_only else target),  # >1.0 => never
                                                                 # triggers early
                                                                 # stop in probe mode
                max_rounds=max_r, K=K, mu=args.mu, policy_name=args.policy,
                seed=seed, epochs=args.epochs, lr=lr, batch_size=args.batch_size,
                stability_window=args.stability_window,
                max_batches_per_client=args.max_batches_per_client,
                progress_tag=f"{args.domain}/{model_name}/seed{seed}")
            elapsed = time.time() - t0
            if args.probe_only:
                print(f"  seed {seed:2d}  final balanced_acc="
                      f"{result['final_balanced_acc']:.3f}  "
                      f"raw_acc={result['final_raw_acc']:.3f}  ({elapsed:.0f}s)")
            elif result["reached_target"]:
                fmcl_kg = result["lifecycle_to_target_kg_CO2e"]["FMCL"]
                print(f"  seed {seed:2d}  reached target at round "
                      f"{result['rounds_to_target']}  FMCL={fmcl_kg:.4f} kg  "
                      f"({elapsed:.0f}s)")
            else:
                print(f"  seed {seed:2d}  DID NOT reach target={target:.2f} within "
                      f"{args.max_rounds} rounds (final balanced_acc="
                      f"{result['final_balanced_acc']:.3f})  ({elapsed:.0f}s)")
            result["domain"] = args.domain
            result["model_name"] = model_name
            with open(fpath, "w") as f:
                json.dump(result, f)

    print(f"\nDone. Results saved to {args.out_dir}")
    print("Compare tiny vs resnet: rounds_to_target, balanced_acc_at_target, "
          "and lifecycle_to_target_kg_CO2e['FMCL'] across the saved JSON files.")
