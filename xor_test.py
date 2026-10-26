"""
Milestone 1: can ONE neuron solve XOR?
  - Point neuron (weighted sum + threshold)      -> should fail (max 3/4 correct)
  - Active dendrite neuron (branches spike first) -> should succeed (4/4)

Milestone 2: gradient check on the surrogate backward pass.

Run:  python xor_test.py
Outputs: xor_decision_boundary.png, xor_results.txt
"""
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from active_dendrite import ActiveDendriteLayer

# ---- XOR truth table ------------------------------------------------------
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = torch.tensor([[0.], [1.], [1.], [0.]])   # "exactly one input on"

SEEDS = 20
EPOCHS = 2000
NUM_BRANCHES = 8
THRESHOLD = 0.5

lines = []
def log(s=""):
    print(s)
    lines.append(s)


def train(model, seed):
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.BCEWithLogitsLoss()   # output > 0 means "predict 1"
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model(X), Y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        correct = ((model(X) > 0).float() == Y).sum().item()
    return int(correct), loss.item()


# ---- Milestone 1: XOR over many random seeds -------------------------------
log("=" * 62)
log("MILESTONE 1: single-neuron XOR  (%d random seeds each)" % SEEDS)
log("=" * 62)

results = {"Point neuron": [], "Active dendrite neuron (%d branches)" % NUM_BRANCHES: []}
best = {}
for seed in range(SEEDS):
    torch.manual_seed(seed)
    point = nn.Linear(2, 1)
    c, l = train(point, seed)
    results["Point neuron"].append(c)
    if "point" not in best or c > best["point"][0]:
        best["point"] = (c, point)

    torch.manual_seed(seed)
    active = ActiveDendriteLayer(in_features=2, num_dendrites=NUM_BRANCHES,
                                 threshold=THRESHOLD, scale=10.0)
    c, l = train(active, seed)
    results["Active dendrite neuron (%d branches)" % NUM_BRANCHES].append(c)
    if "active" not in best or c > best["active"][0]:
        best["active"] = (c, active)

for name, scores in results.items():
    solved = sum(s == 4 for s in scores)
    log(f"{name:38s} solved XOR in {solved:2d}/{SEEDS} seeds | "
        f"best = {max(scores)}/4 correct")

# Robustness: how many seeds succeed under other settings we tried
log()
log("Robustness (active neuron, %d seeds each):" % SEEDS)
for nb, th in [(4, 1.0), (8, 1.0), (NUM_BRANCHES, THRESHOLD)]:
    ok = 0
    for seed in range(SEEDS):
        torch.manual_seed(seed)
        ok += train(ActiveDendriteLayer(2, nb, threshold=th, scale=10.0), seed)[0] == 4
    log(f"  {nb} branches, threshold {th}: solved {ok:2d}/{SEEDS}")
log("  -> fewer branches / higher threshold leaves 'dead' branches that never")
log("     fire, so their surrogate gradient is tiny and they never learn.")

a = best["active"][1]
with torch.no_grad():
    spikes = (X @ a.w_dend.t() - a.threshold > 0).int()
    out = a(X)
log()
log("Best active neuron, inside view (which branches spiked):")
log("  x1 x2 | target | branch spikes | soma output | prediction")
for i in range(4):
    log(f"   {int(X[i,0])}  {int(X[i,1])} |   {int(Y[i,0])}    | {spikes[i].tolist()} |"
        f"   {out[i,0].item():+6.2f}    |     {int(out[i,0] > 0)}")


# ---- Milestone 2: gradient check ------------------------------------------
log()
log("=" * 62)
log("MILESTONE 2: gradient check (torch.autograd.gradcheck, float64)")
log("=" * 62)

from torch.func import functional_call

torch.manual_seed(0)
layer = ActiveDendriteLayer(2, NUM_BRANCHES, threshold=THRESHOLD).double()
names = [n for n, _ in layer.named_parameters()]
params = tuple(p.detach().clone().requires_grad_(True) for p in layer.parameters())
x = torch.randn(8, 2, dtype=torch.float64, requires_grad=True)

def f(x, *ps):
    return functional_call(layer, dict(zip(names, ps)), (x,))

# (a) Smooth twin: forward = u/(1+k|u|), whose exact derivative IS our surrogate.
#     Passing means our backward() formula + the chain rule through both
#     matmuls are implemented correctly.
layer.smooth = True
ok_smooth = torch.autograd.gradcheck(f, (x, *params), eps=1e-6, atol=1e-4)
log(f"(a) Backward pass vs. finite differences (smooth twin): "
    f"{'PASSED' if ok_smooth else 'FAILED'}")

# (b) Real step function: gradcheck SHOULD fail. The true slope of a step is 0,
#     so finite differences see ~0 while our surrogate reports a nonzero slope.
#     This is the whole reason surrogate gradients exist.
layer.smooth = False
ok_step = torch.autograd.gradcheck(f, (x, *params), eps=1e-6, atol=1e-4,
                                   raise_exception=False)
log(f"(b) Same check on the hard step:                       "
    f"{'PASSED' if ok_step else 'FAILED (expected: true gradient of a step is 0)'}")

with open("xor_results.txt", "w") as fh:
    fh.write("\n".join(lines) + "\n")


# ---- Figure: decision regions ---------------------------------------------
gx, gy = torch.meshgrid(torch.linspace(-0.3, 1.3, 400),
                        torch.linspace(-0.3, 1.3, 400), indexing="xy")
grid = torch.stack([gx.flatten(), gy.flatten()], dim=1)

INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
REGION = ListedColormap([SURF, "#b7d3f6"])   # surface = predicts 0, blue = predicts 1

fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), facecolor=SURF)
panels = [
    (best["point"][1], f"Point neuron: {best['point'][0]}/4 correct",
     "One straight line can never isolate (0,1) and (1,0)"),
    (best["active"][1], f"Active dendrite neuron: {best['active'][0]}/4 correct",
     "Branch spikes carve a non-linear region"),
]
for ax, (m, title, sub) in zip(axes, panels):
    with torch.no_grad():
        z = (m(grid) > 0).float().reshape(gx.shape)
    ax.set_facecolor(SURF)
    ax.pcolormesh(gx, gy, z, cmap=REGION, shading="auto", vmin=0, vmax=1)
    ax.contour(gx, gy, z, levels=[0.5], colors="#2a78d6", linewidths=2)
    for (x1, x2), t in zip(X.tolist(), Y.flatten().tolist()):
        if t == 1:
            ax.scatter(x1, x2, s=160, c=INK, marker="o", zorder=3,
                       edgecolors=SURF, linewidths=2)
        else:
            ax.scatter(x1, x2, s=160, facecolors=SURF, edgecolors=INK,
                       marker="o", linewidths=2, zorder=3)
        ax.annotate(f"XOR={int(t)}", (x1, x2), xytext=(0, -24 if x2 == 0 else 16),
                    textcoords="offset points", fontsize=9, color=INK2, ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec="none", alpha=0.85))
    ax.set_title(title, fontsize=12, color=INK, loc="left", pad=22)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9, color=INK2)
    ax.set_xlabel("Input x1", color=INK2)
    ax.set_ylabel("Input x2", color=INK2)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    for s in ax.spines.values():
        s.set_color("#d6d5d0")
    ax.tick_params(colors=INK2)
    ax.set_aspect("equal")

fig.text(0.01, 0.01,
         "Shaded = neuron predicts 1.  Filled dot = XOR target 1, hollow = target 0.  "
         f"Best of {SEEDS} seeds per model, {EPOCHS} Adam steps.",
         fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig("xor_decision_boundary.png", dpi=200, facecolor=SURF)
print("\nSaved xor_decision_boundary.png and xor_results.txt")
