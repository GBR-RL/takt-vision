"""Charts for the README and docs, rendered in light and dark variants.

  overload  End-to-end latency over time for each ingress policy (JSON-lines from takt_run --jsonl)
      python scripts/plot_latency.py overload --run block=results/block.jsonl \\
          --run drop-newest=results/drop.jsonl --run latest=results/latest.jsonl --out docs/assets/overload

  compare   Per-stage p50 latency, C++ vs Python (JSON from takt_bench / python_baseline.py)
      python scripts/plot_latency.py compare --run "C++ (takt)=results/cpp.json" \\
          --run "Python=results/python.json" --out docs/assets/cpp_vs_python

Each command writes <out>-light.png and <out>-dark.png.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference data-viz palette: first three categorical slots, validated per mode.
THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
              "series": ["#2a78d6", "#eb6834", "#1baf7a"]},
    "dark": {"surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#3a3937",
             "series": ["#3987e5", "#d95926", "#199e70"]},
}


def parse_runs(items):
    runs = []
    for item in items:
        label, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--run expects label=path, got {item!r}")
        runs.append((label, Path(path)))
    return runs


def style_axes(ax, theme):
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["grid"])
    ax.tick_params(colors=theme["muted"], labelsize=9)
    ax.grid(True, axis="y", color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def new_figure(theme, size=(8, 4.2)):
    fig, ax = plt.subplots(figsize=size, dpi=150)
    fig.patch.set_facecolor(theme["surface"])
    style_axes(ax, theme)
    return fig, ax


def finish(fig, ax, theme, title, subtitle, out: Path, mode: str):
    fig.suptitle(title, x=0.06, y=0.97, ha="left", fontsize=12, fontweight="bold", color=theme["text"])
    fig.text(0.06, 0.905, subtitle, ha="left", fontsize=9, color=theme["muted"])
    legend = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for text in legend.get_texts():
        text.set_color(theme["text"])
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out.parent.mkdir(parents=True, exist_ok=True)
    path = out.with_name(f"{out.name}-{mode}.png")
    fig.savefig(path, facecolor=theme["surface"])
    plt.close(fig)
    print(f"wrote {path}")


def overload(args):
    runs = []
    for label, path in parse_runs(args.run):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        runs.append((label, [r["t_ms"] / 1000 for r in rows], [r["e2e_ms"] for r in rows]))

    for mode, theme in THEMES.items():
        fig, ax = new_figure(theme)
        for (label, t, e2e), color in zip(runs, theme["series"]):
            ax.plot(t, e2e, color=color, linewidth=2, label=label)
            ax.annotate(f"{label}  {e2e[-1]:,.0f} ms", xy=(t[-1], e2e[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9, color=theme["text"])
        ax.set_yscale("log")
        ax.set_xlabel("time since start (s)", color=theme["muted"], fontsize=9)
        ax.set_ylabel("end-to-end latency (ms, log scale)", color=theme["muted"], fontsize=9)
        ax.margins(x=0.02)
        right = max(t[-1] for _, t, _ in runs)
        ax.set_xlim(0, right * 1.22)  # room for the end labels
        finish(fig, ax, theme, args.title, args.subtitle, Path(args.out), mode)


def compare(args):
    runs = [(label, json.loads(path.read_text())) for label, path in parse_runs(args.run)]
    # Inference is the same engine on both sides, so it is reported in the subtitle rather than
    # drawn: at ~55 ms it would flatten the stages the implementations actually differ in.
    stages = ["preprocess", "postprocess", "pre + post"]

    def p50(data, stage):
        if stage == "pre + post":
            return data["stages"]["preprocess"]["p50_ms"] + data["stages"]["postprocess"]["p50_ms"]
        return data["stages"][stage]["p50_ms"]

    inference = ", ".join(f"{label} {data['stages']['inference']['p50_ms']:.1f} ms" for label, data in runs)
    subtitle = f"{args.subtitle}\nInference is the same engine in both and not drawn: {inference}.".strip()

    for mode, theme in THEMES.items():
        fig, ax = new_figure(theme, size=(8, 3.8))
        n = len(runs)
        height = 0.8 / n
        for i, ((label, data), color) in enumerate(zip(runs, theme["series"])):
            ys = [s + (i - (n - 1) / 2) * height for s in range(len(stages))]
            values = [p50(data, s) for s in stages]
            ax.barh(ys, values, height=height * 0.92, color=color, label=label)
            for stage, y, v in zip(stages, ys, values):
                text = f"{v:.2f} ms"
                if n == 2 and i == 0:  # first run is the reference: state its speed-up
                    text += f"  ·  {p50(runs[1][1], stage) / v:.1f}x faster"
                ax.annotate(text, xy=(v, y), xytext=(4, 0), textcoords="offset points",
                            va="center", fontsize=8, color=theme["text"],
                            fontweight="bold" if n == 2 and i == 0 else "normal")
        ax.set_yticks(range(len(stages)), stages)
        ax.invert_yaxis()
        ax.grid(True, axis="x", color=theme["grid"], linewidth=0.8)
        ax.grid(False, axis="y")
        ax.set_xlabel("p50 latency per frame (ms)", color=theme["muted"], fontsize=9)
        ax.margins(x=0.30)
        fig.suptitle(args.title, x=0.06, y=0.97, ha="left", fontsize=12, fontweight="bold", color=theme["text"])
        fig.text(0.06, 0.915, subtitle, ha="left", va="top", fontsize=8.5, color=theme["muted"],
                 linespacing=1.5)
        legend = ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0),
                           ncol=len(runs))
        for text in legend.get_texts():
            text.set_color(theme["text"])
        fig.tight_layout(rect=(0, 0, 1, 0.80))
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        path = out.with_name(f"{out.name}-{mode}.png")
        fig.savefig(path, facecolor=theme["surface"])
        plt.close(fig)
        print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn, title in (
        ("overload", overload, "Latency when the detector is slower than the camera"),
        ("compare", compare, "Pre- and post-processing: C++ vs Python, same ONNX model"),
    ):
        p = sub.add_parser(name)
        p.add_argument("--run", action="append", required=True, help="label=path (repeatable)")
        p.add_argument("--out", required=True, help="output path without extension")
        p.add_argument("--title", default=title)
        p.add_argument("--subtitle", default="")
        p.set_defaults(fn=fn)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
