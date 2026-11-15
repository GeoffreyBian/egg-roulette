"""python -m eggsim <command>

  human-grip   grip statistics from the OpenGraph Labs glove dataset   -> results/human_grip.json
  collect      demonstrations (teleop, glove, heavy-handed, varied eggs) -> data/demos_*.npz
  gain         headline sweep: 4 conditions x 4 data sizes x 5 seeds    -> results/gain.json
  ablations    32 configs x 5 seeds                                     -> results/ablations.json
  robustness   friction x mass grid for the n=25 policies               -> results/robustness.json
  validate     physics checks: friction cone, solver, timestep, sensors -> results/physics.json
  figures      every figure in figures/
  render       hero stills, side-by-side video and README gif           -> media/
  pixels       pixel policies: render demos, gain sweep (add mps to train on the GPU) -> results/gain_pixels.json
  web-export   scene, policies and parity references for the browser demo -> web/assets, web/test
  all          everything above, in order
"""
import sys


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "human-grip":
        from .human_grip import main as f; f()
    elif cmd == "collect":
        from .collect import collect; collect(sys.argv[2:] or None)
    elif cmd in ("gain", "ablations", "robustness"):
        from . import experiments; getattr(experiments, cmd)()
    elif cmd == "validate":
        from .validate import main as f; f()
    elif cmd == "figures":
        from .figures import main as f; f(sys.argv[2:] or None)
    elif cmd == "render":
        from .render import main as f; f(sys.argv[2:] or None)
    elif cmd == "pixels":
        from . import pixels
        sub = sys.argv[2] if len(sys.argv) > 2 else "gain"
        if sub == "render":
            pixels.render()
        elif sub == "mps":
            pixels.mps_trainer()
        else:
            pixels.gain(workers=int(sys.argv[3]) if len(sys.argv) > 3 else None)
    elif cmd == "web-export":
        from .web_export import main as f; f(sys.argv[2:] or None)
    elif cmd == "all":
        for c in ("human-grip", "collect", "gain", "ablations", "robustness", "validate", "figures", "render"):
            sys.argv = [sys.argv[0], c]; main()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
