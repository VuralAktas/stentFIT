"""
Command line for the simulation pipeline.

Long solves belong in a terminal rather than in a notebook cell, and several can run at once::

    python -m stentfit.run doctor
    python -m stentfit.run build  stent_balloon stent01
    python -m stentfit.run solve  stent_balloon stent01              # the newest run folder
    python -m stentfit.run solve  stent_balloon stent01 run001       # that one
    python -m stentfit.run report stent_balloon stent01 run001
    python -m stentfit.run all    stent_only    stent01

``build`` is the only command that creates a folder. ``solve`` and ``report`` work on what is
already on disk, so the run that was built, with the parameters that were set, is the one that gets
solved. Naming no folder takes the newest.

Each run locks its own folder, so launching from several terminals cannot collide.
"""

import argparse
import sys
from pathlib import Path

from .sim.runner import FourCRunner, RunnerConfig, preflight_ok, print_preflight
from .sim.settings import (StentArterySettings, StentBalloonSettings, StentOnlySettings)
from .simulation import DEFAULT_OUTPUT_DIR, Simulation
from .stent import Stent

DEFAULT_STENT_ROOT = "examples/data/output/stent_skeleton"


def _parser() -> argparse.ArgumentParser:
    """:returns: The argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m stentfit.run",
        description="Build, solve and measure 4C stent simulations.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check the 4C runner and stop")

    for name, help_text in [("build", "write the 4C input files"),
                            ("solve", "run 4C on already-built inputs"),
                            ("report", "measure solved results"),
                            ("all", "build, solve and report")]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("type", choices=["stent_only", "stent_balloon", "stent_artery"])
        p.add_argument("stent", help="stent name, e.g. stent01")
        if name in ("solve", "report"):
            p.add_argument("run", nargs="?", default=None,
                           help="which run folder, e.g. run001 or radial_expand. "
                                "Omitted takes the newest.")
        p.add_argument("--stent-root", default=DEFAULT_STENT_ROOT,
                       help="where skeletonisation output lives")
        p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)

        # Parameters belong only to the commands that build. On solve and report they would be
        # read and then ignored, which would look like a test that never happened.
        if name in ("build", "all"):
            p.add_argument("--material", choices=["elastic", "elastoplastic"],
                           default="elastic")
            p.add_argument("--case", action="append", dest="cases",
                           help="stent_only: which load case (repeatable)")
            p.add_argument("--l-el-per-strut", type=float, default=1.0)
            p.add_argument("--factor-solid", type=float, default=1.5)
            p.add_argument("--factor-beam", type=float, default=1.2)
            p.add_argument("--balloon-material",
                           choices=["orthotropic", "isotropic"], default="orthotropic")
            p.add_argument("--pressure", type=float, default=0.3,
                           help="peak balloon pressure in MPa")
            p.add_argument("--load-profile",
                           choices=["ramp_inflate", "ramp_inflate_deflate",
                                    "parabola_inflate_deflate"],
                           default="parabola_inflate_deflate")
            p.add_argument("--artery-type", choices=["straight", "curved", "s_bend"],
                           default="curved")

        p.add_argument("--backend", choices=["auto", "docker", "local"], default="auto")
        p.add_argument("--cpus", type=float, default=None, help="docker --cpus limit")
        p.add_argument("--memory", default=None, help="docker --memory limit, e.g. 8g")
        p.add_argument("--mpi", type=int, default=1, dest="mpi_ranks")
        p.add_argument("--force", action="store_true",
                       help="clear a stale lock left by a dead process")
    return parser


def _simulation(args) -> Simulation:
    """
    Build a Simulation from the parsed arguments.

    Every parameter flag lands in the type's settings object, so the CLI builds what was asked
    for. Those flags exist only on ``build`` and ``all``. ``solve`` and ``report`` read the settings
    back from the run folder instead, so they take the defaults here and never use them.

    :param args: Parsed arguments.
    :returns: The simulation.
    """
    stent = Stent.load(str(Path(args.stent_root) / args.stent))

    get = lambda name, default: getattr(args, name, default)
    shared = dict(material=get("material", "elastic"),
                  l_el_per_strut=get("l_el_per_strut", 1.0))
    extras = {}

    if args.type == "stent_only":
        settings = StentOnlySettings(cases=tuple(get("cases", None) or ()), **shared)
    elif args.type == "stent_balloon":
        settings = StentBalloonSettings(
            balloon_material=get("balloon_material", "orthotropic"),
            pressure_max=get("pressure", 0.3),
            load_profile=get("load_profile", "parabola_inflate_deflate"),
            factor_solid=get("factor_solid", 1.5), factor_beam=get("factor_beam", 1.2),
            **shared)
    else:
        from .artery import Artery
        settings = StentArterySettings(factor_solid=get("factor_solid", 1.5),
                                       factor_beam=get("factor_beam", 1.2), **shared)
        extras["artery"] = Artery(stent, artery_type=get("artery_type", "curved"))

    return Simulation(
        stent, sim_type=args.type, settings=settings, output_dir=args.output_dir,
        runner=RunnerConfig(backend=args.backend, cpus=args.cpus, memory=args.memory,
                            mpi_ranks=args.mpi_ranks),
        **extras)


def main(argv: list = None) -> int:
    """
    Run the command line.

    :param argv: Arguments. ``None`` uses ``sys.argv``.
    :returns: Process exit code.
    """
    args = _parser().parse_args(argv)

    if args.command == "doctor":
        report = FourCRunner().preflight()
        print_preflight(report)
        return 0 if preflight_ok(report) else 1

    sim = _simulation(args)

    if args.command in ("build", "all"):
        # the only commands that create a folder
        sim.build_input()
    else:
        # solve and report work on what is already on disk, so the parameters you built with are
        # the ones that get solved
        sim.find_runs(args.run)

    if args.command in ("solve", "all"):
        sim.run(force=args.force)
    if args.command in ("report", "all"):
        sim.postprocess()

    if args.command in ("solve", "all"):
        return 0 if all(r["ok"] for r in sim.runs.values()) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
