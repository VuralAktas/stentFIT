"""
The one entry point: pick a simulation type, build it, run it, measure it.

    sim = Simulation(stent, sim_type="stent_only", settings=StentOnlySettings())
    sim.build_input(cases=["radial_expand"])
    sim.run()
    sim.postprocess()

Three types are available. ``stent_only`` drives every beam node by prescribed displacement,
``stent_balloon`` opens the stent through contact with a balloon, and ``stent_artery`` ties it into
a test artery with meshtying. Each one is a module in :mod:`stentfit.sim.cases`, and none of them
knows about the others.

Everything they share lives in :mod:`stentfit.sim`: the meshing, the materials, the coupling rules,
the runner and the run record.
"""

from pathlib import Path

import yaml

from .sim import beam_model as bm
from .sim import results as _results
from .sim.materials import section_properties
from .sim.settings import (SETTINGS_CLASSES, default_settings, resolve_strut_thickness,
                           settings_from_dict, settings_to_dict)
from .sim.cases import BuildContext, get_builder
from .sim.coupling import print_coupling
from .sim.record import (plain, read_run_parameters, update_run_results, write_run_index,
                         write_run_parameters, write_text_atomic)
from .sim.runner import FourCRunner, RunnerConfig
from .stent import Stent

#: Where results go by default, next to the skeletonisation output.
DEFAULT_OUTPUT_DIR = "examples/data/output/simulation"

#: Header written above every ``results/summary.yaml``.
_SUMMARY_HEADER = ("# Headline results for one 4C run, written by stentFIT.\n"
                   "#\n"
                   "# The full per-step history is in metrics.csv beside this file.\n\n")


def _balloon_arg(record: dict):
    """
    The balloon geometry the result measurements need, pulled out of a run record.

    :param record: A parsed ``run_parameters.yaml``.
    :returns: ``{"r_inner": mm, "r_outer": mm}``, or ``None`` for a type with no balloon.
    """
    balloon = record.get("balloon") or {}
    if "r_inner_mm" not in balloon or "r_outer_mm" not in balloon:
        return None
    return {"r_inner": float(balloon["r_inner_mm"]), "r_outer": float(balloon["r_outer_mm"])}


def _pressure_arg(record: dict):
    """
    The load profile the ``pressure_MPa`` column needs, pulled out of a run record.

    Taken from the record rather than from ``self.settings`` so that re-measuring an old run
    reports the pressure that run was actually built with.

    :param record: A parsed ``run_parameters.yaml``.
    :returns: ``{"max": MPa, "expression": str}``, or ``None`` for a run with no balloon pressure.
    """
    balloon = record.get("balloon") or {}
    loading = record.get("loading") or {}
    if "pressure_max_MPa" not in balloon:
        return None
    expression = loading.get("scale_of_t") or balloon.get("load_expression")
    if not expression:
        return None
    return {"max": float(balloon["pressure_max_MPa"]), "expression": str(expression)}


#: Files a run writes alongside its input: the separate bodies, and the warped stent. They are 4C
#: inputs in their own right, but none of them is the file the run solves.
INTERMEDIATE_INPUTS = ("balloon.4C.yaml", "artery_solid.4C.yaml", "artery_stent.4C.yaml",
                       "stent_warped.4C.yaml")


def _sole_input(run_dir: Path) -> Path:
    """
    Work out which ``.4C.yaml`` in a folder is the one 4C solves.

    Only needed for folders built before the run record started naming it. Records written now
    carry an ``input`` block, and this is not called.

    :param run_dir: The run folder.
    :returns: The input file.
    :raises FileNotFoundError: If there is not exactly one candidate.
    """
    candidates = [p for p in sorted(run_dir.glob("*.4C.yaml"))
                  if p.name not in INTERMEDIATE_INPUTS]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"cannot tell which input {run_dir} solves - found {len(candidates)} candidates. "
            f"Rebuild it, and the record will name the file.")
    return candidates[0]


class Simulation:
    """
    One stent, one simulation type, from input file to measured result.

    :param stent: The stent to simulate. Its skeletonisation must have run.
    :param sim_type: ``"stent_only"``, ``"stent_balloon"`` or ``"stent_artery"``.
    :param settings: Every parameter of the simulation, from :mod:`stentfit.sim.settings`.
        ``None`` uses that type's defaults. Must match ``sim_type``.
    :param output_dir: Root for results. The type and the stent name are appended, so several
        types and stents coexist.
    :param runner: How 4C is launched.
    :param artery: The artery, required by ``stent_artery``.
    :param options: Anything else the case accepts.
    :raises ValueError: If the type is unknown, if the settings are for a different type, or if
        the artery was built for a different stent.
    """

    def __init__(self: "Simulation",
                 stent: Stent,
                 sim_type: str = "stent_only",
                 settings=None,
                 output_dir=DEFAULT_OUTPUT_DIR,
                 runner: RunnerConfig = None,
                 artery=None,
                 **options):
        self.builder = get_builder(sim_type)             # raises early on a bad type
        self.stent = stent
        self.sim_type = sim_type

        if settings is None:
            settings = default_settings(sim_type)
        expected = SETTINGS_CLASSES[sim_type]
        if not isinstance(settings, expected):
            raise ValueError(
                f"sim_type {sim_type!r} needs {expected.__name__}, "
                f"got {type(settings).__name__} - the two must agree, or half your parameters "
                f"would be silently ignored")

        self.settings = settings
        self.runner = runner or RunnerConfig()
        self.output_dir = Path(output_dir) / sim_type / stent.stent_name
        self.options = dict(options)

        # The balloon is built from the settings, so the notebook has one block to read rather
        # than two objects to keep in step.
        self.balloon = None
        if sim_type == "stent_balloon":
            from .balloon import Balloon
            self.balloon = Balloon(
                stent, material=settings.balloon_material,
                clearance_frac=settings.clearance_frac, overhang_frac=settings.overhang_frac,
                wall=settings.wall, pressure_max=settings.pressure_max,
                end_spring_stiffness=settings.end_spring_stiffness,
                load_profile=settings.load_profile,
                neohooke_youngs=settings.neohooke_youngs,
                neohooke_poisson=settings.neohooke_poisson,
                fibre_longitudinal=settings.fibre_longitudinal,
                fibre_circumferential=settings.fibre_circumferential)
            self.options["balloon"] = self.balloon

        # An artery built against a different stent would pass the coupling checks against one
        # stent and be meshed around another, so catch it here rather than at the solve.
        self.artery = artery
        if artery is not None:
            if getattr(artery, "stent", stent) is not stent:
                raise ValueError(
                    f"this artery was built for a different stent "
                    f"({artery.stent.stent_name!r} vs {stent.stent_name!r}) - build it with "
                    f"Artery(stent, ...) using the same stent")
            self.options["artery"] = artery

        #: What :meth:`build_input` produced: one dict per written input.
        self.built = []
        #: What :meth:`run` produced, by case name.
        self.runs = {}

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def build_input(self: "Simulation", **options) -> list:
        """
        Build the 4C input files for this simulation.

        Each case gets its own folder, holding the input, a mesh preview and a
        ``run_parameters.yaml`` recording every parameter that determined it. The record is
        written now, not after the solve, so a run that later fails is still identifiable.

        :param options: Extra case options for this call, e.g. ``cases=["radial_expand"]``.
        :returns: The input files written.
        """
        ctx = BuildContext(
            stent_dir=Path(self.stent.output_dir),
            stent_name=self.stent.stent_name,
            features=bm.load_features(self.stent.output_dir),
            output_dir=self.output_dir,
            settings=self.settings, runner=self.runner,
            options={**self.options, **options})

        self.built = self.builder(ctx)

        for case in self.built:
            case["record"].setdefault("simulation_type", self.sim_type)
            case["record"]["runner"] = settings_to_dict(self.runner)
            write_run_parameters(case["run_dir"], case["record"])
        if self.built:
            index = write_run_index(self.output_dir)
            print(f"\n[saved] {index}   (index of every run for this stent)")

        return [case["input_path"] for case in self.built]

    def find_runs(self: "Simulation", name: str = None) -> list:
        """
        Find inputs already on disk, instead of building new ones.

        This is what solves the folder that was built rather than a fresh one. Building always
        claims a new folder, so a solve that rebuilt would discard the parameters that were set and
        quietly test something else.

        Each run's own ``run_parameters.yaml`` names the file that was written, so nothing here has
        to know how a simulation type names things.

        :param name: Which folder, e.g. ``"run001"`` or ``"radial_expand"``. ``None`` takes every
            run that has no results yet, so a folder that already cost hours is never re-solved by
            accident. Name one to solve it again regardless.
        :returns: The runs found, in the same shape :meth:`build_input` returns.
        :raises FileNotFoundError: If nothing has been built for this stent and type, or if the
            folder you named is not there.
        """
        folders = []
        if (self.output_dir / "run_parameters.yaml").exists():
            folders.append(self.output_dir)          # stent_artery writes into the root itself
        if self.output_dir.is_dir():
            folders += [d for d in sorted(self.output_dir.iterdir())
                        if d.is_dir() and (d / "run_parameters.yaml").exists()]

        if not folders:
            raise FileNotFoundError(
                f"no runs built under {self.output_dir} - build one first:\n"
                f"  python -m stentfit.run build {self.sim_type} {self.stent.stent_name}")

        if name is not None:
            folders = [d for d in folders if d.name == name]
            if not folders:
                raise FileNotFoundError(f"no run named {name!r} under {self.output_dir}")
        else:
            solved = [d for d in folders if (d / "run.log").exists()]
            folders = [d for d in folders if d not in solved]
            if not folders:
                raise FileNotFoundError(
                    f"every run under {self.output_dir} has results already "
                    f"({', '.join(d.name for d in solved)}). Name one to solve it again.")

        self.built = []
        for run_dir in folders:
            record = read_run_parameters(run_dir)
            written = record.get("input") or {}
            input_path = run_dir / written.get("file", "")
            if not input_path.is_file():
                input_path = _sole_input(run_dir)
            self.built.append({
                "name": run_dir.name, "input_path": input_path, "run_dir": run_dir,
                "output_base": written.get("output_base", f"out_{run_dir.name}"),
                "record": record, "coupling": None})

        print(f"found {len(self.built)} run(s): "
              + ", ".join(c["name"] for c in self.built))
        return self.built

    def check(self: "Simulation") -> list:
        """
        Report the coupling verdicts from the last build.

        The build already refuses to write an input that violates them, so this is for looking at
        rather than for gating.

        :returns: One report per case that has one.
        :raises ValueError: If nothing has been built yet.
        """
        if not self.built:
            raise ValueError("nothing built yet - call build_input() first")
        reports = []
        for case in self.built:
            if case["coupling"] is None:
                # A type with no solid body has no beam-to-solid coupling to check. Saying so is
                # better than printing nothing and leaving it ambiguous whether the checks passed
                # or never ran.
                print(f"{case['name']}: no beam-to-solid coupling in this simulation type "
                      f"(no solid body), so there is nothing to check")
                continue
            print(f"{case['name']}:")
            print_coupling(case["coupling"])
            reports.append(case["coupling"])
        return reports

    def run(self: "Simulation", cases: list = None, runner: FourCRunner = None,
            force: bool = False) -> dict:
        """
        Solve the built inputs.

        Cases run one after another, and a failure does not stop the rest, so one case diverging
        does not cost the others. Each takes an exclusive lock on its own folder, so this is safe to
        launch from several terminals at once.

        :param cases: Which built cases to run. ``None`` runs all of them.
        :param runner: A runner to use. ``None`` builds one from this simulation's config.
        :param force: Clear a stale lock left by a dead process.
        :returns: The result per case name.
        :raises ValueError: If nothing has been built yet.
        """
        if not self.built:
            raise ValueError("nothing built yet - call build_input() first")

        runner = runner or FourCRunner(self.runner)
        wanted = [c for c in self.built if cases is None or c["name"] in cases]

        for case in wanted:
            print(f"\n{'=' * 78}\n=== {case['name']}\n{'=' * 78}")
            result = runner.run(case["input_path"], case["output_base"], force=force)
            self.runs[case["name"]] = result
            status = ("converged" if result["ok"]
                      else f"failed (exit {result['returncode']})")
            update_run_results(case["run_dir"], {"status": status, **result})

        write_run_index(self.output_dir)
        ok = sum(1 for r in self.runs.values() if r["ok"])
        print(f"\n{ok}/{len(self.runs)} case(s) finished cleanly")
        return self.runs

    def postprocess(self: "Simulation", cases: list = None) -> dict:
        """
        Measure every solved case and write its metrics.

        :param cases: Which cases to measure. ``None`` measures all built ones.
        :returns: Per-step metrics by case name, for cases that have results.
        :raises ValueError: If nothing has been built yet.
        """
        if not self.built:
            raise ValueError("nothing built yet - call build_input() first")

        junctions = bm.read_junctions(self.stent.output_dir)
        # The thickness the run was *built* at, which is settings.strut_thickness whenever that
        # overrides the measurement. Reading the features file directly would measure utilisation
        # and contact penetration against a section the solve never used, so M/Mp and
        # penetration-per-strut would be normalised by the wrong strut.
        strut = resolve_strut_thickness(self.settings,
                                        bm.load_features(self.stent.output_dir))
        section = section_properties(strut, self.settings)

        out = {}
        for case in self.built:
            if cases is not None and case["name"] not in cases:
                continue

            # Both blocks are absent for a type with no balloon, and process_case then simply
            # leaves out the columns that would have needed them.
            record = case.get("record") or {}
            balloon = _balloon_arg(record)
            pressure = _pressure_arg(record)

            rows = _results.process_case(case["run_dir"], junctions, section,
                                         balloon=balloon, pressure=pressure)
            if not rows:
                print(f"[skip] {case['name']}: no results yet")
                continue
            out[case["name"]] = rows

            results_dir = case["run_dir"] / "results"
            csv = _results.write_csv(results_dir / "metrics.csv", rows)

            # Earlier versions wrote metrics.csv into the run folder itself. Leaving it there
            # beside the new one means two files with the same name and different contents,
            # and no way to tell which a plot came from.
            stale = case["run_dir"] / "metrics.csv"
            if stale.is_file():
                stale.unlink()
            summary = _results.summarise(rows)
            back = _results.recoil(rows)

            print(f"\n{case['name']}: {summary['steps']} steps, "
                  f"t_end {summary['t_end']:.2f}")
            print(f"  diameter       {rows[0]['diameter']:.4f} -> "
                  f"{summary['diameter_mm']:.4f} mm "
                  f"({summary['radial_strain'] * 100:+.2f}%)")
            if "diameter_peak_mm" in summary:
                peak = f"  peak           {summary['diameter_peak_mm']:.4f} mm"
                if "pressure_at_peak_MPa" in summary:
                    peak += f" at {summary['pressure_at_peak_MPa']:.4f} MPa"
                print(peak)
            print(f"  foreshortening {summary['foreshortening_pct']:+.2f}%")
            if "balloon_dogboning_pct_at_peak" in summary:
                print(f"  dogboning      {summary['balloon_dogboning_pct_at_peak']:+.2f}% balloon, "
                      f"{summary['stent_dogboning_pct_at_peak']:+.2f}% stent "
                      f"(tip {summary['balloon_r_tip_ratio_at_peak']:.2f}x)")
            if "penetration_per_strut_at_peak" in summary:
                print(f"  contact        struts sink {summary['penetration_max_mm_at_peak']:.4f} mm, "
                      f"{summary['penetration_per_strut_at_peak']:.2f} of a strut diameter")
            if "radial_stiffness_mm_per_MPa" in summary:
                print(f"  stiffness      {summary['radial_stiffness_mm_per_MPa']:.3f} mm/MPa")
            if "peak_M_over_Mp" in summary:
                print(f"  utilisation    M/Mp {summary['peak_M_over_Mp']:.2f} peak, "
                      f"N/Np {summary.get('peak_N_over_Np', float('nan')):.2f} peak")
            if back:
                print(f"  recoil         {back['recoil_pct']:.2f}% "
                      f"(peak {back['diameter_peak_mm']:.4f} -> "
                      f"final {back['diameter_final_mm']:.4f} mm)")

            full = {**summary, **({"recoil": back} if back else {})}
            update_run_results(case["run_dir"], full)
            write_text_atomic(results_dir / "summary.yaml",
                              _SUMMARY_HEADER + yaml.safe_dump(plain(full), sort_keys=False))
            print(f"  [saved] {csv.parent.name}/")

            if balloon:
                profile = _results.profile_table(case["run_dir"], rows, balloon)
                if profile:
                    _results.write_profile_csv(results_dir / "balloon_profile.csv", profile)
                    print(f"  [saved] {csv.parent.name}/balloon_profile.csv   "
                          f"({len(profile['curves'])} pressures, the paper's Fig. 5)")

        write_run_index(self.output_dir)
        return out

    def setup(self: "Simulation", **options) -> "Simulation":
        """
        Build, check, run and measure, in that order.

        :param options: Extra case options, passed to :meth:`build_input`.
        :returns: ``self``.
        """
        self.build_input(**options)
        self.check()
        self.run()
        self.postprocess()
        return self

    # ------------------------------------------------------------------
    # Reproducing an earlier run
    # ------------------------------------------------------------------

    @classmethod
    def from_run(cls, run_dir, stent: Stent = None) -> "Simulation":
        """
        Rebuild a simulation from a recorded run.

        Every run writes the parameters that determined it, so a result can be reproduced without
        remembering what was typed.

        :param run_dir: A run folder holding a ``run_parameters.yaml``.
        :param stent: The stent. ``None`` loads it from the path in the record.
        :returns: A simulation configured as that run was.
        :raises FileNotFoundError: If there is no record there.
        """
        record = read_run_parameters(run_dir)
        sim_type = record.get("simulation_type", "stent_only")
        settings = settings_from_dict(SETTINGS_CLASSES[sim_type], record.get("settings"))
        runner = settings_from_dict(RunnerConfig, record.get("runner"))

        if stent is None:
            source = (record.get("stent") or {}).get("source")
            if not source:
                raise ValueError(f"{run_dir} records no stent source - pass stent=...")
            stent = Stent.load(source)

        run_path = Path(run_dir)
        # output_dir/<type>/<stent>/<run>, so climb back to output_dir.
        root = run_path.parent.parent.parent

        return cls(stent, sim_type, settings=settings, output_dir=root, runner=runner)

    def __repr__(self: "Simulation") -> str:
        """:returns: A short summary of how far this simulation has got."""
        bits = []
        if self.built:
            bits.append(f"{len(self.built)} case(s) built")
        if self.runs:
            bits.append(f"{sum(1 for r in self.runs.values() if r['ok'])}/{len(self.runs)} run")
        stage = ", ".join(bits) or "not built"
        return (f"<Simulation {self.sim_type} {self.stent.stent_name!r} "
                f"[{stage}] -> {self.output_dir}>")
