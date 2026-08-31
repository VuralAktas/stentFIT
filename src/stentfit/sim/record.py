"""
Run folders, run records, and the index across them.

Three jobs, all of them safe to run from several terminals at once:

* allocate the next ``runNNN`` folder without two processes taking the same number;
* write ``run_parameters.yaml`` at build time, so a run that later fails is still identifiable;
* rebuild ``runs_summary.csv``, an index over every run for one stent.

The index is rebuilt from the records rather than appended to. Appending would fix the columns at
whatever mattered when the first run was built, and what distinguishes runs changes as a study goes
on. Rebuilding also means the table cannot drift from the records, since it holds no content of its
own.
"""

import csv
import io
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import yaml

#: Header written above every run record.
_HEADER = ("# 4C simulation run record\n"
           "#\n"
           "# Every parameter that determined this result, written at build time by stentFIT.\n"
           "# The 'results' section is filled in after the solve.\n\n")


def write_text_atomic(path: Path, text: str) -> Path:
    """
    Write a file so a reader never sees it half-written.

    Written to a temporary file in the same directory, then moved into place. ``os.replace`` is
    atomic within a filesystem, so a concurrent rebuild of the same file ends with one complete
    version rather than two interleaved ones.

    :param path: Where to write.
    :param text: What to write.
    :returns: The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def next_run_dir(stent_out: str | Path, prefix: str = "run") -> Path:
    """
    Create the next numbered run folder.

    Every build gets its own folder, so no two runs can overwrite each other. The number is claimed
    by creating the directory with ``exist_ok=False``, in a loop, because reading the highest
    existing number and adding one would let two terminals pick the same name. Here the filesystem
    arbitrates and the loser moves on to the next number.

    :param stent_out: The stent's output folder.
    :param prefix: Folder name prefix.
    :returns: The new folder, created.
    :raises OSError: If no free number could be claimed.
    """
    stent_out = Path(stent_out)
    stent_out.mkdir(parents=True, exist_ok=True)

    existing = [int(p.name[len(prefix):]) for p in stent_out.glob(f"{prefix}[0-9]*")
                if p.name[len(prefix):].isdigit()]
    candidate = max(existing, default=0) + 1

    for number in range(candidate, candidate + 1000):
        run_dir = stent_out / f"{prefix}{number:03d}"
        try:
            run_dir.mkdir(exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise OSError(f"could not claim a free {prefix}NNN folder under {stent_out}")


def plain(value: Any) -> Any:
    """
    Convert a value into something ``yaml.safe_dump`` can write.

    The record is assembled from measurements, settings and reports, and a numpy scalar can arrive
    from any of them, because ``stent_features`` holds ``np.float64``. ``safe_dump`` refuses
    anything it has no tag for, and it would fail the build after the input file had already been
    written. Converting at this one boundary means no caller has to remember.

    :param value: Anything that might go into the record.
    :returns: The same data as plain Python types.
    """
    if isinstance(value, dict):
        return {plain(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):        # np.float64, np.int64, np.bool_, ...
        return value.item()
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def write_run_parameters(run_dir: str | Path, params: dict[str, Any]) -> Path:
    """
    Record every parameter that determined this run.

    Written at build time, so it exists whether or not the solve succeeds. A failed run is worth
    telling apart from other failed runs too.

    :param run_dir: This run's folder.
    :param params: Nested dict of parameters.
    :returns: The path written.
    """
    run_dir = Path(run_dir)
    body = yaml.safe_dump(plain(params), sort_keys=False, default_flow_style=False)
    return write_text_atomic(run_dir / "run_parameters.yaml", _HEADER + body)


def read_run_parameters(run_dir: str | Path) -> dict[str, Any]:
    """
    Read a run's record back.

    :param run_dir: The run folder, or the record file itself.
    :returns: The parsed record.
    :raises FileNotFoundError: If there is no record there.
    """
    path = Path(run_dir)
    if path.is_dir():
        path = path / "run_parameters.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no run_parameters.yaml at {path}")
    return yaml.safe_load(path.read_text()) or {}


def update_run_results(run_dir: str | Path, results: dict[str, Any]) -> Path:
    """
    Fill in a record's ``results`` section after the solve.

    :param run_dir: This run's folder.
    :param results: What the run produced.
    :returns: The path written.
    """
    params = read_run_parameters(run_dir)
    params["results"] = results
    return write_run_parameters(run_dir, params)


def _first(*values: Any) -> Any:
    """
    The first value that was actually recorded.

    ``or`` cannot be used for this: ``0``, ``0.0`` and ``""`` are all legitimate recorded values
    and all falsy, so a zero penalty or a zero Poisson's ratio would silently fall through to the
    next candidate.

    :param values: Candidates, most authoritative first.
    :returns: The first that is not ``None``, or ``None``.
    """
    for value in values:
        if value is not None:
            return value
    return None


def _cell(value: Any) -> str:
    """
    Render one recorded value as a CSV cell.

    A number is written plainly, so a column of them sorts and plots as numbers rather than as
    text. Anything absent becomes an empty cell, because a parameter a simulation type does not have
    is genuinely blank. Numeric strings are converted too, since YAML 1.1 only recognises an
    exponent that carries a sign, so a value like ``2.0e5`` round-trips as text.

    :param value: Whatever was recorded, or ``None``.
    :returns: The cell.
    """
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:g}"
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


class _Blocks:
    """
    One parsed record, split into the blocks the column extractors read.

    Every block defaults to an empty dict, so a record written before a section existed -- or by a
    simulation type that has no balloon at all -- still produces a full row instead of raising.

    :param name: The run folder's name.
    :param params: The parsed record.
    """

    def __init__(self: "_Blocks", name: str, params: dict[str, Any]) -> None:
        block = lambda key: params.get(key) or {}

        self.name = name
        self.params = params
        self.run = block("run")
        self.results = block("results")
        self.stent = block("stent")
        self.beam = block("beam_model")
        self.balloon = block("balloon")
        self.loading = block("loading")
        self.contact = block("contact")
        self.case = block("case")
        self.solver_4c = block("solver_4c")
        self.coupling = block("coupling_constraints")
        self.artery = block("artery")
        # The flat settings block is where every parameter lives now. ``beam`` is the older shape,
        # kept because the index exists to describe history.
        self.settings = params.get("settings") or block("beam")

    def param(self: "_Blocks", key: str, *fallbacks: Any) -> Any:
        """
        A settings field, falling back to wherever older records kept it.

        :param key: The settings field name.
        :param fallbacks: Values from other blocks, tried in order.
        :returns: The value, or ``None``.
        """
        return _first(self.settings.get(key), *fallbacks)

    def mortar(self: "_Blocks", key: str) -> Any:
        """
        A mortar-only field, blank unless this run actually used mortar.

        The two keys are written into the 4C input only under mortar -- 4C's own default for both
        is ``none``, which has no meaning at runtime -- so reporting them for a GPTS run would
        describe a setting that never reached the solver.

        :param key: The settings field name.
        :returns: The value, or ``None`` under any other discretization.
        """
        if self.param("discretization", self.contact.get("discretization")) != "mortar":
            return None
        return self.settings.get(key)

    def stent_only(self: "_Blocks", key: str) -> Any:
        """
        A field only ``stent_only`` actually reads, blank for every other type.

        ``l_el_per_strut`` is on every settings class so the beam fields keep the same shape, but
        only ``stent_only`` derives its element length from it. The types with a solid body take the
        beam length from the coupling rule instead, so reporting it for them would put a number in
        the table that reached nothing.

        :param key: The settings field name.
        :returns: The value, or ``None`` for any other simulation type.
        """
        if self.params.get("simulation_type") != "stent_only":
            return None
        return self.settings.get(key)

    def beam_length(self: "_Blocks", key: str) -> Any:
        """
        One end of the beam element length spread that was actually meshed.

        The target is a single number but what comes out is a range, because the mesher divides
        each strut curve into a whole number of elements, so a curve whose length is not a multiple
        of the target lands either side of it. It is the shortest element that decides whether the
        coupling rule holds.

        :param key: ``"min"``, ``"mean"`` or ``"max"``.
        :returns: The length in mm, or ``None`` if this record never measured it.
        """
        return (self.beam.get("actual_element_length_mm") or {}).get(key)

    def solid_size(self: "_Blocks", key: str) -> Any:
        """
        One direction of the solid body's element size.

        This is per direction rather than one number, because the mesher rounds the circumference
        and the length to whole element counts independently, so the two in-plane sizes land either
        side of the target. The radial one is the wall thickness, which the coupling rule does not
        apply to.

        :param key: ``"circumferential"``, ``"axial"`` or ``"radial"``.
        :returns: The size in mm, or ``None`` for a type with no solid body.
        """
        return (self.balloon.get("element_size_mm") or {}).get(key)


#: Index columns, as ``(header, extractor)``.
#:
#: Everything that can differ between two runs of the same stent is here, because what
#: distinguishes them changes as a study goes on. A parameter a type does not have comes out blank,
#: which is how one table covers all three simulation types.
_COLUMN_SPECS = [
    ("run",                        lambda r: r.name or r.run.get("folder")),
    ("built",                      lambda r: r.run.get("date")),
    ("status",                     lambda r: _first(r.results.get("status"),
                                                    r.run.get("status"))),
    ("case",                       lambda r: _first(r.case.get("name"),
                                                    r.params.get("simulation_type"))),
    ("beam material",              lambda r: r.param("material", r.beam.get("material_law"))),
    # The element formulation, not the material. The two are independent choices, and a run that
    # differs only in this one is a different simulation with an identical-looking settings block.
    ("beam class",                 lambda r: r.param("beam_class", r.beam.get("element"))),
    ("balloon material",           lambda r: r.param("balloon_material",
                                                     r.balloon.get("material"))),
    ("load profile",               lambda r: r.param("load_profile", r.loading.get("profile"),
                                                     r.balloon.get("load_profile"))),

    # --- stent ------------------------------------------------------------------------
    ("stent E [MPa]",              lambda r: r.param("youngs",
                                                     r.beam.get("youngs_modulus_MPa"))),
    # Zero, not blank, for an elastic run: the field is defined for it and its value is that
    # there is no yield limit at all.
    ("stent yield [MPa]",          lambda r: (r.settings.get("yield_strength")
                                              if r.settings.get("material") == "elastoplastic"
                                              else 0)),
    ("strut thickness [mm]",       lambda r: _first(r.stent.get("strut_thickness_mm"),
                                                    r.settings.get("strut_thickness"))),

    # --- balloon ----------------------------------------------------------------------
    ("balloon clearance [x strut]", lambda r: r.param("clearance_frac",
                                                      r.balloon.get("clearance_frac"))),
    # What the clearance fraction actually came to on this stent, surface to surface. The fraction
    # alone does not say: it is measured from the beam mesh's true innermost node, which sits a
    # different distance inside the averaged r_inner on every stent.
    ("initial clearance [mm]",     lambda r: r.params.get("initial_clearance_mm")),
    ("balloon overhang [x length]", lambda r: r.param("overhang_frac",
                                                      r.balloon.get("overhang_frac"))),
    ("balloon wall [mm]",          lambda r: r.param("wall", r.balloon.get("wall_mm"))),
    ("balloon radial strain",      lambda r: r.param(
        "radial_strain", r.loading.get("target_stent_radial_strain"))),
    ("balloon neohooke E [MPa]",   lambda r: r.param("neohooke_youngs",
                                                     r.balloon.get("neohooke_youngs_MPa"))),
    ("balloon end spring [MPa/mm]", lambda r: r.param(
        "end_spring_stiffness", r.balloon.get("end_spring_stiffness_MPa_per_mm"))),
    ("balloon pressure max [MPa]", lambda r: r.param("pressure_max",
                                                     r.balloon.get("pressure_max_MPa"))),

    # --- contact ----------------------------------------------------------------------
    ("penalty law",                lambda r: r.param("penalty_law")),
    ("penalty [N/mm^2]",           lambda r: r.param("penalty")),
    ("penalty g0 [x strut]",       lambda r: r.param(
        "penalty_g0_per_strut", r.contact.get("penalty_g0_per_strut"))),
    ("discretization",             lambda r: r.param("discretization",
                                                     r.contact.get("discretization"))),
    ("gauss points",               lambda r: r.param("gauss_points")),
    ("contact type",               lambda r: r.param("contact_type")),
    ("mortar shape",               lambda r: r.mortar("mortar_shape_function")),
    ("mortar defined in",          lambda r: r.mortar("mortar_contact_defined_in")),

    # --- stent_only's own knobs -------------------------------------------------------
    # Blank on every other type. Without these the stent-only table says nothing about how that
    # type is meshed, gripped, or whether its struts could pass through each other.
    ("self contact",               lambda r: _first(r.settings.get("self_contact"),
                                                    r.contact.get("enabled"))),
    ("grip [x tip ring]",          lambda r: r.settings.get("grip_frac")),
    ("l_el [x strut]",             lambda r: r.stent_only("l_el_per_strut")),

    # --- mesh -------------------------------------------------------------------------
    # The two factors first, then every length in mm they actually produced. The factors are
    # ratios and mean nothing on their own: the same 1.5 is a different element on every stent,
    # because both meshes are scaled by that stent's strut thickness. The mm columns are the
    # measured mesh, not the target -- both meshers round to a whole number of elements, so what
    # comes out scatters around what was asked for.
    #
    # The last three are the inputs to the coupling rules, kept beside the sizes they are checked
    # against: solid element >= beam diameter, and E_beam / E_solid >= 10.
    ("factor solid/diameter",      lambda r: r.param("factor_solid")),
    ("factor beam/solid",          lambda r: r.param("factor_beam")),
    ("h beam max [mm]",            lambda r: r.beam_length("max")),
    ("h beam min [mm]",            lambda r: r.beam_length("min")),
    ("h beam mean [mm]",           lambda r: r.beam_length("mean")),
    ("h solid circ [mm]",          lambda r: r.solid_size("circumferential")),
    ("h solid axial [mm]",         lambda r: r.solid_size("axial")),
    ("h solid radial [mm]",        lambda r: r.solid_size("radial")),
    ("beam diameter [mm]",         lambda r: _first(r.coupling.get("d_beam"),
                                                    r.stent.get("strut_thickness_mm"))),
    ("beam stiffness [MPa]",       lambda r: r.param("youngs",
                                                     r.beam.get("youngs_modulus_MPa"))),
    ("solid stiffness [MPa]",      lambda r: _first(r.balloon.get("youngs_modulus_MPa"),
                                                    r.artery.get("youngs_modulus_MPa"))),
    ("beam elements",              lambda r: r.beam.get("n_elements")),

    # --- solver -----------------------------------------------------------------------
    ("steps",                      lambda r: r.param("n_steps")),
    ("max iter",                   lambda r: r.param("max_iter")),
    ("tol residual",               lambda r: r.param("tol_residuum")),
    ("tol increment",              lambda r: r.param("tol_increment")),
    ("predictor",                  lambda r: r.param("predictor")),
    ("line search",                lambda r: r.param("line_search")),

    # --- provenance -------------------------------------------------------------------
    # Written in full, not truncated: ``:main`` is a rolling nightly that points somewhere else
    # within days, so the digest is the only thing that names one build forever, and a shortened
    # one cannot be cited.
    ("4C digest",                  lambda r: r.solver_4c.get("digest")),
]

#: Just the headers, in order.
_COLUMNS = [header for header, _ in _COLUMN_SPECS]


def _row(name: str, params: dict[str, Any]) -> list[str]:
    """
    Reduce one run record to an index row.

    Every field is looked up defensively: a record written before a field existed, or by a type
    that never had it, must still produce a row of the right width, not an exception.

    :param name: The run folder's name.
    :param params: The parsed record.
    :returns: The row's cells.
    """
    blocks = _Blocks(name, params)
    return [_cell(extract(blocks)) for _, extract in _COLUMN_SPECS]


def write_run_index(stent_out: str | Path, name: str = "runs_summary.csv") -> Path:
    """
    Rebuild the index over every run for one stent, as CSV.

    Sequential folder names are unambiguous but opaque, so this is the lookup that makes them
    usable. It writes one line per run, carrying every parameter that could differ between two of
    them, so a folder can be identified without opening a single record. It is CSV so that it opens
    in a spreadsheet and ``pandas.read_csv`` gives the whole study as a dataframe.

    :param stent_out: The stent's output folder.
    :param name: The index file's name.
    :returns: The path written.
    """
    stent_out = Path(stent_out)

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(_COLUMNS)

    for run_dir in sorted(p for p in stent_out.iterdir() if p.is_dir()):
        record = run_dir / "run_parameters.yaml"
        if not record.exists():
            continue
        try:
            params = yaml.safe_load(record.read_text()) or {}
        except yaml.YAMLError:
            continue
        writer.writerow(_row(run_dir.name, params))

    return write_text_atomic(stent_out / name, buffer.getvalue())


def base_record(simulation_type: str, run_dir: Path) -> dict[str, Any]:
    """
    The part of a record that every simulation type shares.

    :param simulation_type: Which type this is.
    :param run_dir: This run's folder.
    :returns: The shared fields.
    """
    from .runner import FOUR_C_DIGEST, FOUR_C_IMAGE, FOUR_C_VERSION

    return {
        "run": {"folder": run_dir.name, "date": str(date.today()), "status": "not yet run"},
        "simulation_type": simulation_type,
        "solver_4c": {"image": FOUR_C_IMAGE, "digest": FOUR_C_DIGEST,
                      "version": FOUR_C_VERSION},
    }
