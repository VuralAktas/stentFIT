"""Check the simulation input is built correctly.

Nothing here solves anything, because 4C runs for hours. What is checked is that every type
builds an input 4C's own schema accepts, that the shared rules hold, and that the numbers a
solved run is measured with mean what they say.

The skeletonisation itself is covered by the other three files, so the stent fixture is used
here only as the input a simulation is built from.
"""

import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from stentfit.core import balloon_geom
from stentfit.sim import (STENT_ONLY_ELASTIC, STENT_ONLY_PLASTIC, RunnerConfig,
                          StentBalloonSettings, StentOnlySettings, check_coupling,
                          coupling_limits, section_properties, settings_from_dict,
                          settings_to_dict)
from stentfit.sim.record import next_run_dir
from stentfit.sim.runner import (FourCRunner, RunLock, RunnerError, preflight_ok,
                                 raise_if_not_ready)


# ======================================================================================
# The strut section and the coupling rules
# ======================================================================================


def test_unknown_type_is_rejected_early():
    """An unknown simulation type fails at construction, not halfway through a build."""
    from stentfit.sim.cases import get_builder

    with pytest.raises(ValueError, match="unknown simulation_type"):
        get_builder("stent_moon")


def test_section_properties_match_hand_computation():
    """The section formulas, against values computed by hand for a known thickness."""
    thickness = 0.2
    settings = StentOnlySettings(youngs=2.0e5, yield_strength=300.0,
                                 tangent_modulus_ratio=64.0 / 380.0)
    section = section_properties(thickness, settings)

    r = 0.1
    assert section["radius"] == pytest.approx(r)
    assert section["area"] == pytest.approx(np.pi * r ** 2)
    assert section["mom"] == pytest.approx(np.pi * r ** 4 / 4)
    assert section["EA"] == pytest.approx(2.0e5 * np.pi * r ** 2)
    assert section["Np"] == pytest.approx(300.0 * np.pi * r ** 2)
    # First yield, sigma_y * pi * r^3 / 4 -- not the fully plastic moment, which is 1.70x larger.
    assert section["Mp"] == pytest.approx(300.0 * np.pi * r ** 3 / 4)
    ratio = 64.0 / 380.0
    assert section["ISOHARDM"] == pytest.approx(
        2.0e5 * ratio / (1 - ratio) * np.pi * r ** 4 / 4)


def test_yield_moment_is_onset_not_fully_plastic():
    """
    ``Mp`` must be the onset of yielding, which is what 4C's ``YIELDM`` means.

    Using the fully plastic moment would keep the struts elastic 1.70x too long and then yield
    them abruptly.
    """
    section = section_properties(0.2, StentOnlySettings(yield_strength=300.0))
    fully_plastic = 300.0 * 4.0 / 3.0 * 0.1 ** 3
    assert section["Mp"] < fully_plastic
    assert fully_plastic / section["Mp"] == pytest.approx(16 / (3 * np.pi), rel=1e-6)


def test_coupling_limits_derive_both_meshes_from_one_number():
    """Both element sizes follow from the strut thickness and the two factors."""
    limits = coupling_limits(0.1, StentBalloonSettings(factor_solid=1.5, factor_beam=1.2))
    assert limits["d_beam"] == pytest.approx(0.1)
    assert limits["h_solid"] == pytest.approx(0.15)
    assert limits["l_beam"] == pytest.approx(0.18)
    assert limits["l_beam_min"] == pytest.approx(0.15)
    assert limits["l_beam_optimal_max"] == pytest.approx(0.90)
    assert limits["l_beam_max"] == pytest.approx(1.20)


def test_check_coupling_passes_in_the_good_band():
    """A mesh comfortably inside every band passes and is flagged optimal."""
    report = check_coupling([0.2, 0.25], {"circumferential": 0.15, "axial": 0.15},
                            d_beam=0.1, e_beam=2.0e5, e_solid=17.0)
    assert report["all_passed"] and report["all_optimal"]


@pytest.mark.parametrize("lengths,sizes,e_solid,failing", [
    ([0.2], {"solid": 0.05}, 17.0, "solid element >= beam diameter, solid"),   # rule 1
    ([0.10], {"solid": 0.15}, 17.0, "shortest beam / solid element"),          # rule 2 low
    ([1.5], {"solid": 0.15}, 17.0, "longest beam / solid element"),            # rule 2 high
    ([0.2], {"solid": 0.15}, 5.0e4, "beam / solid stiffness"),                 # rule 3
])


def test_check_coupling_catches_each_violation(lengths, sizes, e_solid, failing):
    """Each rule fails on its own, and names itself."""
    report = check_coupling(lengths, sizes, d_beam=0.1, e_beam=2.0e5, e_solid=e_solid)
    assert not report["all_passed"]
    assert failing in [r["name"] for r in report["rules"] if not r["passed"]]


# ======================================================================================
# The balloon mesh
# ======================================================================================


def _hex_volume(points):
    """
    Signed volume of a HEX8, by splitting it into six tetrahedra.

    :param points: ``(8, 3)`` corner coordinates in 4C's node order.
    :returns: The signed volume. Negative means the element is inside out.
    """
    tets = [(0, 1, 3, 4), (1, 2, 3, 6), (1, 3, 4, 6), (1, 4, 5, 6), (3, 4, 6, 7)]
    total = 0.0
    for a, b, c, d in tets:
        total += np.dot(points[b] - points[a],
                        np.cross(points[c] - points[a], points[d] - points[a])) / 6.0
    return total


def test_balloon_elements_are_not_inside_out():
    """
    Every element must have positive volume.

    The winding has to run radial-first, because ``e_r x e_theta = e_z``. Angular-first reverses
    it and every element comes out negative, which 4C would reject only much later.
    """
    coords, conn, _ = balloon_geom.build_balloon(1.0, 1.04, -2.0, 2.0, n_circ=24, n_axial=20)
    volumes = np.array([_hex_volume(coords[cell - 1]) for cell in conn])
    assert (volumes > 0).all(), f"{(volumes <= 0).sum()} of {len(volumes)} elements inverted"


def test_balloon_tube_closes_without_a_seam():
    """
    The angular grid wraps, so there are no duplicate nodes where it meets itself.

    A seam would leave the tube open, and pressure would push it apart there.
    """
    coords, _, _ = balloon_geom.build_balloon(1.0, 1.04, -1.0, 1.0, n_circ=16, n_axial=8)
    unique = np.unique(np.round(coords, 9), axis=0)
    assert len(unique) == len(coords)


def test_balloon_radii_are_where_they_were_asked_for():
    """Inner and outer surfaces sit at the requested radii."""
    coords, _, surfaces = balloon_geom.build_balloon(1.0, 1.04, -1.0, 1.0,
                                                     n_circ=16, n_axial=8)
    inner = coords[np.array(surfaces[1]) - 1]
    outer = coords[np.array(surfaces[2]) - 1]
    assert np.hypot(inner[:, 0], inner[:, 1]) == pytest.approx(1.0)
    assert np.hypot(outer[:, 0], outer[:, 1]) == pytest.approx(1.04)


def test_fibre_directions_are_orthogonal_unit_vectors():
    """
    The two fibre families must be perpendicular, and pointing the right way.

    Ten orders of magnitude separate their stiffnesses, so swapping them turns a balloon that
    inflates into one that stretches, and nothing in the solve would say so.
    """
    coords, conn, _ = balloon_geom.build_balloon(1.0, 1.04, -1.0, 1.0, n_circ=16, n_axial=8)
    fibres = balloon_geom.fibre_directions(coords, conn)

    longitudinal, circumferential = fibres[:, 0, :], fibres[:, 1, :]
    assert np.allclose(longitudinal, [0.0, 0.0, 1.0])            # along the tube
    assert np.allclose(np.linalg.norm(circumferential, axis=1), 1.0)
    assert np.allclose(circumferential[:, 2], 0.0)               # around it, no axial part
    assert np.allclose(np.einsum("ij,ij->i", longitudinal, circumferential), 0.0)


# ======================================================================================
# Settings
# ======================================================================================


@pytest.mark.parametrize("settings", [
    StentOnlySettings(material="elastoplastic", youngs=3.8e5),
    StentBalloonSettings(discretization="mortar", n_steps=400),
    RunnerConfig(backend="local", cpus=4.0),
])


def test_configs_round_trip(settings):
    """Settings written to the run record must read back identical."""
    assert settings_from_dict(type(settings), settings_to_dict(settings)) == settings


def test_config_round_trip_survives_old_records():
    """Unknown and missing fields do not break reading a record.

    Fields come and go between versions, and the record describes what was run at the time.
    """
    assert settings_from_dict(
        StentOnlySettings,
        {"material": "elastic", "retired_field": 1}) == StentOnlySettings()
    assert settings_from_dict(StentOnlySettings, None) == StentOnlySettings()


def test_setting_one_field_leaves_the_others_alone():
    """Setting one field must not move any other.

    This is why each type has one flat settings class. If setting ``max_iter`` also moved
    ``n_steps`` and the predictor, a run would not be the one that was asked for.
    """
    from dataclasses import replace

    base = StentOnlySettings()
    changed = replace(base, max_iter=30)

    assert changed.max_iter == 30
    assert changed.n_steps == base.n_steps
    assert changed.predictor == base.predictor
    assert changed.line_search == base.line_search


@pytest.mark.parametrize("make,message", [
    (lambda: StentOnlySettings(material="plastic"), "material must be one of"),
    (lambda: StentOnlySettings(youngs=-1), "youngs must be positive"),
    (lambda: StentBalloonSettings(factor_solid=0.8), "factor_solid must be at least 1"),
    (lambda: StentBalloonSettings(factor_beam=7.0), "factor_beam must lie in the 1 to 6 band"),
    (lambda: StentBalloonSettings(discretization="mortor"), "discretization must be one of"),
    (lambda: StentOnlySettings(n_steps=0), "n_steps must be at least 1"),
    (lambda: RunnerConfig(backend="podman"), "backend must be one of"),
])


def test_configs_reject_impossible_values(make, message):
    """A bad value fails at construction, not at the solve."""
    with pytest.raises(ValueError, match=message):
        make()


def test_presets_spell_out_the_material_specific_solver():
    """A plastic run needs its solver changed along with the material, so the preset carries both.

    Switching the material alone would leave the elastic solver settings, which diverge.
    """
    assert STENT_ONLY_ELASTIC.n_steps == 20
    assert STENT_ONLY_ELASTIC.predictor == "TangDis"
    assert STENT_ONLY_PLASTIC.n_steps == 200
    assert STENT_ONLY_PLASTIC.predictor == "ConstDis"
    assert STENT_ONLY_PLASTIC.line_search == "Backtrack"


# ======================================================================================
# Run folders, locks and the 4C runner
# ======================================================================================


def test_run_folders_are_claimed_atomically(tmp_path):
    """Two builds at the same moment must not take the same folder.

    The folder is claimed by creating it, so the filesystem decides who gets the number.
    """
    with ThreadPoolExecutor(max_workers=16) as pool:
        dirs = list(pool.map(lambda _: next_run_dir(tmp_path), range(16)))
    assert len({d.name for d in dirs}) == 16
    assert all(d.is_dir() for d in dirs)


def test_lock_excludes_a_second_solver(tmp_path):
    """A locked run refuses a second process rather than letting both write the same files."""
    with RunLock(tmp_path):
        with pytest.raises(RunnerError, match="already being solved"):
            with RunLock(tmp_path):
                pass
    assert not (tmp_path / ".lock").exists()        # released on exit


def test_stale_lock_is_diagnosed_and_can_be_forced(tmp_path):
    """A lock left by a dead process says so, and can be cleared deliberately."""
    (tmp_path / ".lock").write_text("pid: 999999\nstarted: 2020-01-01 00:00:00\n")
    with pytest.raises(RunnerError, match="stale"):
        with RunLock(tmp_path):
            pass
    with RunLock(tmp_path, force=True):
        pass


def test_docker_platform_is_detected_not_assumed(monkeypatch):
    """``linux/amd64`` is forced only on an arm64 host.

    Forcing it everywhere would tie the runner to Apple Silicon.
    """
    runner = FourCRunner(RunnerConfig(backend="docker"))
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    assert runner.docker_platform() == "linux/amd64"
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    assert runner.docker_platform() is None


def test_docker_command_mounts_and_relativises(tmp_path):
    """The mount is what makes results appear on the host, and paths inside are relative to it.

    4C sees ``/work`` rather than the host path, so the arguments have to be rewritten.
    """
    runner = FourCRunner(RunnerConfig(backend="docker", platform="linux/amd64"))
    run_dir = tmp_path / "stent01" / "radial_expand"
    run_dir.mkdir(parents=True)
    command = runner.build_command(run_dir / "in.4C.yaml", run_dir / "out", tmp_path,
                                   name="4c-test")

    assert command[:3] == ["docker", "run", "--rm"]
    assert "--name" in command and "4c-test" in command
    assert f"{tmp_path}:/work" in command
    assert command[-2:] == ["stent01/radial_expand/in.4C.yaml", "stent01/radial_expand/out"]


def test_preflight_names_what_is_missing(tmp_path):
    """A failing check must say what and how to fix it, not raise a traceback."""
    report = FourCRunner(RunnerConfig(backend="local",
                                      local_executable=str(tmp_path / "nope"))).preflight()
    assert not preflight_ok(report)
    failed = [c for c in report["checks"] if not c["ok"]]
    assert failed and failed[0]["fix"]
    with pytest.raises(RunnerError, match="cannot launch 4C"):
        raise_if_not_ready(report)


def test_run_refuses_a_missing_input(tmp_path):
    """Solving a file that is not there fails immediately with a clear message."""
    with pytest.raises(RunnerError, match="no input file"):
        FourCRunner().run(tmp_path / "absent.4C.yaml", "out", check_preflight=False)


# ======================================================================================
# End to end: every type builds a schema-valid input
#
# These share the session-wide skeletonisation fixture, so the whole file costs
# one pipeline run. Nothing is solved.
# ======================================================================================


@pytest.fixture(scope="module")


def sim_output(tmp_path_factory):
    """A scratch folder shared by the build tests."""
    return tmp_path_factory.mktemp("simulation")


def test_stent_only_builds_every_case(stent, sim_output):
    """Both load cases build, validate against 4C's schema, and record themselves.

    ``dump(validate=True)`` is what makes this worth running, because a wrong key fails here
    instead of being ignored by the solver.
    """
    import yaml

    from stentfit import Simulation
    from stentfit.sim.cases.stent_only import CASES

    sim = Simulation(stent, sim_type="stent_only", output_dir=sim_output,
                     settings=StentOnlySettings(material="elastic"))
    written = sim.build_input()

    assert len(written) == len(CASES)
    for path in written:
        assert path.exists()
        sections = yaml.safe_load(path.read_text())
        assert sections["STRUCTURE ELEMENTS"] and sections["NODE COORDS"]
        # One coupling condition per crown: without them the struts are not joined and the stent
        # is not a structure.
        assert len(sections["DESIGN POINT COUPLING CONDITIONS"]) == 90
        assert (path.parent / "run_parameters.yaml").exists()

    assert (sim.output_dir / "runs_summary.csv").exists()


def test_radial_case_leaves_the_axial_direction_free(stent, sim_output):
    """A radial case must not prescribe z.

    The foreshortening is the headline result of these runs, so it has to come out as a result
    rather than being imposed.
    """
    import yaml

    from stentfit import Simulation

    sim = Simulation(stent, sim_type="stent_only", output_dir=sim_output,
                     settings=StentOnlySettings(cases=("radial_expand",)))
    path = sim.build_input()[0]
    conditions = yaml.safe_load(path.read_text())["DESIGN POINT DIRICH CONDITIONS"]

    driven = conditions[0]
    assert driven["ONOFF"][:3] == [1, 1, 0]        # x and y driven, z free
    # Exactly one node is pinned in z, to remove the last rigid-body mode and nothing more.
    pinned = conditions[1]
    assert pinned["ONOFF"][:3] == [0, 0, 1]


def test_stent_balloon_builds_with_contact(stent, sim_output):
    """The coupled input carries contact, a follower pressure, and LOADLIN for it."""
    import yaml

    from stentfit import Balloon, Simulation

    sim = Simulation(stent, sim_type="stent_balloon", output_dir=sim_output)
    path = sim.build_input()[0]

    sections = yaml.safe_load(path.read_text())
    contact = sections["BEAM INTERACTION/BEAM TO SOLID SURFACE CONTACT"]
    assert contact["CONTACT_TYPE"] == "gap_variation"      # 4C's default "none" aborts at runtime
    assert contact["PENALTY_PARAMETER"] == 10.0            # Datz et al.
    # A follower load depends on the displacement, so it belongs in the stiffness matrix; 4C
    # aborts on orthopressure without this.
    assert sections["STRUCTURAL DYNAMIC"]["LOADLIN"] is True
    assert any(bc.get("TYPE") == "orthopressure"
               for bc in sections["DESIGN SURF NEUMANN CONDITIONS"])
    assert "DESIGN SURF ROBIN SPRING DASHPOT CONDITIONS" in sections


def test_balloon_sits_inside_the_stent_without_touching(stent, tmp_path):
    """The balloon is placed from the innermost beam node, not from the averaged ``r_inner``.

    How far the innermost node sits inside that average changes from stent to stent, and a
    balloon that starts overlapping fails for reasons that have nothing to do with the solver.
    """
    from stentfit import Balloon
    from stentfit.sim import (build_stent_beams, centreline_nodes, coupling_limits,
                              innermost_surface_radius, load_features)

    features = load_features(stent.output_dir)
    thickness = features["strut_thickness"]
    limits = coupling_limits(thickness, StentBalloonSettings())
    mesh, section = build_stent_beams(stent.output_dir, StentOnlySettings(),
                                      l_el=limits["l_beam"], strut_thickness=thickness)
    face = innermost_surface_radius(centreline_nodes(mesh), section)

    balloon = Balloon(stent, clearance_frac=0.1)
    balloon.mesh_solid(tmp_path / "balloon.4C.yaml", stent_inner_face=face, limits=limits)

    clearance = face - balloon.r_outer
    assert clearance > 0
    assert clearance == pytest.approx(0.1 * thickness)
    assert balloon.info["wall"] == pytest.approx(0.04)


def test_simulation_rejects_settings_for_another_type(stent, sim_output):
    """The settings class and the simulation type have to agree.

    Balloon settings handed to a stent-only run would leave most parameters unread, so half of
    what was asked for would quietly not happen.
    """
    from stentfit import Simulation

    with pytest.raises(ValueError, match="needs StentOnlySettings"):
        Simulation(stent, sim_type="stent_only", settings=StentBalloonSettings(),
                   output_dir=sim_output)


def test_from_run_reproduces_the_configuration(stent, sim_output):
    """A recorded run reads back as the configuration that produced it."""
    from stentfit import Simulation

    settings = StentOnlySettings(material="elastoplastic", youngs=3.8e5,
                                 l_el_per_strut=0.35, cases=("radial_expand",),
                                 n_steps=200, predictor="ConstDis",
                                 line_search="Backtrack")
    sim = Simulation(stent, sim_type="stent_only", output_dir=sim_output, settings=settings)
    written = sim.build_input()

    restored = Simulation.from_run(written[0].parent, stent=stent)
    assert restored.sim_type == "stent_only"
    assert restored.settings == settings


# ======================================================================================
# Measuring a solved run
# ======================================================================================


def test_alignment_removes_rigid_motion_and_nothing_else():
    """A stent that has only moved measures as unchanged, and one that has deformed does not.

    The stent is held by only a few points so the radial expansion stays free, which leaves the
    rigid-body modes alive. Radius measured about a fixed axis would read a tilt as diameter.
    """
    import numpy as np

    from stentfit.sim.results import align_to_reference

    rng = np.random.default_rng(0)
    reference = rng.normal(size=(200, 3))

    angle = 0.3
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                         [np.sin(angle), np.cos(angle), 0.0],
                         [0.0, 0.0, 1.0]])
    moved = reference @ rotation.T + np.array([0.5, -0.2, 1.0])
    assert align_to_reference(moved, reference) == pytest.approx(reference, abs=1e-9)

    # a real deformation has to survive: scaling out by 10% must still read as 10%.
    # Measured about the centroid, since removing the translation is half of what
    # alignment is for.
    stretched = reference * 1.1
    recovered = align_to_reference(stretched, reference)
    centre = reference.mean(axis=0)
    assert np.linalg.norm(recovered - centre, axis=1) == pytest.approx(
        1.1 * np.linalg.norm(reference - centre, axis=1), rel=1e-9)


def test_contact_gap_is_signed_from_the_strut_surface():
    """Penalty contact always overlaps, so the sign has to be unambiguous.

    The beam is a centreline and its contact geometry is a tube of the section radius around it,
    so the gap is measured from that surface.
    """
    import numpy as np

    from stentfit.sim.results import contact_gap

    balloon = np.array([[0.0, 0.0, 0.0]])
    radius = 0.05
    beams = np.array([[0.20, 0.0, 0.0],      # clear of the surface by 0.15
                      [0.05, 0.0, 0.0],      # exactly touching
                      [0.01, 0.0, 0.0]])     # sunk 0.04 in

    gap = contact_gap(beams, balloon, radius)
    assert gap[0] == pytest.approx(0.15)
    assert gap[1] == pytest.approx(0.0, abs=1e-12)
    assert gap[2] == pytest.approx(-0.04)


def test_recoil_is_measured_on_the_aligned_diameter():
    """Recoil has to ignore rigid-body drift.

    Otherwise a perfectly elastic stent, which recovers everything, looks partly permanent.
    """
    from stentfit.sim.results import recoil

    rows = [{"time": 0.0, "diameter": 3.0, "diameter_aligned": 3.0},
            {"time": 0.5, "diameter": 4.0, "diameter_aligned": 4.0},
            {"time": 1.0, "diameter": 3.1, "diameter_aligned": 3.0}]

    assert recoil(rows)["recoil_pct"] == pytest.approx(100.0)

    # with no aligned column it falls back, so old records still measure
    raw = [{k: v for k, v in row.items() if k != "diameter_aligned"} for row in rows]
    assert recoil(raw)["recoil_pct"] == pytest.approx(90.0)
