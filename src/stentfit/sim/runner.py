"""
Launch 4C, either in Docker or from a local binary.

A few things matter here:

* the output goes to the screen and to ``run.log``, because the convergence history is what is
  needed when a run fails and it scrolls off the terminal otherwise;
* the exit code is read from the process rather than from ``tee``, which swallows it;
* on macOS the run is wrapped in ``caffeinate -is``, because these runs take hours and a Mac that
  falls asleep suspends Docker's VM with them;
* the Docker platform is detected rather than assumed, so the same code works on Apple Silicon,
  Intel, Linux and Windows.

Runs are safe to launch side by side from several terminals. Each one is its own container with its
own name, each writes only inside its own folder, and a lock file stops two processes solving the
same input into the same output files.
"""

import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------------------
# How 4C is launched
# --------------------------------------------------------------------------------------

RUNNER_BACKENDS = ("auto", "docker", "local")

#: The exact 4C build every result in this project was produced with.
#:
#: Cite the digest and not the tag. ``:main`` is a rolling nightly that points somewhere else within
#: days, while the digest is that one build forever.
#:
#: The nightly is used rather than a release tag because the installed ``fourcipp`` bundles the
#: input schema for the nightly. Pinning an older release would validate input files against a
#: newer schema than the container's 4C understands, and the mismatch would pass silently.
FOUR_C_IMAGE = "ghcr.io/4c-multiphysics/4c:main"
FOUR_C_DIGEST = "sha256:17ac2c98d5d54e189c8ef2a4224731cc8cde45ab48856d8933e0d28f8fe5c6d1"
FOUR_C_VERSION = "2026.3.0-dev"
FOUR_C_EXE_IN_IMAGE = "/home/user/4C/build/4C"


@dataclass(frozen=True)
class RunnerConfig:
    """
    How the 4C executable is launched.

    :param backend: ``"docker"`` runs the pinned image, ``"local"`` runs an existing 4C binary, and
        ``"auto"`` prefers a local binary when one is configured and falls back to Docker.
    :param image: Container image. Kept as a tag for readability; ``digest`` is what identifies
        the build.
    :param digest: The image digest this project's results were produced with. Checked at
        preflight, and recorded with every run.
    :param exe_in_image: Path to the 4C executable inside the container.
    :param local_executable: Path to a 4C binary on this machine, for the ``local`` backend.
        ``None`` falls back to ``$BEAMME_FOUR_C_EXE``.
    :param platform: Docker platform string. ``None`` detects it. ``linux/amd64`` is forced only on
        an arm64 host, where the amd64-only image runs under Rosetta. On an amd64 host nothing
        needs forcing.
    :param workdir: Mount point inside the container. The host output folder appears here, which is
        why results are written straight back to your machine.
    :param mpi_ranks: Ranks for ``mpirun``. Run serial first, since MPI is the most likely thing to
        misbehave under emulation.
    :param cpus: Docker ``--cpus`` limit. Useful when several runs share one Docker VM.
    :param memory: Docker ``--memory`` limit, e.g. ``"8g"``.
    :param caffeinate: On macOS, hold off idle sleep for the length of the solve. These runs take
        hours, and a sleeping Mac suspends Docker's VM along with them. Ignored elsewhere.
    """

    backend: str = "auto"
    image: str = FOUR_C_IMAGE
    digest: str = FOUR_C_DIGEST
    exe_in_image: str = FOUR_C_EXE_IN_IMAGE
    local_executable: str | None = None
    platform: str | None = None
    workdir: str = "/work"
    mpi_ranks: int = 1
    cpus: float | None = None
    memory: str | None = None
    caffeinate: bool = True

    def __post_init__(self: "RunnerConfig") -> None:
        if self.backend not in RUNNER_BACKENDS:
            raise ValueError(f"backend must be one of {RUNNER_BACKENDS}, got {self.backend!r}")
        if self.mpi_ranks < 1:
            raise ValueError(f"mpi_ranks must be at least 1, got {self.mpi_ranks}")
        if self.cpus is not None and self.cpus <= 0:
            raise ValueError(f"cpus must be positive, got {self.cpus}")


class RunnerError(RuntimeError):
    """Raised when 4C cannot be launched at all, as opposed to running and failing."""


# --------------------------------------------------------------------------------------
# Locking, so several terminals cannot collide
# --------------------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    """
    :param pid: A process id.
    :returns: Whether a process with that id currently exists.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True          # exists, owned by someone else
    return True


class RunLock:
    """
    An exclusive lock on one run folder, held for the length of a solve.

    Created with ``O_EXCL``, so the filesystem itself arbitrates and two terminals cannot both
    take it. Without this, a second process solving the same input would write the same output
    files as the first, and both results would be nonsense.

    :param run_dir: The folder to lock.
    :param force: Take the lock even if one is present. Intended for a lock left behind by a
        process that has since died.
    """

    def __init__(self: "RunLock", run_dir, force: bool = False):
        self.path = Path(run_dir) / ".lock"
        self.force = force
        self.held = False

    def __enter__(self: "RunLock") -> "RunLock":
        if self.force and self.path.exists():
            self.path.unlink()

        try:
            handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            holder = self._read_holder()
            pid = holder.get("pid")
            stale = pid is not None and not _pid_alive(int(pid))
            raise RunnerError(
                f"{self.path.parent} is already being solved by pid {pid} "
                f"(started {holder.get('started', '?')})"
                + (" - that process is gone, so the lock is stale; pass force=True to clear it"
                   if stale else " - wait for it, or solve a different case"))

        with os.fdopen(handle, "w") as lock_file:
            lock_file.write(f"pid: {os.getpid()}\n"
                            f"started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                            f"host: {platform.node()}\n")
        self.held = True
        return self

    def __exit__(self: "RunLock", *exc) -> None:
        if self.held:
            self.path.unlink(missing_ok=True)
            self.held = False

    def _read_holder(self: "RunLock") -> dict:
        """:returns: The current lock holder's details, as far as they can be read."""
        try:
            return dict(line.split(": ", 1)
                        for line in self.path.read_text().splitlines() if ": " in line)
        except OSError:
            return {}


class NoLock:
    """A stand-in for :class:`RunLock` when locking is switched off."""

    def __enter__(self: "NoLock") -> "NoLock":
        """:returns: ``self``."""
        return self

    def __exit__(self: "NoLock", *exc) -> None:
        """Does nothing."""


def container_name(stent_name: str, run_name: str) -> str:
    """
    Build a container name Docker will accept.

    Docker allows only ``[a-zA-Z0-9][a-zA-Z0-9_.-]*``, while a stent folder is named after the
    design it came from and can hold anything a filename can. A single disallowed character makes
    Docker refuse the run, and that happens after the input files are built, so the failure looks
    like a solver problem when it is only a naming rule.

    Every disallowed character becomes ``-``. The name is only a label for ``docker ps``, and the
    run folder stays the real identifier.

    :param stent_name: The stent's folder name.
    :param run_name: The run's folder name.
    :returns: A name Docker accepts, at most 60 characters.
    """
    return re.sub(r"[^a-zA-Z0-9_.-]", "-", f"4c-{stent_name}-{run_name}")[:60]


# --------------------------------------------------------------------------------------
# Preflight reports
# --------------------------------------------------------------------------------------


def preflight_ok(report: dict) -> bool:
    """
    :param report: A report from :meth:`FourCRunner.preflight`.
    :returns: Whether every check passed.
    """
    return all(check["ok"] for check in report["checks"])


def preflight_lines(report: dict) -> list:
    """
    Format a preflight report as printable lines.

    :param report: A report from :meth:`FourCRunner.preflight`.
    :returns: One line per check, plus a header and a verdict.
    """
    checks = report["checks"]
    width = max((len(check["name"]) for check in checks), default=10)

    lines = [f"4C runner preflight  (backend: {report['backend']})"]
    for check in checks:
        lines.append(f"  [{'OK  ' if check['ok'] else 'FAIL'}] {check['name']:{width}s} "
                     f"{check['detail']}")
        if not check["ok"] and check["fix"]:
            lines.append(f"         {' ' * width} fix: {check['fix']}")
    lines.append(f"  => {'ready' if preflight_ok(report) else 'NOT ready'}")
    return lines


def print_preflight(report: dict) -> None:
    """
    Print a preflight report.

    :param report: A report from :meth:`FourCRunner.preflight`.
    """
    for line in preflight_lines(report):
        print(line)


def raise_if_not_ready(report: dict) -> None:
    """
    Stop before launching if any check failed, naming what and how to fix it.

    :param report: A report from :meth:`FourCRunner.preflight`.
    :raises RunnerError: If any check failed.
    """
    if preflight_ok(report):
        return

    failed = [f"{check['name']}: {check['detail']}"
              + (f" (fix: {check['fix']})" if check["fix"] else "")
              for check in report["checks"] if not check["ok"]]
    raise RunnerError("cannot launch 4C:\n  " + "\n  ".join(failed))


def _check(name: str, ok: bool, detail: str, fix: str = "") -> dict:
    """
    Build one preflight check result.

    :param name: What was checked.
    :param ok: Whether it is satisfied.
    :param detail: What was found.
    :param fix: The command or action that would fix it, when it is not satisfied.
    :returns: The check.
    """
    return {"name": name, "ok": bool(ok), "detail": detail, "fix": fix}


# --------------------------------------------------------------------------------------
# The runner
# --------------------------------------------------------------------------------------


def _run_text(command: list, timeout: int = 30) -> tuple:
    """
    Run a short command and capture its output.

    :param command: The command.
    :param timeout: Seconds to wait.
    :returns: ``(returncode, stripped stdout+stderr)``.
    """
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        return 1, str(error)
    return done.returncode, (done.stdout + done.stderr).strip()


class FourCRunner:
    """
    Runs 4C on an input file and writes the results next to it.

    :param config: How to launch. ``None`` uses the defaults, which auto-detect the backend.
    """

    def __init__(self: "FourCRunner", config: RunnerConfig = None):
        self.config = config or RunnerConfig()

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    def local_executable(self: "FourCRunner"):
        """:returns: The configured local 4C binary, or what the environment names, or None."""
        return self.config.local_executable or os.environ.get("BEAMME_FOUR_C_EXE")

    def backend(self: "FourCRunner") -> str:
        """
        Decide which backend to use.

        ``auto`` prefers a local binary when one is configured and exists, since running natively
        is always faster than an emulated container. Otherwise it falls back to Docker.

        :returns: ``"docker"`` or ``"local"``.
        """
        if self.config.backend != "auto":
            return self.config.backend

        executable = self.local_executable()
        return "local" if executable and Path(executable).exists() else "docker"

    def docker_platform(self: "FourCRunner"):
        """
        Decide the ``--platform`` flag.

        The 4C image is published for amd64 only. On an arm64 host it therefore has to be asked
        for explicitly and runs under emulation. On an amd64 host nothing needs forcing, and
        forcing it anyway is what tied the old scripts to Apple Silicon.

        :returns: The platform string, or ``None`` when none is needed.
        """
        if self.config.platform is not None:
            return self.config.platform
        return "linux/amd64" if platform.machine().lower() in ("arm64", "aarch64") else None

    # ------------------------------------------------------------------
    # Preflight
    # ------------------------------------------------------------------

    def preflight(self: "FourCRunner", mount=None) -> dict:
        """
        Check everything needed to launch, before spending a solve on finding out.

        Never guesses: each failing item is named along with the command that fixes it.

        :param mount: The folder that will be mounted, checked for existence and writability.
        :returns: Dict with ``backend`` and a ``checks`` list. Print it with
            :func:`print_preflight`, test it with :func:`preflight_ok`.
        """
        backend = self.backend()
        checks = []

        if backend == "local":
            executable = self.local_executable()
            checks.append(_check(
                "4C executable", bool(executable) and Path(executable or "").exists(),
                str(executable) if executable else "not set",
                "set RunnerConfig(local_executable=...) or $BEAMME_FOUR_C_EXE"))
            if self.config.mpi_ranks > 1:
                found = shutil.which("mpirun")
                checks.append(_check("mpirun", bool(found), found or "not found",
                                     "install an MPI runtime, or set mpi_ranks=1"))
        else:
            checks.extend(self._docker_checks())

        if mount is not None:
            mount = Path(mount)
            writable = mount.is_dir() and os.access(mount, os.W_OK)
            checks.append(_check(
                "mount", writable,
                f"{mount} -> {self.config.workdir}" if writable
                else f"{mount} missing or not writable",
                "build the input first, which creates the folder"))

        return {"backend": backend, "checks": checks}

    def _docker_checks(self: "FourCRunner") -> list:
        """:returns: The Docker-specific preflight checks."""
        checks = []

        found = shutil.which("docker")
        checks.append(_check("docker cli", bool(found), found or "not found",
                             "install Docker Desktop (macOS/Windows) or Docker Engine (Linux)"))
        if not found:
            return checks

        code, out = _run_text(["docker", "version", "--format", "{{.Server.Version}}"])
        checks.append(_check("docker daemon", code == 0,
                             f"server {out}" if code == 0 else "not reachable",
                             "start Docker Desktop, or `sudo systemctl start docker`"))
        if code != 0:
            return checks

        host = platform.machine().lower()
        wanted = self.docker_platform()
        checks.append(_check("host / image", True,
                             f"{host} -> {wanted or 'native'}"
                             + (" (emulated, expect it to be slower)" if wanted else "")))

        code, out = _run_text(["docker", "image", "inspect", self.config.image,
                               "--format", "{{index .RepoDigests 0}}"])
        present = code == 0
        checks.append(_check(
            "image", present,
            f"{self.config.image} present" if present else f"{self.config.image} not pulled",
            f"docker pull --platform {wanted or 'linux/amd64'} {self.config.image}"))

        if present:
            digest = out.split("@")[-1] if "@" in out else out
            matches = digest == self.config.digest
            # A mismatch is a warning in substance but a failure in form: :main is a rolling
            # nightly, so a silently different build would make the recorded digest a lie.
            checks.append(_check(
                "digest", matches,
                f"{digest[:23]}... matches the pinned {FOUR_C_VERSION} build" if matches
                else f"{digest[:23]}... differs from the pinned {self.config.digest[:23]}...",
                f"docker pull {self.config.image.split(':')[0]}@{self.config.digest}"))

        return checks

    # ------------------------------------------------------------------
    # Launching
    # ------------------------------------------------------------------

    def build_command(self: "FourCRunner", input_path, output_base, mount, name=None) -> list:
        """
        Assemble the command that runs 4C.

        Under Docker the input and output paths are made relative to the mount, because that is
        what they are called inside the container.

        :param input_path: The ``.4C.yaml`` to solve.
        :param output_base: Base path 4C writes its results under.
        :param mount: The folder made visible to the container.
        :param name: Container name, so ``docker ps`` is readable and a run can be stopped.
        :returns: The command, as a list of arguments.
        """
        config = self.config

        if self.backend() == "local":
            command = [str(self.local_executable()), str(input_path), str(output_base)]
            if config.mpi_ranks > 1:
                command = ["mpirun", "-np", str(config.mpi_ranks)] + command
            return command

        command = ["docker", "run", "--rm"]
        if name:
            command += ["--name", name]
        wanted = self.docker_platform()
        if wanted:
            command += ["--platform", wanted]
        if config.cpus:
            command += ["--cpus", str(config.cpus)]
        if config.memory:
            command += ["--memory", str(config.memory)]
        command += ["-v", f"{mount}:{config.workdir}", "-w", config.workdir, config.image]

        inner = [config.exe_in_image,
                 str(Path(input_path).relative_to(mount)),
                 str(Path(output_base).relative_to(mount))]
        if config.mpi_ranks > 1:
            inner = ["mpirun", "-np", str(config.mpi_ranks)] + inner
        return command + inner

    def run(self: "FourCRunner", input_path, output_base, mount=None, log_path=None,
            name=None, lock: bool = True, force: bool = False,
            check_preflight: bool = True) -> dict:
        """
        Solve one input file.

        :param input_path: The ``.4C.yaml`` to solve.
        :param output_base: Base path for 4C's results, normally inside the run folder.
        :param mount: Folder to make visible to the container. ``None`` mounts the input's own
            folder, which is enough for every case here.
        :param log_path: Where to tee the output. ``None`` writes ``run.log`` beside the input.
        :param name: Container name. ``None`` derives one from the run folder.
        :param lock: Take an exclusive lock on the run folder for the duration.
        :param force: Clear a stale lock first.
        :param check_preflight: Verify the backend before launching.
        :returns: Dict with ``ok``, ``returncode``, ``seconds``, ``log``, ``command``, ``backend``.
        :raises RunnerError: If the launch itself is impossible, or the folder is locked.
        """
        input_path = Path(input_path).resolve()
        run_dir = input_path.parent
        output_base = Path(output_base)
        if not output_base.is_absolute():
            output_base = run_dir / output_base
        mount = Path(mount).resolve() if mount else run_dir
        log_path = Path(log_path) if log_path else run_dir / "run.log"
        name = name or container_name(run_dir.parent.name, run_dir.name)

        if not input_path.exists():
            raise RunnerError(f"no input file at {input_path}")

        if check_preflight:
            raise_if_not_ready(self.preflight(mount))

        command = self.build_command(input_path, output_base, mount, name=name)
        if self.config.caffeinate and platform.system() == "Darwin" and shutil.which("caffeinate"):
            # -i blocks idle sleep, -s blocks system sleep on AC. Deliberately not -d: the display
            # is free to switch off, since that costs nothing and stops nothing. The assertion is
            # released when the command returns, so there is nothing to undo.
            command = ["caffeinate", "-is"] + command

        run_dir.mkdir(parents=True, exist_ok=True)
        keeper = RunLock(run_dir, force=force) if lock else NoLock()

        with keeper:
            print(f"[4c] {' '.join(command)}")
            print(f"[4c] log -> {log_path}")
            started = time.monotonic()
            returncode = stream_to_log(command, log_path)
            seconds = time.monotonic() - started

        ok = returncode == 0
        print(f"[4c] {'OK' if ok else f'FAILED (exit {returncode})'} in {seconds:.0f}s")
        if not ok:
            print(f"[4c] see {log_path}")

        return {"ok": ok, "returncode": returncode, "seconds": round(seconds, 1),
                "log": str(log_path), "command": " ".join(command), "backend": self.backend()}


def stream_to_log(command: list, log_path) -> int:
    """
    Run a command, sending its output to the screen and to a file at once.

    Done by hand rather than with a shell pipe to ``tee``, because a pipe hides the exit code of
    the process that matters.

    :param command: The command.
    :param log_path: Where to write the log.
    :returns: The exit code.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return process.wait()
