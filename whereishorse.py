from __future__ import annotations
import argparse
import getpass
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta, time as dtime
from pathlib import Path
from typing import List

# macOS / PyObjC
from Foundation import NSObject, NSRunLoop, NSDate
from AppKit import (
    NSWorkspace,
    NSWorkspaceDidWakeNotification,
    NSWorkspaceScreensDidWakeNotification,
)
import objc
import yaml
from zoneinfo import ZoneInfo
import fcntl  # singleton lock

# ---------- Paths ----------
CFG_DIR = Path.home() / ".whereishorse"
CFG_PATH = CFG_DIR / "config.yaml"
LAUNCHAGENT_PATH = Path.home() / "Library" / "LaunchAgents" / "com.whereishorse.agent.plist"
LOG_DIR = Path.home() / "Library" / "Logs" / "whereishorse"
LOCK_PATH = CFG_DIR / ".lock"

# packaged support files (installed from your wheel's resources/)
SUPPORT_DIR = Path(__file__).resolve().parent / "whereishorse_support"
TPL_CONFIG = SUPPORT_DIR / "config.yaml"
TPL_PLIST = SUPPORT_DIR / "com.whereishorse.agent.plist"
TPL_HORSE = SUPPORT_DIR / "horse.png"
TPL_EMPTY = SUPPORT_DIR / "empty.png"

LABEL = "com.whereishorse.agent"

# ---------- Logging ----------
log = logging.getLogger("whereishorse")


def _setup_logging(verbose: bool):
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(
        logging.Formatter(
            "%(asctime)s [whereishorse] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    for old in list(log.handlers):
        log.removeHandler(old)
    log.addHandler(h)


# ---------- Config ----------
@dataclass
class ScheduleCfg:
    tz: str = "Europe/Warsaw"  # IANA tz
    cutoff_hour: int = 3       # boundary hour (03:00 local)


@dataclass
class Cfg:
    names: List[str]
    me: str
    seed: str
    schedule: ScheduleCfg
    horse_wallpaper: str
    empty_wallpaper: str
    verbose: bool


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def _expand(p: str | None) -> str:
    return str(Path(os.path.expanduser(p or "")).resolve())


def _load_cfg(path: Path) -> Cfg:
    d = _read_yaml(path)

    names = d.get("names") or ["jpawlowski", "milski"]
    if isinstance(names, str):
        names = [x.strip() for x in names.split(",") if x.strip()]

    me = (d.get("me") or getpass.getuser()).strip()
    seed = (d.get("seed") or "whereishorse-seed-1").strip()

    sched = d.get("schedule") or {}
    schedule = ScheduleCfg(
        tz=sched.get("tz") or "Europe/Warsaw",
        cutoff_hour=int(sched.get("cutoff_hour") or 3),
    )

    wp = d.get("wallpaper") or {}
    horse_wp = _expand(wp.get("horse") or "~/.whereishorse/horse.png")
    empty_wp = _expand(wp.get("empty") or "~/.whereishorse/empty.png")

    verbose = bool(d.get("verbose", True))

    return Cfg(
        names=names,
        me=me,
        seed=seed,
        schedule=schedule,
        horse_wallpaper=horse_wp,
        empty_wallpaper=empty_wp,
        verbose=verbose,
    )


# ---------- Deterministic daily holder ----------
def _slot_daily(now_utc: datetime, tz: ZoneInfo, cutoff_hour: int) -> int:
    """
    Compute a daily slot index that flips at cutoff_hour in the given tz.
    Uses the "shift by cutoff" trick so DST days are handled.
    """
    now_local = now_utc.astimezone(tz)
    shifted = now_local - timedelta(hours=cutoff_hour)
    return shifted.date().toordinal()


def _holder_for(slot: int, names: List[str], seed: str) -> str:
    h = hashlib.blake2b(
        f"{seed}|{slot}|{','.join(names)}".encode(),
        digest_size=8,
    ).digest()
    return names[int.from_bytes(h, "big") % len(names)]


# ---------- Wallpaper helper (single method, as requested) ----------
def _set_wallpaper(img_path: str) -> bool:
    """
    Refresh wallpaper for all Spaces/Displays using the single, simple command:
      plist="$HOME/Library/Application Support/com.apple.wallpaper/Store/Index.plist"
      /usr/libexec/PlistBuddy -c \
        "set AllSpacesAndDisplays:Desktop:Content:Choices:0:Files:0:relative file:///$img" \
        "$plist" && \
      killall WallpaperAgent
    """
    img_path = _expand(img_path)
    plist = os.path.expanduser(
        "~/Library/Application Support/com.apple.wallpaper/Store/Index.plist"
    )

    # Build the exact argument expected by PlistBuddy (file:///… URI)
    file_uri = Path(img_path).resolve().as_uri()  # e.g. file:///Users/you/.whereishorse/horse.png
    cmd = [
        "/usr/libexec/PlistBuddy",
        "-c",
        f"set AllSpacesAndDisplays:Desktop:Content:Choices:0:Files:0:relative {file_uri}",
        plist,
    ]

    r1 = subprocess.run(cmd, capture_output=True, text=True)
    if r1.returncode != 0:
        log.error(
            "PlistBuddy failed (rc=%s): %s",
            r1.returncode,
            (r1.stderr or r1.stdout or "").strip(),
        )
        return False

    r2 = subprocess.run(["killall", "WallpaperAgent"], capture_output=True, text=True)
    if r2.returncode != 0:
        log.error(
            "killall WallpaperAgent failed (rc=%s): %s",
            r2.returncode,
            (r2.stderr or r2.stdout or "").strip(),
        )
        return False

    log.info("Wallpaper refreshed via PlistBuddy + WallpaperAgent: %s", img_path)
    return True


# ---------- Observers (wake only) ----------
class _WakeObserver(NSObject):
    def initWithCallback_(self, cb):
        self = objc.super(_WakeObserver, self).init()
        if self is None:
            return None
        self._cb = cb
        return self

    def receive_(self, _):
        try:
            log.info("wake detected (system/screens)")
            self._cb()
        except Exception as e:
            log.exception("wake callback error: %s", e)


def _start_wake_observer(cb):
    """
    Listen for system wake + screens wake. This corresponds to opening the Mac after sleep.
    """
    def _runner():
        ws = NSWorkspace.sharedWorkspace()
        nc = ws.notificationCenter()
        obs = _WakeObserver.alloc().initWithCallback_(cb)

        for name in (NSWorkspaceDidWakeNotification, NSWorkspaceScreensDidWakeNotification):
            nc.addObserver_selector_name_object_(obs, "receive:", name, None)

        log.info("native wake callbacks installed (DidWake + ScreensDidWake)")
        rl = NSRunLoop.currentRunLoop()
        while True:
            rl.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(60.0))

    threading.Thread(target=_runner, daemon=True).start()


# ---------- Singleton ----------
def _acquire_singleton_lock() -> tuple[bool, object | None]:
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    f = open(LOCK_PATH, "w")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        f.write(str(os.getpid()))
        f.flush()
        return True, f
    except BlockingIOError:
        return False, f


# ---------- Agent ----------
def _daemon(cfg_path: Path, cli_verbose: bool):
    cfg = _load_cfg(cfg_path)
    _setup_logging(cli_verbose or cfg.verbose)

    ok, lockfile = _acquire_singleton_lock()
    if not ok:
        log.error(
            "another whereishorse instance is already running (lock: %s); exiting.",
            LOCK_PATH,
        )
        sys.exit(1)

    tz = ZoneInfo(cfg.schedule.tz)
    H = cfg.schedule.cutoff_hour

    log.info(
        "service start | me=%s names=%s daily %02d:00 %s",
        cfg.me,
        cfg.names,
        H,
        cfg.schedule.tz,
    )

    state = {
        "slot": None,
        "current_img": None,
        "last_holder": None,
    }
    lock = threading.Lock()

    def compute(now_utc: datetime):
        s = _slot_daily(now_utc, tz, H)
        holder = _holder_for(s, cfg.names, cfg.seed)
        mine = holder.lower() == cfg.me.lower()
        img = cfg.horse_wallpaper if mine else cfg.empty_wallpaper
        return s, holder, mine, img

    def apply(reason: str, force: bool = False):
        with lock:
            now = datetime.now(timezone.utc)
            s, holder, mine, img = compute(now)

            if force or s != state["slot"] or holder != state["last_holder"]:
                if mine:
                    log.info("I have the horse 🐎 (slot=%s)", s)
                else:
                    log.info("someone else has the horse: %s (slot=%s)", holder, s)

            if force or img != state["current_img"]:
                log.info("apply(%s): %s (force=%s)", reason, img, force)
                if _set_wallpaper(img):
                    state["current_img"] = img

            state["slot"] = s
            state["last_holder"] = holder

    def on_wake():
        # ONLY re-apply on wake from sleep
        apply("wake", force=True)

    # Only this observer (no screen lock / space observers)
    _start_wake_observer(on_wake)

    # Initial apply ONLY at app start
    apply("start", force=True)

    # Keep Cocoa runloop alive for notifications
    try:
        rl = NSRunLoop.currentRunLoop()
        while True:
            rl.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(60.0))
    except KeyboardInterrupt:
        pass
    finally:
        if lockfile:
            try:
                fcntl.flock(lockfile.fileno(), fcntl.LOCK_UN)
                lockfile.close()
            except Exception:
                pass


# ---------- Setup CLI ----------
def _subst(text: str, mapping: dict[str, str]) -> str:
    for k, v in mapping.items():
        text = text.replace(f"{{{{{k}}}}}", v)
    return text


def setup_cli():
    ap = argparse.ArgumentParser(
        prog="whereishorse-setup",
        description="Copy config + images, write LaunchAgent, load it.",
    )
    ap.add_argument(
        "--silent",
        action="store_true",
        help="Write files but do not load the agent.",
    )
    ap.add_argument(
        "--uninstall",
        action="store_true",
        help="Unload & remove LaunchAgent (keeps config/images).",
    )
    args = ap.parse_args()

    if args.uninstall:
        subprocess.run(
            ["launchctl", "bootout", f"gui/{os.getuid()}", LABEL],
            check=False,
        )
        try:
            LAUNCHAGENT_PATH.unlink()
            print(f"Removed {LAUNCHAGENT_PATH}")
        except FileNotFoundError:
            print("LaunchAgent not found.")
        return

    CFG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Config
    if not CFG_PATH.exists():
        if not TPL_CONFIG.exists():
            print(f"Template missing: {TPL_CONFIG}", file=sys.stderr)
            sys.exit(1)
        shutil.copyfile(TPL_CONFIG, CFG_PATH)
        CFG_PATH.write_text(
            CFG_PATH.read_text().replace("__ME__", getpass.getuser())
        )
        print(f"Created {CFG_PATH}")
    else:
        print(f"Config exists: {CFG_PATH}")

    # Wallpapers
    for src, dst in [
        (TPL_HORSE, CFG_DIR / "horse.png"),
        (TPL_EMPTY, CFG_DIR / "empty.png"),
    ]:
        if not src.exists():
            print(
                f"Wallpaper template missing: {src}",
                file=sys.stderr,
            )
            sys.exit(1)
        if not dst.exists():
            shutil.copyfile(src, dst)
            print(f"Copied {dst}")
        else:
            print(f"Wallpaper exists: {dst}")

    # Plist
    exe = shutil.which("whereishorse") or ""
    if not exe:
        print("Could not resolve 'whereishorse' executable (pipx install first).", file=sys.stderr)
        sys.exit(1)
    if not TPL_PLIST.exists():
        print(f"Plist template missing: {TPL_PLIST}", file=sys.stderr)
        sys.exit(1)

    mapping = {
        "EXECUTABLE": exe,
        "CONFIG_PATH": str(CFG_PATH),
        "LOG_DIR": str(LOG_DIR),
        "VERBOSE_FLAG": "-v",
    }
    LAUNCHAGENT_PATH.write_text(_subst(TPL_PLIST.read_text(), mapping))
    print(f"Wrote LaunchAgent: {LAUNCHAGENT_PATH}")

    if not args.silent:
        uid = os.getuid()
        subprocess.run(["launchctl", "bootout", f"gui/{uid}", LABEL], check=False)
        r = subprocess.run(
            ["launchctl", "bootstrap", f"gui/{uid}", str(LAUNCHAGENT_PATH)],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            subprocess.run(
                ["launchctl", "kickstart", "-k", f"gui/{uid}/{LABEL}"],
                check=False,
            )
            print("Loaded (bootstrap) and started whereishorse.")
            print("It will run at login and after waking from sleep.")
            print(f"Logs: {LOG_DIR}/out.log  {LOG_DIR}/err.log")
        else:
            print(
                f"[whereishorse] bootstrap failed (rc={r.returncode}): "
                f"{(r.stderr or r.stdout).strip()}"
            )
            print("Try manually:")
            print(f"  launchctl bootout gui/{uid} {LABEL} || true")
            print(f"  launchctl bootstrap gui/{uid} {LAUNCHAGENT_PATH}")
            print(f"  launchctl kickstart -k gui/{uid}/{LABEL}")


# ---------- Control CLI ----------
def _is_loaded() -> bool:
    r = subprocess.run(
        ["launchctl", "print", f"gui/{os.getuid()}/{LABEL}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return r.returncode == 0


def control_cli():
    p = argparse.ArgumentParser(
        prog="whereishorsectl",
        description="Control the whereishorse LaunchAgent",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("start")
    sub.add_parser("stop")
    sub.add_parser("restart")
    lp = sub.add_parser("logs")
    lp.add_argument("-f", "--follow", action="store_true")
    args = p.parse_args()

    uid = os.getuid()
    domain = f"gui/{uid}"
    service = f"{domain}/{LABEL}"  # e.g. gui/501/com.whereishorse.agent

    if args.cmd == "status":
        loaded = _is_loaded()
        print(f"loaded: {loaded}")
        pr = subprocess.run(
            ["pgrep", "-fl", "whereishorse"],
            capture_output=True,
            text=True,
        )
        lines = [
            ln
            for ln in pr.stdout.strip().splitlines()
            if ln
            and "whereishorsectl" not in ln
            and "whereishorse-setup" not in ln
            and "grep" not in ln
        ]
        print(f"processes: {len(lines)}")
        for ln in lines:
            print("  " + ln)
        sys.exit(0)

    if args.cmd == "start":
        subprocess.run(["launchctl", "enable", service], check=False)
        if not _is_loaded():
            subprocess.run(
                ["launchctl", "bootstrap", domain, str(LAUNCHAGENT_PATH)],
                check=False,
            )
        subprocess.run(
            ["launchctl", "kickstart", "-k", service],
            check=False,
        )
        sys.exit(0)

    if args.cmd == "stop":
        subprocess.run(
            ["launchctl", "bootout", service],
            check=False,
        )
        sys.exit(0)

    if args.cmd == "restart":
        subprocess.run(["launchctl", "bootout", service], check=False)
        subprocess.run(["launchctl", "enable", service], check=False)
        subprocess.run(["launchctl", "bootstrap", domain, str(LAUNCHAGENT_PATH)], check=False)
        subprocess.run(["launchctl", "kickstart", "-k", service], check=False)
        sys.exit(0)

    if args.cmd == "logs":
        outp = LOG_DIR / "out.log"
        errp = LOG_DIR / "err.log"
        print(f"stdout: {outp}\nstderr: {errp}")
        if args.follow:
            os.execvp("tail", ["tail", "-f", str(outp), str(errp)])
        sys.exit(0)


# ---------- Simple utilities / entrypoints ----------
def _compute_holder(cfg: Cfg) -> tuple[int, str, bool, str]:
    tz = ZoneInfo(cfg.schedule.tz)
    s = _slot_daily(datetime.now(timezone.utc), tz, cfg.schedule.cutoff_hour)
    holder = _holder_for(s, cfg.names, cfg.seed)
    mine = holder.lower() == cfg.me.lower()
    img = cfg.horse_wallpaper if mine else cfg.empty_wallpaper
    return s, holder, mine, img


def whereishe_main():
    """
    Console entrypoint: prints who has the horse today.
    """
    cfg = _load_cfg(CFG_PATH)
    s, holder, mine, _ = _compute_holder(cfg)
    if mine:
        print(f"Today (slot {s}) YOU ({cfg.me}) have the horse 🐎")
    else:
        print(f"Today (slot {s}) {holder} has the horse")


def fixhim_main():
    """
    Console entrypoint: forces a wallpaper refresh to match today's holder.
    """
    cfg = _load_cfg(CFG_PATH)
    s, holder, mine, img = _compute_holder(cfg)
    ok = _set_wallpaper(img)
    if ok:
        who = "you" if mine else holder
        print(f"Refreshed wallpaper for slot {s}; holder is {who}.")
    else:
        print("Failed to refresh wallpaper.", file=sys.stderr)
        sys.exit(1)


# ---------- Entrypoint for the agent ----------
def main():
    ap = argparse.ArgumentParser(
        prog="whereishorse",
        description="Serverless, deterministic horse hand-off (macOS).",
    )
    ap.add_argument(
        "-c",
        "--config",
        default=str(CFG_PATH),
        help="Path to YAML config.",
    )
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logs to stdout.",
    )
    args = ap.parse_args()
    _daemon(Path(os.path.expanduser(args.config)), cli_verbose=args.verbose)


if __name__ == "__main__":
    main()
