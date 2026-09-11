from collections import Counter
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from pokemon_red_env import worker_data_directory
from stitching.stitch import stitch_images
import cv2
import glob
import numpy as np
import os
import shutil

def merge_visit_count_deltas(global_counts, deltas):
    """
    Merges one or more worker-local visit-count deltas into a global
    visit-count table.

    Each delta represents a single worker's counts incremented since its
    last sync (see PokemonRedEnv.pop_visit_count_delta). A given state is
    only ever incremented, between two syncs, by the worker(s) that
    actually observed it during that interval, so when a key appears in
    more than one delta (or already exists in global_counts) the counts
    represent independent new observations and are summed together, not
    deduplicated.

    This is pure and standalone -- it doesn't touch any vectorized-env
    machinery -- so it can be unit tested directly against plain dicts.

    Args:
        global_counts: the current global visit-count table. Not mutated.
        deltas: an iterable of per-worker delta dicts (state hash -> count
            increment observed since that worker's last sync).

    Returns:
        A new Counter: global_counts with every delta's counts added in.
        Returned as a Counter, not a plain dict -- this value gets pushed
        back onto each worker's `visit_counts` via
        VecEnv.env_method("set_visit_counts", ...), and
        PokemonRedEnv.calculate_fitness relies on Counter's zero-default
        behavior (`self.visit_counts[key] += 1`) for keys it hasn't seen
        yet; downgrading to a plain dict here would KeyError on that line
        the next time a worker visits a genuinely new state.
    """
    merged = Counter(global_counts)
    for delta in deltas:
        merged.update(delta)
    return merged


def sync_visit_counts(vec_env, global_counts):
    """
    Runs one pull/merge/push visit-count sync round across every worker in
    a vectorized environment.

    Pulls each worker's local delta via env_method("pop_visit_count_delta")
    (all workers, not just index 0). If every worker's delta is empty (no
    new states since the last sync), every local table already matches
    global_counts from the previous round, so the merge and the
    full-table env_method broadcast are skipped -- that broadcast cost
    grows with the number of distinct states found so far, and paying it
    when nothing changed is pure waste. Otherwise, merges the deltas into
    global_counts with merge_visit_count_deltas, then broadcasts a copy of
    the resulting global table to every worker via
    env_method("set_visit_counts", ...) so each worker's local table
    converges on the shared baseline.

    A distinct Counter copy is set per worker rather than sharing one
    object across the broadcast: under DummyVecEnv, all workers run in
    this same process, so pushing the same mutable object to every
    worker (and global_counts itself) would let every worker's subsequent
    local increments double-count directly into global_counts, silently
    inflating visit counts further with every sync. SubprocVecEnv doesn't
    have this problem (each worker is a separate process, so env_method's
    pickling already copies the value), but the fix must hold for both.

    The broadcast uses VecEnv.env_method("set_visit_counts", ...) rather
    than VecEnv.set_attr("visit_counts", ...): set_attr does a plain
    setattr on whatever object each VecEnv slot holds, while env_method
    resolves through Gymnasium's Wrapper.get_wrapper_attr, which reaches
    the wrapped PokemonRedEnv instance correctly even when a gym.make()
    wrapper sits in front of it. set_attr would silently shadow the
    attribute on the outer wrapper instead.

    Args:
        vec_env: a VecEnv-like object (DummyVecEnv, SubprocVecEnv, or a
            duck-typed stub for testing) exposing env_method.
        global_counts: the current global visit-count table.

    Returns:
        The updated global visit-count table (also pushed to every
        worker, unless no worker had anything new to report). Never the
        same object as any worker's local table.
    """
    deltas = vec_env.env_method("pop_visit_count_delta")
    if not any(deltas):
        return global_counts
    merged = merge_visit_count_deltas(global_counts, deltas)
    for i in range(vec_env.num_envs):
        vec_env.env_method("set_visit_counts", Counter(merged), indices=[i])
    return merged


def collect_new_screenshots(env_data_directory, image_directory, already_seen):
    """
    Finds every worker's newly-discovered-state screenshot that hasn't
    been incorporated into a master-map stitch yet.

    Each worker saves one screenshot per newly-discovered state hash into
    its own subdirectory under env_data_directory (see
    PokemonRedEnv.calculate_fitness), so -- unlike visit_counts -- the
    main process can read this directly off disk with a glob, without any
    env_method round-trip into the workers.

    The glob pattern is built via worker_data_directory (imported from
    pokemon_red_env) with worker_id="*", so this stays in sync with
    however PokemonRedEnv.__init__ actually lays out a worker's directory
    on disk -- rather than re-deriving that shape independently here.

    Deliberately does not mutate already_seen itself: a failed stitch
    must be able to retry this exact same batch (plus whatever else shows
    up) on the next sync, per the "only advance on success" rule that
    governs the whole stitching sync (mirrors merge_visit_count_deltas
    being pure and letting the caller decide when to commit the result).

    Args:
        env_data_directory: the shared root directory under which every
            worker has its own numbered subdirectory (matches
            PokemonRedEnv.env_data_directory).
        image_directory: the per-worker subdirectory name screenshots are
            saved under (matches PokemonRedEnv.image_directory).
        already_seen: the set of screenshot paths (as strings) already
            incorporated into a successful stitch. Not mutated.

    Returns:
        A list of path strings present on disk but not in already_seen.
        Order is whatever glob returns -- callers that need determinism
        should sort it themselves.
    """
    pattern = str(
        Path(worker_data_directory(env_data_directory, "*")) / image_directory / "*.png"
    )
    on_disk = glob.glob(pattern)
    return [path for path in on_disk if path not in already_seen]


def _is_readable_image(path):
    """
    True only if path both exists and cv2.imread can decode it.

    os.path.exists alone isn't enough: a truncated or corrupted PNG (for
    example, from a crash mid-write before this feature's atomic-replace
    safeguard existed, or from a corrupted backup) is present but useless.
    Treating "present" as "readable" would let a corrupt file get backed
    up over a last-known-good backup, or get fed straight into
    cv2.Stitcher, which is the exact failure resolve_current_map and
    write_master_map exist to avoid.
    """
    if path is None or not os.path.exists(path):
        return False
    return cv2.imread(str(path)) is not None


def resolve_current_map(map_path, backup_path):
    """
    Picks which on-disk master map, if any, is safe to feed back into the
    next stitch.

    Prefers map_path over backup_path -- the backup only exists to cover
    for a missing/corrupt primary, so a healthy primary always wins.
    Falls back to the backup before giving up entirely, so a transient
    I/O failure on the primary doesn't discard a perfectly good prior
    mosaic in favor of starting from scratch. Returns None only when
    neither is usable, which the caller treats as "bootstrap a fresh map
    from this interval's batch alone."

    Args:
        map_path: path to the primary master map file.
        backup_path: path to the one-shot backup of the previous map.

    Returns:
        map_path, backup_path, or None -- whichever of the first two is
        confirmed readable, in that preference order.
    """
    if _is_readable_image(map_path):
        return map_path
    if _is_readable_image(backup_path):
        return backup_path
    return None


def write_master_map(pano, map_path, backup_path, map_path_is_readable=None):
    """
    Safely persists a freshly-stitched panorama as the new master map.

    Three hazards this guards against:

    1. A crash or interruption mid-write must never leave map_path
       truncated or corrupt for the next sync (or a human) to read. Fixed
       by writing to a temp file in the same directory as map_path first,
       then using os.replace -- an atomic rename on POSIX and Windows --
       to swap it into place. A partial write only ever touches the temp
       file; map_path itself is either the old content or the complete
       new content, never something in between.
    2. Backing up the current map before overwriting it must not silently
       destroy the last known-good backup. If the current map_path exists
       but is corrupt/unreadable, copying it over backup_path would
       clobber the one thing resolve_current_map could still fall back
       on. So the backup copy is skipped whenever the current map isn't
       confirmed readable -- using the same readability check
       resolve_current_map uses -- leaving whatever backup already exists
       untouched.
    3. cv2.imwrite returns False on failure (e.g. disk full, permission
       error) instead of raising. Treating that return value as success
       would go on to os.replace a temp file that was never actually
       written, raising an uncaught FileNotFoundError. So a False return
       is treated the same as any other failed-sync case (see
       _sync_master_map's stitch-status check): map_path is left
       untouched and the function reports failure without raising.

    Args:
        pano: the stitched panorama (a numpy array, as returned by
            cv2.Stitcher / stitch_images) to write.
        map_path: path the new master map should end up at.
        backup_path: path the previous map is copied to first, if it's
            currently readable.
        map_path_is_readable: whether map_path is currently a readable
            image, if the caller already knows (e.g. because it just
            called resolve_current_map and can report whether map_path
            was the path that resolved). Passing this avoids a second
            cv2.imread decode of the same file. If None (the default),
            it's derived here via _is_readable_image, same as before.

    Returns:
        True if the panorama was written to map_path, False if
        cv2.imwrite failed (in which case map_path/backup_path are left
        exactly as they were before this call, aside from the backup
        copy in hazard 2 above, which -- like the existing map_path --
        is unaffected either way since map_path itself never changes).
    """
    if map_path_is_readable is None:
        map_path_is_readable = _is_readable_image(map_path)
    if map_path_is_readable:
        shutil.copyfile(map_path, backup_path)

    map_path = Path(map_path)
    # cv2.imwrite picks its codec from the filename's extension, so the
    # temp file must keep map_path's original suffix (e.g. ".png") rather
    # than a generic ".tmp" -- otherwise cv2 can't determine how to
    # encode it at all.
    tmp_path = map_path.with_name(f".{map_path.stem}.tmp{map_path.suffix}")
    if not cv2.imwrite(str(tmp_path), pano):
        return False
    os.replace(tmp_path, map_path)
    return True


def calculate_values(info):
    sum_dict = {}
    count_dict = {}
    distrib_dict = {}
    scalar_list = []

    for dict in info:
        del dict["TimeLimit.truncated"]
        scalar_dict = {}
        for k, v in dict.items():
            if isinstance(v, (int, float)): 
                sum_dict[k] = sum_dict.get(k, 0) + v
                count_dict[k] = count_dict.get(k, 0) + 1
                distrib_dict.setdefault(k, []).append(v)
                scalar_dict[dict["steps"]] = dict[k] if k != "steps" else None
        if scalar_dict:
            scalar_list.append(scalar_dict)

    mean_dict = {}
    for k in sum_dict:
        mean_dict[k] = sum_dict[k] / count_dict[k]
        distrib_dict[k] = np.array(distrib_dict[k])

    return scalar_list, mean_dict, distrib_dict


class TensorBoardCallback(BaseCallback):
    def __init__(
        self,
        log_dir,
        verbose: int = 0,
        sync_interval: int = 1,
        stitch_sync_interval: int = 10,
        master_map_path=None,
        master_map_backup_path=None,
        env_data_directory=None,
        image_directory=None,
    ):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.writer = None
        # How many rollouts to wait between cross-worker visit-count syncs.
        # Named and tunable independently of PPO's n_steps -- see
        # sync_visit_counts / _on_rollout_end.
        self.sync_interval = sync_interval
        self._rollouts_since_sync = 0
        self.global_visit_counts = {}

        # How many rollouts to wait between master-map stitching syncs.
        # Kept independent of sync_interval -- cv2.Stitcher costs
        # meaningfully more per call than the visit-count merge, so it
        # runs on its own, sparser cadence (see _sync_master_map).
        self.stitch_sync_interval = stitch_sync_interval
        self._rollouts_since_map_sync = 0
        self.master_map_path = master_map_path
        self.master_map_backup_path = master_map_backup_path
        self.env_data_directory = env_data_directory
        self.image_directory = image_directory
        # Screenshot paths already incorporated into a successful stitch.
        # Only advanced on success (see _sync_master_map / R4) -- a failed
        # stitch's batch must roll forward into the next sync's attempt.
        self._already_stitched_screenshots = set()

    def _on_training_start(self) -> None:
        if self.verbose >= 1:
            print(f"Logging with TensorBoard to {self.log_dir}")
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=Path(self.log_dir))

    def _on_step(self) -> bool:
        truncated = self.training_env.env_method("pre_truncated_check", indices=[0])[0]
        if self.verbose > 1:
            print(f"Truncated check: {truncated}")

        if truncated:
            info = self.training_env.get_attr("info")
            if self.verbose > 1:
                print(f"Info: {info}")
            final_info = [stat[-1] for stat in info]
            _, mean, distributions = calculate_values(final_info)
            # for scalar in scalars:
            #     for key, val in scalar.items():
            #         self.writer.add_scalar(f"env_stats/{key}", val, self.n_calls)

            if self.verbose > 1:
                print("Recording environment mean stats:", mean)
            for key, val in mean.items():
                self.logger.record(f"env_stats/{key}", val)

            if self.verbose > 1:
                print("Recording environment distribution stats:", distributions)
            for key, distrib in distributions.items():
                self.writer.add_histogram(f"env_stats_distribs/{key}", distrib, self.n_calls)
                self.logger.record(f"env_stats_max/{key}", max(distrib))
        return True

    def _on_training_end(self) -> None:
        if self.verbose > 1:
            print(f"Ending training with TensorBoard to {self.log_dir}")
        if self.writer:
            self.writer.close()

    def _on_rollout_start(self) -> None:
        pass

    def _on_rollout_end(self) -> None:
        self._rollouts_since_sync += 1
        if self._rollouts_since_sync >= self.sync_interval:
            self._rollouts_since_sync = 0
            distinct_before = len(self.global_visit_counts)
            self.global_visit_counts = sync_visit_counts(
                self.training_env, self.global_visit_counts
            )
            distinct_after = len(self.global_visit_counts)
            self.logger.record("env_stats/distinct_states_total", distinct_after)
            self.logger.record(
                "env_stats/distinct_states_new", distinct_after - distinct_before
            )

        self._rollouts_since_map_sync += 1
        if self._rollouts_since_map_sync >= self.stitch_sync_interval:
            self._rollouts_since_map_sync = 0
            self._sync_master_map()

    def _sync_master_map(self) -> None:
        """
        Runs one attempt at stitching every worker's newly-discovered-state
        screenshots into the persistent master map, on its own cadence
        (stitch_sync_interval), independent of visit_counts' sync.

        Mirrors sync_visit_counts' empty-delta skip: if no worker has
        anything new since the last *successful* stitch, no cv2.Stitcher
        call is made at all (R3). A failed stitch, or a stitch that
        succeeds but fails to write (e.g. disk full -- see
        write_master_map), leaves the master map file, the last-logged
        TensorBoard image, and the already-stitched set all untouched, so
        the same batch (plus whatever else shows up) is retried on the
        next sync (R4, R13) -- this never raises or halts training either
        way.
        """
        new_screenshots = collect_new_screenshots(
            self.env_data_directory,
            self.image_directory,
            self._already_stitched_screenshots,
        )
        if not new_screenshots:
            return

        resolved_map = resolve_current_map(
            self.master_map_path, self.master_map_backup_path
        )
        if resolved_map is None:
            image_paths = new_screenshots
            map_path_is_readable = False
        else:
            image_paths = [resolved_map] + new_screenshots
            # resolve_current_map prefers map_path over backup_path, so
            # if map_path is the one that resolved, it's already
            # confirmed readable -- pass that along instead of making
            # write_master_map decode the same file again to find out.
            map_path_is_readable = (
                self.master_map_path is not None
                and resolved_map == self.master_map_path
            )
        status, pano = stitch_images(image_paths)
        if status != cv2.Stitcher_OK:
            return

        wrote_map = write_master_map(
            pano,
            self.master_map_path,
            self.master_map_backup_path,
            map_path_is_readable,
        )
        if not wrote_map:
            return

        # cv2 reads/writes BGR; TensorBoard's image logging expects RGB.
        pano_rgb = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)
        self.logger.record("env_stats/master_map", Image(pano_rgb, "HWC"))
        self._already_stitched_screenshots.update(new_screenshots)
