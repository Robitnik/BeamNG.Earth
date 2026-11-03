"""Town entry and exit signs (the user, 2026-11-03: white boards with a settlement's name where a road enters it, the same
name crossed out where it leaves, a board of its own each way; opt-in with the direction signs: BNE_DIRSIGNS=on or the
settings page's Direction signs).

The glue of the native kernels earthfast.town_sign_place (where the limit of a settlement is on a road, which side the
sign stands, which settlement it names), dirsign_town_face (the picture: PNG) and dirsign_mesh (panel and posts, shared
with the direction boards). The rules are in native/earthfast/src/entrysigns.rs; here the data they need: the roads'
built-up land cover (sampled every town.placement.step_m along each road worth a sign: the land cover the road
widths use), every settlement (server/data/settlements_<release>.json.gz, tools/build_places.py --all) and the
styles by country (server/data/dirsigns_styles.json, section "town").

Built in the sign tiles beside the direction boards (signstream.py: one mesh, one set of faces). Render-only.
"""
from functools import lru_cache
import gzip
import hashlib
import json
import logging
import os

import numpy as np

import dirsigns
import exitsigns
import fast
import mapcells
import revisit
import roads
import speedsigns
import staging

log = logging.getLogger("earthd.entrysigns")

KERNEL = 152                    # earthfast VERSION with town_sign_place / dirsign_town_face
SETTLEMENTS_FILE = os.path.join(dirsigns.DATA, "settlements_2026-09-23.1.json.gz")
SETTLEMENTS_SHA256 = "8a1dde823acb6969ff243df7d9829b953481981a763b2adc9fcc516b5338556a"
FACE_VERSION = 1
MAX_FACES = 24                  # distinct pictures a tile makes at most
MARGIN_M = 800.0                # m round the tile the roads are read (the chain beyond a road's end, the pavements)
MAX_RANK = 6                    # no town signs on motorways (rank 7)


class Incomplete(Exception):
    """The land cover a tile's settlements limits are found by was not all there yet: the tile is built again later (a
    tile built on it would show a cached cell's edge as a settlement's limit)."""


@lru_cache(maxsize=1)
def settlements():
    """Every city, town and village as dirsigns.Places (names, English names, [lon, lat, population, class])."""
    with open(SETTLEMENTS_FILE, "rb") as f:
        raw = f.read()
    if hashlib.sha256(raw).hexdigest() != SETTLEMENTS_SHA256:
        raise ValueError("settlements snapshot checksum mismatch")
    return dirsigns.Places(json.loads(gzip.decompress(raw)))


def _params(p):
    return [p["step_m"], p["run_in_m"], p["run_out_m"], p["inward_m"], p["lateral_urban_m"], p["lateral_rural_m"],
            p["panel_w_m"], p["dup_m"], p["name_slack"], p["name_extra_m"], p["core_share"], float(p["core_pop"])]


def sample_positions(line, xy, step):
    """(lon, lat) of a road's samples at s = min(k * step, length), k = 0 .. ceil(length / step) (the kernel's s: the
    arc length of the road's metric line xy)."""
    xy = np.asarray(xy, np.float64)
    line = np.asarray(line, np.float64)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])))])
    total = float(s[-1])
    at = np.minimum(np.arange(int(np.ceil(total / step)) + 1 if total > 0 else 1, dtype=np.float64) * step, total)
    return np.interp(at, s, line[:, 0]), np.interp(at, s, line[:, 1])


def _arrays(rows, box, margin, cfg, urban):
    """town_sign_place's road arrays of the speed-signs' rows near box: the roads worth a sign and their built-up
    samples. None where there are no roads."""
    x0, y0, x1, y1 = box
    step = float(cfg["step_m"])
    min_rank = int(cfg["min_rank"])
    ll, xy, off, meta = [], [], [0], []
    con_at, con_id, con_off = [], [], [0]
    skip, skip_off = [], [0]
    ids = {}
    sample_lat, sample_lon, counts = [], [], []
    for fid, (line, pts, m, cons, _limits, skips) in sorted(rows.items()):
        if not min_rank <= m[2] <= MAX_RANK or m[4]:        # (rank, link): minor streets and ramps take no sign
            continue
        if (pts[:, 0].max() < x0 - margin or pts[:, 0].min() > x1 + margin or
                pts[:, 1].max() < y0 - margin or pts[:, 1].min() > y1 + margin):
            continue
        lon, lat = sample_positions(line, pts, step)
        sample_lon.append(lon)
        sample_lat.append(lat)
        counts.append(len(lon))
        ll.append(line[:, :2])
        xy.append(pts)
        off.append(off[-1] + len(pts))
        meta.append(m)
        for f, c in cons:
            con_at.append(f)
            con_id.append(ids.setdefault(c, len(ids)))
        con_off.append(len(con_at))
        skip.extend(skips)
        skip_off.append(len(skip))
    if not xy:
        return None
    faults = revisit.thread_faults()
    flags = np.asarray(urban(np.concatenate(sample_lat), np.concatenate(sample_lon)), bool)
    if revisit.thread_faults() != faults:
        raise Incomplete("land cover not cached yet")
    st_off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    f64 = lambda a, *shape: np.ascontiguousarray(np.asarray(a, np.float64).reshape(*shape))    # noqa: E731
    i64 = lambda a: np.ascontiguousarray(np.asarray(a, np.int64))                              # noqa: E731
    return (f64(np.concatenate(xy), -1, 2), f64(np.concatenate(ll), -1, 2), i64(off), f64(meta, -1, 5),
            f64(con_at, -1), i64(con_id), i64(con_off), np.ascontiguousarray(flags.astype(np.uint8)), st_off,
            f64(skip, -1, 2), i64(skip_off))


def _face(out_dir, vfs_dir, country, face, name, en, exit_):
    """(material name, vfs texture path, width m, height m) of a sign's picture: drawn once (by its hash)."""
    flat = [*country["bg"], *country["ink"], *country["edge"], *country["bar"], face["cap_m"], face["en_share"],
            face["min_w_m"], face["max_w_m"], face["inset_m"], face["line_w_m"], face["bar_w_m"], 0.0, 0.0]
    h = hashlib.sha1(json.dumps([FACE_VERSION, flat, name, en, bool(exit_), face["px_per_m"]], ensure_ascii=False)
                     .encode("utf-8")).hexdigest()[:12]
    material = f"bne_town_{h}"
    png = os.path.join(out_dir, material + ".color.png")
    data, w_m, h_m = fast._ef.dirsign_town_face(name, en, bool(exit_), flat, float(face["px_per_m"]))
    if not os.path.exists(png):
        tmp = staging.tmp_for(png)
        with open(tmp, "wb") as f:
            f.write(bytes(data))
        staging.replace(tmp, png)
    return material, f"{vfs_dir}/{material}.color.png", w_m, h_m


def build_tile(md, frame, tx, ty, elev, context, decks, left_hand, mph, out_dir, vfs_dir, show_en=True):
    """(mesh, faces) of road tile (tx, ty)'s town signs, as dirsigns.build_tile gives them; (None, {}) where there are
    none (or no land cover to find a settlement's limit by)."""
    if context.urban is None:
        return None, {}
    fast.require(KERNEL, "town signs")
    dirsigns._set_fonts()
    st = dirsigns.styles()
    cfg = st["town"]
    pl = settlements()
    box = roads.tile_bbox(tx, ty)
    x0, y0, x1, y1 = box
    la, lo, _ = frame.enu_to_geodetic(np.array([x0 - MARGIN_M, x1 + MARGIN_M]), np.array([y0 - MARGIN_M, y1 + MARGIN_M]), 0.0)
    rows, seen = {}, set()
    for key in mapcells.cells_for_bbox(float(la.min()), float(lo.min()), float(la.max()), float(lo.max())):
        feats = context.cell(key)
        if feats is None or not len(feats):
            continue
        for fid, row in speedsigns._cell_rows(feats, frame, context.urban, context.widths, left_hand, mph).items():
            if fid not in seen:
                seen.add(fid)
                rows[fid] = row
    arrays = _arrays(rows, box, MARGIN_M, cfg["placement"], context.urban)
    if arrays is None:
        return None, {}
    cla, clo, _ = frame.enu_to_geodetic(np.array([x0, x1, x0, x1]), np.array([y0, y0, y1, y1]), 0.0)
    poly, poly_off = speedsigns._outlines(context.pavements(frame, np.c_[clo, cla], left_hand))
    got = fast._ef.town_sign_place(*arrays, poly, poly_off, pl.array, [float(v) for v in box], bool(left_hand),
                                   _params(cfg["placement"]))
    if not len(got):
        return None, {}
    keep, faces, materials, mats_of = [], {}, [dirsigns.METAL], {}
    for b in got:
        if exitsigns._on_deck(b[0:2], decks, 1.5):
            continue
        i = int(b[7])
        country = cfg["countries"].get(pl.cc[i])
        if country is None:
            continue
        # a city or a town has its English name under the local one, a village only the local (the user, 2026-11-03)
        en = pl.en[i] if (show_en and pl.array[i, 3] >= 1) else ""
        key = (pl.names[i], en, bool(b[6]), pl.cc[i])
        if key not in mats_of and len(mats_of) >= MAX_FACES:
            continue
        name, tex, w, h = _face(out_dir, vfs_dir, country, cfg["face"], pl.names[i], en, bool(b[6]))
        if key not in mats_of:
            mats_of[key] = len(materials)
            materials.append(name)
            faces[name] = tex
        keep.append((b, mats_of[key], w, h))
    if not keep:
        return None, {}
    pts = np.vstack([np.array([k[0][0:2] for k in keep]), np.array([k[0][4:6] for k in keep])])
    la, lo, _ = frame.enu_to_geodetic(pts[:, 0], pts[:, 1], 0.0)
    hz = np.asarray(elev(la, lo), np.float64).ravel()
    z = np.asarray(frame.geodetic_to_enu(la, lo, hz)[2], np.float64).ravel()
    n = len(keep)
    signs = [[b[0], b[1], z[i], z[n + i], b[2], b[3], w, h, mat, cfg["placement"]["mount_m"]]
             for i, (b, mat, w, h) in enumerate(keep)]
    p, t, uv, mi = fast._ef.dirsign_mesh(np.ascontiguousarray(signs, np.float64), [float(box[0]), float(box[1]), 0.0])
    if not len(t):
        return None, {}
    return (p, t, uv, materials, mi), faces


def merge(a, b):
    """One mesh (vertices, triangles, uv, materials, material per triangle) of two that share the metal's name, or
    whichever of them exists."""
    if a is None:
        return b
    if b is None:
        return a
    pa, ta, uva, ma, mia = a
    pb, tb, uvb, mb, mib = b
    mats = list(ma)
    remap = []
    for name in mb:
        if name not in mats:
            mats.append(name)
        remap.append(mats.index(name))
    mib = np.asarray(remap, np.int64)[np.asarray(mib, np.int64)]
    return (np.vstack([pa, pb]), np.vstack([ta, np.asarray(tb) + len(pa)]), np.vstack([uva, uvb]), mats,
            np.concatenate([np.asarray(mia, np.int64), mib]))
