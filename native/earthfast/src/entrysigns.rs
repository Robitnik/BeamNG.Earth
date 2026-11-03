//! Town entry and exit signs (the user, 2026-11-03: white boards with the settlement's name where a road enters it, the
//! same name crossed out where it leaves, a board of its own each way). The glue is server/entrysigns.py; the picture is
//! dirsigns_face.rs (dirsign_town_face), the mesh dirsigns.rs (dirsign_mesh).
//!
//! Where (town_sign_place): the server samples the built-up land cover along every road worth a sign every STEP metres
//! (one byte a sample, 1 built-up) and hands over those states. Every direction of travel a road allows is walked as a
//! chain (the road, with the best way on at each end for the samples beyond it: a settlement's limit is often a
//! junction). A smoothed state that turns from open land into built-up land, with RUN_OUT of open land behind it and
//! RUN_IN of built-up land ahead, is an entry; the other way round an exit. A sign stands at the turn on the driving
//! side, clear of every pavement and off bridges and tunnels, facing the traffic it is for.
//!
//! What it says: the settlement whose centre is nearest, scaled by its size, to a point INWARD metres inside the turn
//! (the sign's own name for an exit too: the place just left). No settlement near: no sign (no invented names).
//!
//! Everything is a function of the data (the chain, the order of the roads, hashed ties): the same signs from every tile.

use std::collections::HashMap;

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::dirsigns::clear;

const M_PER_DEG: f64 = 111_320.0;
const CHAIN_COS: f64 = 0.5;         // the way on within ~60 degrees of the way we came
const CHAIN_HOPS: usize = 6;
const TANGENT_M: f64 = 15.0;
const ALONG_M: [f64; 7] = [0.0, 10.0, -10.0, 20.0, -20.0, 30.0, -30.0];
const OUT_STEP_M: f64 = 0.5;
const OUT_MAX_M: f64 = 3.0;
const CLEAR_M: f64 = 0.3;
const POST_R: f64 = 0.35;
const MIN_PIECE_M: f64 = 40.0;      // a road shorter than this gets no sign of its own
const END_M: f64 = 4.0;             // a sign this far from a road's ends at least
const SMOOTH: usize = 5;            // majority over this many samples

struct Road {
    a: usize,
    b: usize,
    s: Vec<f64>,        // arc length at each vertex
    hw: f64,
    dir: i64,
    urban: bool,
    start: i64,         // the junction ids of its two ends
    end: i64,
    st: (usize, usize), // its samples in the state array
}

struct Net<'a> {
    pts: numpy::ndarray::ArrayView2<'a, f64>,
    ll: numpy::ndarray::ArrayView2<'a, f64>,
    roads: Vec<Road>,
}

impl<'a> Net<'a> {
    fn total(&self, r: usize) -> f64 { let rd = &self.roads[r]; rd.s[rd.b - rd.a - 1] }

    /// (x, y) and (lon, lat) of road r at arc length s.
    fn at(&self, r: usize, s: f64) -> ([f64; 2], [f64; 2]) {
        let rd = &self.roads[r];
        let n = rd.b - rd.a;
        let s = s.clamp(0.0, rd.s[n - 1]);
        let mut k = match rd.s.binary_search_by(|v| v.partial_cmp(&s).unwrap()) { Ok(k) => k, Err(k) => k.max(1) - 1 };
        k = k.min(n - 2);
        let l = rd.s[k + 1] - rd.s[k];
        let t = if l > 1e-9 { (s - rd.s[k]) / l } else { 0.0 };
        let (i, j) = (rd.a + k, rd.a + k + 1);
        let lerp = |m: &numpy::ndarray::ArrayView2<f64>, c: usize| m[[i, c]] + t * (m[[j, c]] - m[[i, c]]);
        ([lerp(&self.pts, 0), lerp(&self.pts, 1)], [lerp(&self.ll, 0), lerp(&self.ll, 1)])
    }

    /// Arc length at distance u along the direction of travel.
    fn s_at(&self, e: Edge, u: f64) -> f64 { if e.fwd { u } else { self.total(e.road) - u } }

    /// The unit direction of travel between distances u0 and u1 along the edge.
    fn chord(&self, e: Edge, u0: f64, u1: f64) -> [f64; 2] {
        let (p, _) = self.at(e.road, self.s_at(e, u0));
        let (q, _) = self.at(e.road, self.s_at(e, u1));
        unit([q[0] - p[0], q[1] - p[1]])
    }

    fn allowed(&self, e: Edge) -> bool { let d = self.roads[e.road].dir; d == 0 || (d > 0) == e.fwd }
    fn from_node(&self, e: Edge) -> i64 { let r = &self.roads[e.road]; if e.fwd { r.start } else { r.end } }
    fn to_node(&self, e: Edge) -> i64 { let r = &self.roads[e.road]; if e.fwd { r.end } else { r.start } }
    fn samples(&self, e: Edge) -> usize { let r = &self.roads[e.road]; r.st.1 - r.st.0 }
}

/// A road driven in one direction: forward along its geometry, or against it.
#[derive(Clone, Copy, PartialEq)]
struct Edge {
    road: usize,
    fwd: bool,
}

fn unit(v: [f64; 2]) -> [f64; 2] {
    let l = (v[0] * v[0] + v[1] * v[1]).sqrt().max(1e-9);
    [v[0] / l, v[1] / l]
}

fn dot(a: [f64; 2], b: [f64; 2]) -> f64 { a[0] * b[0] + a[1] * b[1] }

/// The settlement's own size in m (a city is a wide place, a hamlet a few houses): a radius from the population.
fn radius_m(pop: f64) -> f64 { (1000.0 * (pop / 9400.0).sqrt()).max(450.0) }

/// A sign worked out.
struct Sign {
    q: [f64; 2],
    t: [f64; 2],
    road_xy: [f64; 2],
    exit: bool,
    place: usize,
    urban: bool,
    inside: bool,
}

/// The entry and exit signs of a box (x0, y0, x1, y1).
///
/// Roads: vertices pts (n, 2) and their lon/lat ll, offsets off (r + 1); meta (r, 5) [half width, direction (+1 / -1
/// one way along / against the geometry, 0 both), rank, urban, link]; junctions (connector ids con_id at geometry
/// fractions con_at, offsets con_off); states (one byte a sample, 1 built-up) at s = min(k * step, length), offsets
/// st_off (r + 1) (a road without samples takes no part); bridge / tunnel ranges (skip, skip_off); pavements poly,
/// poly_off; settlements (p, 4) [lon, lat, population, class]. params: [step m, built-up run m, open run m, inward
/// m, lateral urban m, lateral rural m, panel width m, duplicate distance m, name slack (share of the radius), name
/// extra m, core share (no sign inside this share of a big settlement's radius from its centre), the population
/// from which a settlement is big].
///
/// Returns signs (k, 9) [x, y, travel x, travel y, road x, road y, 0 entry / 1 exit, settlement index, urban].
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn town_sign_place<'py>(py: Python<'py>, pts: PyReadonlyArray2<'py, f64>, ll: PyReadonlyArray2<'py, f64>,
                            off: PyReadonlyArray1<'py, i64>, meta: PyReadonlyArray2<'py, f64>,
                            con_at: PyReadonlyArray1<'py, f64>, con_id: PyReadonlyArray1<'py, i64>,
                            con_off: PyReadonlyArray1<'py, i64>, states: PyReadonlyArray1<'py, u8>,
                            st_off: PyReadonlyArray1<'py, i64>, skip: PyReadonlyArray2<'py, f64>,
                            skip_off: PyReadonlyArray1<'py, i64>, poly: PyReadonlyArray2<'py, f64>,
                            poly_off: PyReadonlyArray1<'py, i64>, places: PyReadonlyArray2<'py, f64>, bx: [f64; 4],
                            left_hand: bool, params: [f64; 12])
                            -> Bound<'py, PyArray2<f64>> {
    let (pts, ll, off, meta) = (pts.as_array(), ll.as_array(), off.as_array(), meta.as_array());
    let (con_at, con_id, con_off) = (con_at.as_array(), con_id.as_array(), con_off.as_array());
    let (states, st_off, skip, skip_off) = (states.as_array(), st_off.as_array(), skip.as_array(), skip_off.as_array());
    let (poly, poly_off, places) = (poly.as_array(), poly_off.as_array(), places.as_array());
    let [step, run_in_m, run_out_m, inward, lat_urban, lat_rural, panel_w, dup_m, slack, extra_m, core_share, core_pop] = params;
    let step = step.max(1.0);
    let (need_in, need_out) = ((run_in_m / step).ceil() as usize, (run_out_m / step).ceil() as usize);
    let need = need_in.max(need_out);

    let mut net = Net { pts, ll, roads: Vec::new() };
    let nr = off.len().saturating_sub(1);
    for r in 0..nr {
        let (a, b) = (off[r] as usize, off[r + 1] as usize);
        let mut s = vec![0.0f64; b.saturating_sub(a)];
        for k in a + 1..b {
            s[k - a] = s[k - a - 1] + ((pts[[k, 0]] - pts[[k - 1, 0]]).powi(2) + (pts[[k, 1]] - pts[[k - 1, 1]]).powi(2)).sqrt();
        }
        let total = if b >= a + 2 { s[b - a - 1] } else { 0.0 };
        // its two end junctions: the connectors at (or a synthetic id for) the ends
        let mut first: Option<(f64, i64)> = None;
        let mut last: Option<(f64, i64)> = None;
        for k in con_off[r] as usize..con_off[r + 1] as usize {
            let f = con_at[k];
            if first.map_or(true, |p| f < p.0) { first = Some((f, con_id[k])); }
            if last.map_or(true, |p| f > p.0) { last = Some((f, con_id[k])); }
        }
        let start = match first { Some((f, id)) if f < 1e-3 => id, _ => -(2 * r as i64) - 1 };
        let end = match last { Some((f, id)) if f > 1.0 - 1e-3 => id, _ => -(2 * r as i64) - 2 };
        let st = (st_off[r] as usize, st_off[r + 1] as usize);
        net.roads.push(Road { a, b, s, hw: meta[[r, 0]], dir: meta[[r, 1]] as i64, urban: meta[[r, 3]] != 0.0, start, end,
                              st: if total >= MIN_PIECE_M { st } else { (st.0, st.0) } });
    }
    let net = net;
    let mut at_node: HashMap<i64, Vec<usize>> = HashMap::new();
    for (r, rd) in net.roads.iter().enumerate() {
        if rd.st.1 > rd.st.0 {
            at_node.entry(rd.start).or_default().push(r);
            at_node.entry(rd.end).or_default().push(r);
        }
    }

    // the best way on / back at a junction: the edge leaving it whose direction keeps ours
    let leaving = |node: i64, from: Edge, back_dir: [f64; 2]| -> Option<Edge> {
        let mut best: Option<(Edge, f64)> = None;
        for &r in at_node.get(&node).map(|v| v.as_slice()).unwrap_or(&[]) {
            for fwd in [true, false] {
                let e = Edge { road: r, fwd };
                if r == from.road || !net.allowed(e) || net.from_node(e) != node { continue; }
                let c = dot(net.chord(e, 0.0, TANGENT_M.min(net.total(r))), back_dir);
                if c >= CHAIN_COS && best.map_or(true, |(_, bc)| c > bc) { best = Some((e, c)); }
            }
        }
        best.map(|p| p.0)
    };
    // the edge arriving at the junction that `to` (leaving it) continues
    let arriving = |node: i64, to: Edge, dir: [f64; 2]| -> Option<Edge> {
        let mut best: Option<(Edge, f64)> = None;
        for &r in at_node.get(&node).map(|v| v.as_slice()).unwrap_or(&[]) {
            for fwd in [true, false] {
                let e = Edge { road: r, fwd };
                if r == to.road || !net.allowed(e) || net.to_node(e) != node { continue; }
                let l = net.total(r);
                let c = dot(net.chord(e, (l - TANGENT_M).max(0.0), l), dir);
                if c >= CHAIN_COS && best.map_or(true, |(_, bc)| c > bc) { best = Some((e, c)); }
            }
        }
        best.map(|p| p.0)
    };
    // an edge's samples in the order of travel
    let seq = |e: Edge| -> Vec<bool> {
        let rd = &net.roads[e.road];
        let mut v: Vec<bool> = (rd.st.0..rd.st.1).map(|k| states[k] != 0).collect();
        if !e.fwd { v.reverse(); }
        v
    };

    let boxes: Vec<[f64; 4]> = (0..poly_off.len().saturating_sub(1)).map(|q| {
        let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        for k in poly_off[q] as usize..poly_off[q + 1] as usize {
            b[0] = b[0].min(poly[[k, 0]]); b[1] = b[1].min(poly[[k, 1]]);
            b[2] = b[2].max(poly[[k, 0]]); b[3] = b[3].max(poly[[k, 1]]);
        }
        b
    }).collect();

    let mut pend: Vec<Sign> = Vec::new();
    for r in 0..nr {
        for fwd in [true, false] {
            let e = Edge { road: r, fwd };
            let n = net.samples(e);
            if n == 0 || !net.allowed(e) { continue; }
            let rd = &net.roads[r];
            let total = net.total(r);
            // the samples before and after, through the best way on at each end
            let mut tail: Vec<bool> = Vec::new();         // reversed: the nearest sample first
            let mut cur = e;
            for _ in 0..CHAIN_HOPS {
                if tail.len() >= need { break; }
                let dir = net.chord(cur, 0.0, TANGENT_M.min(net.total(cur.road)));
                let Some(p) = arriving(net.from_node(cur), cur, dir) else { break };
                let mut v = seq(p);
                v.reverse();
                tail.extend(v);
                cur = p;
            }
            tail.truncate(need);
            let mut head: Vec<bool> = Vec::new();
            let mut cur = e;
            for _ in 0..CHAIN_HOPS {
                if head.len() >= need { break; }
                let l = net.total(cur.road);
                let back = net.chord(cur, (l - TANGENT_M).max(0.0), l);
                let Some(p) = leaving(net.to_node(cur), cur, back) else { break };
                head.extend(seq(p));
                cur = p;
            }
            head.truncate(need);
            let own_start = tail.len();
            let mut ext: Vec<bool> = tail.iter().rev().copied().collect();
            ext.extend(seq(e));
            ext.extend(head.iter().copied());
            // (the median of SMOOTH samples: a gap of a park, a field, a car park is not the settlement's limit)
            let m = ext.len();
            let sm: Vec<bool> = (0..m).map(|i| {
                let (a, b) = (i.saturating_sub(SMOOTH / 2), (i + SMOOTH / 2 + 1).min(m));
                let ones = ext[a..b].iter().filter(|&&v| v).count();
                ones * 2 > b - a
            }).collect();
            let u_of = |j: usize| -> f64 {
                let k = if fwd { j } else { n - 1 - j };
                let s = (k as f64 * step).min(total);
                if fwd { s } else { total - s }
            };
            for j in 0..n {
                let i = own_start + j;
                if i == 0 || sm[i] == sm[i - 1] { continue; }
                let entry = sm[i];
                let (n_new, n_old) = if entry { (need_in, need_out) } else { (need_out, need_in) };
                if i + n_new > m || i < n_old { continue; }
                if !(i..i + n_new).all(|k| sm[k] == sm[i]) || !(i - n_old..i).all(|k| sm[k] == sm[i - 1]) { continue; }
                let u = if j == 0 { 0.0 } else { (u_of(j - 1) + u_of(j)) / 2.0 };
                // the spot: the driving side, clear of the pavements, off bridges and tunnels
                let side = if left_hand { -1.0 } else { 1.0 };
                let lateral = if rd.urban { lat_urban } else { lat_rural };
                let base = rd.hw + lateral + panel_w / 2.0;
                let (sa_, sb_) = (skip_off[r] as usize, skip_off[r + 1] as usize);
                let mut found: Option<([f64; 2], [f64; 2], [f64; 2], f64)> = None;
                'spot: for du in ALONG_M {
                    let uu = u + du;
                    if uu < END_M || uu > total - END_M { continue; }
                    let f = if total > 1e-9 { net.s_at(e, uu) / total } else { 0.0 };
                    if (sa_..sb_).any(|k| skip[[k, 0]] - 1e-6 <= f && f <= skip[[k, 1]] + 1e-6) { continue; }
                    let (road_xy, _) = net.at(r, net.s_at(e, uu));
                    let t = net.chord(e, (uu - 3.0).max(0.0), (uu + 3.0).min(total));
                    let right = [t[1] * side, -t[0] * side];
                    let mut ex = 0.0;
                    while ex <= OUT_MAX_M + 1e-9 {
                        let q = [road_xy[0] + right[0] * (base + ex), road_xy[1] + right[1] * (base + ex)];
                        let p1 = [q[0] - right[0] * (panel_w / 2.0 - 0.3), q[1] - right[1] * (panel_w / 2.0 - 0.3)];
                        let p2 = [q[0] + right[0] * (panel_w / 2.0 - 0.3), q[1] + right[1] * (panel_w / 2.0 - 0.3)];
                        if clear(p1, POST_R + CLEAR_M, &poly, &poly_off, &boxes) && clear(p2, POST_R + CLEAR_M, &poly, &poly_off, &boxes)
                            && clear(q, POST_R + CLEAR_M, &poly, &poly_off, &boxes) {
                            found = Some((q, t, road_xy, uu));
                            break 'spot;
                        }
                        ex += OUT_STEP_M;
                    }
                }
                let Some((q, t, road_xy, uu)) = found else { continue };
                // the name: the settlement nearest (by its size) to a point inside the turn
                let (_, here) = net.at(r, net.s_at(e, uu));
                let sgn = if entry { 1.0 } else { -1.0 };
                let c = here[1].to_radians().cos().max(0.05);
                let probe = [here[0] + t[0] * inward * sgn / (M_PER_DEG * c), here[1] + t[1] * inward * sgn / M_PER_DEG];
                let mut pick: Option<(usize, f64)> = None;
                for p in 0..places.nrows() {
                    let (dx, dy) = ((places[[p, 0]] - probe[0]) * M_PER_DEG * c, (places[[p, 1]] - probe[1]) * M_PER_DEG);
                    let d = (dx * dx + dy * dy).sqrt();
                    let rad = radius_m(places[[p, 2]]);
                    if d > slack * rad + extra_m { continue; }
                    let score = d / rad;
                    if pick.map_or(true, |(_, sc)| score < sc - 1e-9) { pick = Some((p, score)); }
                }
                let Some((place, _)) = pick else { continue };
                // inside a big settlement's core a gap of open land (a park, a river, a field) is no limit
                let (dx, dy) = ((places[[place, 0]] - probe[0]) * M_PER_DEG * c, (places[[place, 1]] - probe[1]) * M_PER_DEG);
                if places[[place, 2]] >= core_pop && (dx * dx + dy * dy).sqrt() < core_share * radius_m(places[[place, 2]]) { continue; }
                let inside = q[0] >= bx[0] && q[0] < bx[2] && q[1] >= bx[1] && q[1] < bx[3];
                pend.push(Sign { q, t, road_xy, exit: !entry, place, urban: rd.urban, inside });
            }
        }
    }
    // one sign for each way and settlement within dup_m (branches that meet at the limit, two carriageways' ends):
    // the first in a stable order
    pend.sort_by(|a, b| (a.exit, (a.q[0] * 100.0).round() as i64, (a.q[1] * 100.0).round() as i64)
        .cmp(&(b.exit, (b.q[0] * 100.0).round() as i64, (b.q[1] * 100.0).round() as i64)));
    let mut kept: Vec<usize> = Vec::new();
    for (i, s) in pend.iter().enumerate() {
        let dup = kept.iter().any(|&k| {
            let o = &pend[k];
            o.exit == s.exit && o.place == s.place && dot(o.t, s.t) > 0.7
                && (o.q[0] - s.q[0]).powi(2) + (o.q[1] - s.q[1]).powi(2) < dup_m * dup_m
        });
        if !dup { kept.push(i); }
    }
    let mut out: Vec<f64> = Vec::new();
    for &i in &kept {
        let s = &pend[i];
        if !s.inside { continue; }
        out.extend_from_slice(&[s.q[0], s.q[1], s.t[0], s.t[1], s.road_xy[0], s.road_xy[1],
                                if s.exit { 1.0 } else { 0.0 }, s.place as f64, if s.urban { 1.0 } else { 0.0 }]);
    }
    let k = out.len() / 9;
    numpy::ndarray::Array2::from_shape_vec((k, 9), out).unwrap().into_pyarray(py)
}
