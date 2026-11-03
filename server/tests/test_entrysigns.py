"""Town entry and exit signs (entrysigns.py, earthfast town_sign_place / dirsign_town_face): where they stand, what they
name, how a face is drawn."""
import os
import struct
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import dirsigns  # noqa: E402
import entrysigns  # noqa: E402
import fast  # noqa: E402

M_LAT = 111320.0
LAT0 = 49.2
LON0 = 28.3
M_LON = M_LAT * np.cos(np.radians(LAT0))
HAVE = getattr(fast._ef, "VERSION", 0) >= entrysigns.KERNEL


def row(xy, rank=5, hw=4.0, way=0, link=False, ends=None, skips=()):
    """A speed-signs road row (what entrysigns._arrays takes): xy (k, 2) metres."""
    xy = np.asarray(xy, np.float64)
    ll = np.column_stack([xy[:, 0] / M_LON + LON0, xy[:, 1] / M_LAT + LAT0])
    a, b = ends or (f"{xy[0][0]:.1f},{xy[0][1]:.1f}", f"{xy[-1][0]:.1f},{xy[-1][1]:.1f}")
    return (ll, xy, (hw, way, rank, 0.0, 1.0 if link else 0.0), [(0.0, a), (1.0, b)], [], list(skips))


def built_up(east_of=0.0, gaps=()):
    """A land cover: built-up east of `east_of` m (less the gaps (from, to))."""
    def urban(lat, lon):
        x = (np.asarray(lon) - LON0) * M_LON
        out = x > east_of
        for a, b in gaps:
            out &= ~((x > a) & (x < b))
        return out
    return urban


def settlements(rows):
    """[lon, lat, population, class] of places given as (east m, north m, population, class)."""
    return np.ascontiguousarray([[x / M_LON + LON0, y / M_LAT + LAT0, pop, cls] for x, y, pop, cls in rows], np.float64)


def signs(roads, places, urban=None, box=(-5000, -5000, 5000, 5000), left_hand=False):
    cfg = dirsigns.styles()["town"]["placement"]
    arrays = entrysigns._arrays(roads, box, entrysigns.MARGIN_M, cfg, urban or built_up(10.0))
    if arrays is None:
        return np.zeros((0, 9))
    return fast._ef.town_sign_place(*arrays, np.zeros((0, 2)), np.zeros(1, np.int64), places,
                                    [float(v) for v in box], left_hand, entrysigns._params(cfg))


def road():
    """A primary road across the origin, east-west; the settlement is east of the origin."""
    return {"a": row([(-3000, 0), (3000, 0)])}


VILLAGE = (500, 40, 800, 0)


@unittest.skipUnless(HAVE, "earthfast with the town-sign kernels")
class PlaceTest(unittest.TestCase):
    def test_an_entry_where_the_built_up_land_starts_and_an_exit_for_the_other_way(self):
        got = signs(road(), settlements([VILLAGE]))
        entry = [s for s in got if s[6] == 0]
        exit_ = [s for s in got if s[6] == 1]
        self.assertEqual((len(entry), len(exit_)), (1, 1))
        e, x = entry[0], exit_[0]
        self.assertGreater(e[2], 0.99, "the entry faces the traffic going east, into the settlement")
        self.assertLess(x[2], -0.99, "the exit is for the traffic going west, out of it")
        self.assertTrue(-45 < e[0] < 45 and -45 < x[0] < 45, f"at the limit: {e[0]}, {x[0]}")
        self.assertTrue(-9 < e[1] < -4, f"the right of an eastbound road (south), off its pavement: {e[1]}")
        self.assertTrue(4 < x[1] < 9, f"the right of a westbound road (north): {x[1]}")
        self.assertEqual((int(e[7]), int(x[7])), (0, 0), "both name the settlement")

    def test_left_hand_traffic_stands_on_the_other_side(self):
        got = signs(road(), settlements([VILLAGE]), left_hand=True)
        e = [s for s in got if s[6] == 0][0]
        self.assertGreater(e[1], 4, "left of an eastbound road: north")

    def test_a_one_way_road_has_only_its_own_way(self):
        got = signs({"a": row([(-3000, 0), (3000, 0)], way=1)}, settlements([VILLAGE]))
        self.assertEqual([int(s[6]) for s in got], [0])
        got = signs({"a": row([(-3000, 0), (3000, 0)], way=-1)}, settlements([VILLAGE]))
        self.assertEqual([int(s[6]) for s in got], [1], "against the geometry: the traffic goes west, out")

    def test_a_gap_in_the_built_up_land_is_not_a_limit(self):
        got = signs(road(), settlements([VILLAGE]), built_up(10.0, gaps=[(1000, 1060)]))
        self.assertEqual(sorted(int(s[6]) for s in got), [0, 1], "one entry, one exit: a park is not a village's end")

    def test_a_long_gap_is_a_limit(self):
        got = signs(road(), settlements([VILLAGE, (1900, 30, 800, 0)]), built_up(10.0, gaps=[(1000, 1600)]))
        self.assertEqual(sorted(int(s[6]) for s in got), [0, 0, 0, 1, 1, 1], "the village's two limits and the next one's")

    def test_the_limit_at_a_junction_is_found_through_the_next_road(self):
        two = {"w": row([(-3000, 0), (0, 0)], ends=("w", "j")), "e": row([(0, 0), (3000, 0)], ends=("j", "e"))}
        got = signs(two, settlements([VILLAGE]), built_up(-30.0))     # (the run of built-up land is mostly the next road's)
        self.assertEqual(sorted(int(s[6]) for s in got), [0, 1], "one each, not none and not two")

    def test_no_name_no_sign(self):
        self.assertEqual(len(signs(road(), settlements([(15000, 0, 800, 0)]))), 0)
        self.assertEqual(len(signs(road(), settlements([]).reshape(0, 4))), 0)

    def test_the_settlement_is_the_nearest_by_its_size(self):
        # a hamlet 600 m beside the limit, a town 1.5 km along: the town's radius is wider
        got = signs(road(), settlements([(300, 560, 30, 0), (900, 0, 9000, 1)]))
        self.assertEqual({int(s[7]) for s in got}, {1})

    def test_no_sign_in_the_core_of_a_big_city_where_a_park_is_no_limit(self):
        # a city of 600 000 whose centre is 500 m from the gap: its core is several km wide
        self.assertEqual(len(signs(road(), settlements([(500, 0, 600000, 2)]))), 0)
        # its edge, 7 km from its centre: outside the core, the limit is a sign's
        self.assertEqual(sorted(int(s[6]) for s in signs(road(), settlements([(7000, 0, 600000, 2)]))), [0, 1])

    def test_only_the_roads_worth_a_sign(self):
        self.assertEqual(len(signs({"a": row([(-3000, 0), (3000, 0)], rank=1)}, settlements([VILLAGE]))), 0)
        self.assertEqual(len(signs({"a": row([(-3000, 0), (3000, 0)], rank=7)}, settlements([VILLAGE]))), 0)
        self.assertEqual(len(signs({"a": row([(-3000, 0), (3000, 0)], link=True)}, settlements([VILLAGE]))), 0)
        self.assertEqual(len(signs({"a": row([(-3000, 0), (3000, 0)], rank=2)}, settlements([VILLAGE]))), 2)

    def test_no_sign_on_a_bridge(self):
        self.assertEqual(len(signs({"a": row([(-3000, 0), (3000, 0)], skips=[(0.0, 1.0)])}, settlements([VILLAGE]))), 0)

    def test_the_same_signs_in_every_tile_of_any_box(self):
        places = settlements([VILLAGE])
        whole = signs(road(), places)
        left = signs(road(), places, box=(-5000, -5000, 20, 5000))
        right = signs(road(), places, box=(20, -5000, 5000, 5000))
        key = lambda a: sorted(map(tuple, np.round(a, 3)))      # noqa: E731
        self.assertEqual(key(np.vstack([left, right])), key(whole))

    def test_nothing_without_land_cover_roads(self):
        self.assertEqual(len(signs({}, settlements([VILLAGE]))), 0)


class StyleTest(unittest.TestCase):
    def test_every_country_has_a_complete_style_and_a_known_board_style(self):
        town = dirsigns.styles()["town"]
        self.assertEqual(set(town["countries"]), set(dirsigns.styles()["countries"]))
        for cc, c in town["countries"].items():
            for k in ("bg", "ink", "edge", "bar"):
                self.assertEqual(len(c[k]), 3, (cc, k))
        for k in ("px_per_m", "cap_m", "en_share", "min_w_m", "max_w_m", "inset_m", "line_w_m", "bar_w_m"):
            self.assertIn(k, town["face"])
        for k in ("core_share", "core_pop", "step_m", "run_in_m", "run_out_m", "inward_m", "lateral_urban_m", "lateral_rural_m", "panel_w_m",
                  "mount_m", "dup_m", "name_slack", "name_extra_m", "min_rank"):
            self.assertIn(k, town["placement"])

    def test_the_settlement_snapshot_is_the_one_that_was_checked(self):
        pl = entrysigns.settlements()
        self.assertGreater(len(pl.names), 100000)
        self.assertEqual(set(pl.cc), {"UA", "CZ", "PL", "DE"})
        self.assertEqual(pl.array.shape, (len(pl.names), 4))
        self.assertTrue((pl.array[:, 3] >= 0).all() and (pl.array[:, 3] <= 2).all())

    def test_the_sample_positions_are_the_kernels(self):
        xy = np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 30.0]])
        ll = np.column_stack([xy[:, 0] / M_LON + LON0, xy[:, 1] / M_LAT + LAT0])
        lon, lat = entrysigns.sample_positions(ll, xy, 20.0)
        self.assertEqual(len(lon), 5, "80 m at 20 m: 0, 20, 40, 60, 80 -> the last at the length")
        self.assertAlmostEqual((lon[-1] - LON0) * M_LON, 50.0, delta=0.01)
        self.assertAlmostEqual((lat[-1] - LAT0) * M_LAT, 30.0, delta=0.01)


@unittest.skipUnless(HAVE, "earthfast with the town-sign kernels")
class FaceTest(unittest.TestCase):
    def setUp(self):
        dirsigns._set_fonts()
        self.town = dirsigns.styles()["town"]

    def face(self, cc, name, en, exit_):
        c, f = self.town["countries"][cc], self.town["face"]
        flat = [*c["bg"], *c["ink"], *c["edge"], *c["bar"], f["cap_m"], f["en_share"], f["min_w_m"], f["max_w_m"],
                f["inset_m"], f["line_w_m"], f["bar_w_m"], 0.0, 0.0]
        return fast._ef.dirsign_town_face(name, en, exit_, flat, float(f["px_per_m"]))

    def size(self, png):
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        return struct.unpack(">II", png[16:24])

    def test_a_face_is_a_power_of_two_png_of_the_signs_size(self):
        png, w, h = self.face("UA", "Вінниця", "Vinnytsia", False)
        pw, ph = self.size(bytes(png))
        for v in (pw, ph):
            self.assertEqual(v & (v - 1), 0, "the engine's texture cooker takes only powers of two")
        self.assertTrue(1.2 <= w <= 2.6 and 0.3 < h < 1.0, (w, h))

    def test_the_exit_is_the_same_name_crossed_out(self):
        entry = bytes(self.face("CZ", "Hradec Králové", "", False)[0])
        exit_ = bytes(self.face("CZ", "Hradec Králové", "", True)[0])
        self.assertNotEqual(entry, exit_)
        self.assertEqual(self.size(entry), self.size(exit_))

    def test_the_english_line_makes_it_taller_and_a_long_name_stays_within_the_widest(self):
        one = self.face("PL", "Kłodzko", "", False)
        two = self.face("PL", "Kłodzko", "Glatz", False)
        self.assertGreater(two[2], one[2])
        long = self.face("UA", "Новоселівка-Придніпровська-Заводська", "", False)
        self.assertLessEqual(long[1], self.town["face"]["max_w_m"] + 0.01)

    def test_the_germans_are_yellow_the_others_white(self):
        de = bytes(self.face("DE", "Kassel", "", False)[0])
        ua = bytes(self.face("UA", "Kassel", "", False)[0])
        self.assertNotEqual(de, ua)
        self.assertEqual(self.town["countries"]["DE"]["bg"], [255, 205, 0])
        self.assertEqual(self.town["countries"]["UA"]["bg"], [255, 255, 255])

    def test_the_same_content_is_the_same_bytes(self):
        a = bytes(self.face("PL", "Opole", "", True)[0])
        b = bytes(self.face("PL", "Opole", "", True)[0])
        self.assertEqual(a, b)

    def test_merging_two_meshes_remaps_the_materials(self):
        p = np.zeros((3, 3))
        t = np.array([[0, 1, 2]])
        uv = np.zeros((3, 2))
        a = (p, t, uv, ["m", "x"], np.array([1]))
        b = (p, t, uv, ["m", "y"], np.array([1]))
        out = entrysigns.merge(a, b)
        self.assertEqual(out[3], ["m", "x", "y"])
        self.assertEqual(list(out[4]), [1, 2])
        self.assertEqual(out[1].tolist(), [[0, 1, 2], [3, 4, 5]])
        self.assertIs(entrysigns.merge(None, b), b)
        self.assertIs(entrysigns.merge(a, None), a)


if __name__ == "__main__":
    unittest.main()
