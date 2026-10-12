"""Tests for the fine-tuning dataset generator and baseline grading.

Deterministic: synthetic packs built inline, fixed seeds. No network,
no GPS, no models, no Tinker credentials.
"""

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_finetune_data import (  # noqa: E402
    QUESTIONS,
    sample_candidates,
    split_train_eval,
    to_jsonl_line,
    validate_example,
    write_split,
)
from scripts.evaluate_baseline import grade, invented_names, parse_user_places  # noqa: E402
from src.llm.answer import fallback_answer  # noqa: E402


def _pack(n: int = 12) -> dict:
    # Synthetic grid pack: 4x3 places 0.01 deg apart (~1.1 km spacing).
    places = []
    for i in range(n):
        places.append({
            "name": f"Place {i:02d}", "type": "park",
            "lat": 12.90 + 0.01 * (i // 4), "lon": 74.89 + 0.01 * (i % 4),
            "osm_id": i, "osm_type": "node",
        })
    return {
        "schema_version": 1, "area_name": "Synthetic", "generated_at": "t",
        "bbox": {"min_lat": 12.895, "min_lon": 74.885, "max_lat": 12.925, "max_lon": 74.925},
        "places": places,
    }


class TestGenerator(unittest.TestCase):
    def test_reproducibility(self):
        pack = _pack()
        a = sample_candidates(pack, 7, 30, 5)
        b = sample_candidates(pack, 7, 30, 5)
        self.assertEqual(a, b)
        c = sample_candidates(pack, 8, 30, 5)
        self.assertNotEqual(a, c)

    def test_distances_use_runtime_geo(self):
        from src.places.geo import haversine_m

        pack = _pack()
        cands = sample_candidates(pack, 7, 5, 5)
        self.assertTrue(cands)
        cand = cands[0]
        for p in cand["places"]:
            src = next(q for q in pack["places"] if q["name"] == p["name"])
            expect = int(round(haversine_m(cand["lat"], cand["lon"], src["lat"], src["lon"])))
            self.assertEqual(p["distance_m"], expect)

    def test_answer_construction_complete(self):
        pack = _pack()
        cands = sample_candidates(pack, 7, 5, 5)
        for cand in cands:
            answer = fallback_answer(cand["places"])
            self.assertEqual(validate_example(cand["places"], answer, 5), [])

    def test_insufficient_data_refuses(self):
        tiny = _pack(3)
        cands = sample_candidates(tiny, 7, 30, 5)
        self.assertEqual(cands, [])  # never pads with fiction
        with self.assertRaises(SystemExit):
            split_train_eval(cands, 7, 10, 20)

    def test_split_no_leakage(self):
        pack = _pack(16)
        cands = sample_candidates(pack, 7, 60, 5)
        train, ev = split_train_eval(cands, 7, 40, 10)
        train_sets = {c["id_set"] for c in train}
        eval_sets = {c["id_set"] for c in ev}
        self.assertEqual(len(ev), 10)
        self.assertEqual(train_sets & eval_sets, set())

    def test_jsonl_schema(self):
        # Tinker FromConversationFileBuilder: {"messages": [{role, content}]}.
        pack = _pack()
        cands = sample_candidates(pack, 7, 3, 5)
        for cand in cands:
            line = to_jsonl_line(QUESTIONS[0], cand["places"], fallback_answer(cand["places"]))
            roles = [m["role"] for m in line["messages"]]
            self.assertEqual(roles, ["system", "user", "assistant"])
            for m in line["messages"]:
                self.assertIn("content", m)
            json.dumps(line)  # serializable single line


class TestGrading(unittest.TestCase):
    def test_parse_user_places(self):
        user = "Intro\n- Alpha Park (park, 100 m)\n- Beta, Place (cafe, 250 m)\nRules"
        places = parse_user_places(user)
        self.assertEqual(
            [(p["name"], p["type"], p["distance_m"]) for p in places],
            [("Alpha Park", "park", 100), ("Beta, Place", "cafe", 250)],
        )

    def test_invented_names_ignores_scaffolding(self):
        places = [{"name": "Alpha Park", "type": "park", "distance_m": 100}]
        self.assertEqual(
            invented_names("Nearby you can visit Alpha Park at 100 meters.", places), [])
        self.assertEqual(
            invented_names("Around here is Alpha Park and also Beta Gardens.", places),
            ["Beta Gardens"],
        )

    def test_grade_complete_answer(self):
        places = [{"name": "Alpha Park", "type": "park", "distance_m": 100}]
        g = grade("Alpha Park is 100 meters away.", places)
        self.assertTrue(g["complete"] and g["grounded"] and g["brief"])
        self.assertEqual(g["direction_words"], [])

    def test_grade_flags_directions(self):
        places = [{"name": "Alpha Park", "type": "park", "distance_m": 100}]
        g = grade("Alpha Park is 100 meters away, turn left at the gate.", places)
        self.assertIn("turn left", g["direction_words"])


if __name__ == "__main__":
    unittest.main()
