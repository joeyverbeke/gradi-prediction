#!/usr/bin/env python3
"""Offline evaluation for the semantic scenario scorer (Change 3).

No hardware required. Loads a scenarios_<lang>.yml, builds the SemanticDetector,
and scores labeled utterance sets per scenario:

  * explicit    — on-topic sentences that also contain a keyword stem
  * paraphrase  — on-topic sentences with NO stem overlap (the case semantics must
                  catch); for ai_future split into pos/neg to check both valences
  * hesitation  — hedging / meta / neutral filler that must stay quiet

It also runs the stem-safety check: every scenario's stems against a neutral
corpus, requiring zero substring false hits (mechanically enforces the
no-short-stems rule).

Acceptance targets (from the brief):
  * politics paraphrase   >= 80% above threshold
  * ai_future paraphrase  >= 90% above threshold, each valence individually >= 90%
  * hesitation             = 0 above threshold (near-misses acceptable)
  * stem safety            = 0 false hits on the neutral corpus

Usage:
  python scripts/semantic_eval.py                       # config/scenarios_en.yml
  python scripts/semantic_eval.py --config config/scenarios_ko.yml
"""

import argparse
import os
import sys

import yaml

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC)

from detector_semantic import SemanticDetector  # noqa: E402
from detector_keywords import KeywordDetector  # noqa: E402


# --- Labeled eval sets (English). Paraphrase items deliberately avoid stems. ---
EVAL_SETS = {
    "politics": {
        "explicit": [
            "i think donald trump is ruining the country",
            "the election was completely stolen",
            "congress needs to actually do something",
            "i'm voting democrat this year",
            "the supreme court decision was wrong",
            "inflation is totally out of control",
            "russia invaded ukraine and nato did nothing",
            "the republican party has completely changed",
            "immigration is a huge problem right now",
            "biden is way too old for the job",
        ],
        "paraphrase": [
            "i feel like the whole system is broken",
            "nobody in charge actually represents us",
            "things are getting more extreme on both sides",
            "i've lost all faith in how this place is run",
            "we're more split than ever as a nation",
            "the folks running things only serve themselves",
            "honestly i'm scared about the direction we're headed",
            "leadership these days is honestly a joke",
            "the people making decisions don't care about ordinary folks",
            "it feels like everything is falling apart around here",
        ],
        "hesitation": [
            "um let me think for a moment",
            "that's a really tough one",
            "i don't have much of an opinion",
            "can you repeat the question",
            "i'm not sure what you mean",
            "give me a second here",
            "hmm i don't really know",
            "what should i even talk about",
            "this is kind of awkward",
            "i've never thought about this before",
        ],
    },
    "ai_future": {
        "explicit": [
            "ai will take over everything soon",
            "artificial intelligence is going to change the world",
            "robots will do all of the work",
            "it will be used against ordinary people",
            "machines will replace us entirely",
            "in the future everything is automated",
            "technology will solve all of our problems",
            "ai is going to cure most diseases",
            "one day computers run everything",
            "eventually there will be no jobs left",
        ],
        # both valences kept separate so each can be checked at >= 90%
        "paraphrase_pos": [
            "honestly i think everything gets better from here",
            "we're about to enter a golden age of discovery",
            "our kids might finally get teachers that truly understand them",
            "so much of the tedious daily grind just handles itself now",
            "medicine could leap forward in ways we can barely imagine",
            "i feel like life is only getting easier for everyone",
        ],
        "paraphrase_neg": [
            "we're handing our whole lives over to these systems",
            "pretty soon no one can tell what is actually real",
            "i worry we completely lose touch with each other",
            "the powerful few end up controlling all of it",
            "everything we do gets watched and recorded constantly",
            "human effort just stops mattering at some point",
        ],
        "hesitation": [
            "um i'm not really sure honestly",
            "that's a hard thing to answer",
            "i don't know a lot about this stuff",
            "let me think for a moment",
            "i haven't given it much thought",
            "what kind of answer are you looking for",
            "hmm this is tricky",
            "can you give me a hint",
            "i really can't say",
            "no idea to be honest",
        ],
    },
}

# Ordinary everyday sentences — no political / AI content. Used for stem safety.
NEUTRAL_CORPUS = [
    "the weather is lovely this morning",
    "i had oatmeal and coffee for breakfast",
    "we walked the dog around the block",
    "she planted tomatoes in the back garden",
    "the bus was a few minutes late today",
    "i need to buy milk and eggs later",
    "the kids are watching cartoons downstairs",
    "he fixed the squeaky door hinge",
    "let's meet at the cafe around noon",
    "the movie last night was pretty funny",
    "my phone battery died on the train",
    "we repainted the kitchen a soft yellow",
    "the library closes early on sundays",
    "i love the smell of fresh bread",
    "they went hiking up the coastal trail",
    "the cat knocked a mug off the shelf",
    "please water the plants while i'm away",
    "we booked a cabin near the lake",
    "the bakery sells amazing cinnamon rolls",
    "i finally finished reading that novel",
    "the printer is out of paper again",
    "she practices the piano every evening",
    "our flight leaves early tomorrow morning",
    "the soup needs a little more salt",
    "he collects old postcards from europe",
    "the park was full of families picnicking",
    "i spilled tea all over my notebook",
    "we are painting the fence this weekend",
    "the puppy chewed through another shoe",
    "grandma sent us a box of cookies",
    "the river was calm and clear today",
    "i forgot my umbrella at the office",
    "they are renovating the corner grocery store",
    "the concert tickets sold out in minutes",
    "she knitted a scarf for her brother",
    "we watched the sunset from the pier",
    "the recipe calls for two cups of flour",
    "my bike tire went flat on the way home",
    "the museum has a new dinosaur exhibit",
    "he brews his own coffee every morning",
    "the garden smells wonderful after rain",
    "we played board games until midnight",
    "the ferry ride across the bay was smooth",
    "i organized the closet this afternoon",
    "the neighbors are having a barbecue tonight",
    "she sketches birds in the local park",
    "the train station has a lovely old clock",
    "we tried a new noodle place downtown",
    "the beach was quiet in the early morning",
    "he is teaching his daughter to ride a bike",
    "the orchard is full of ripe apples",
    "i mailed the birthday card yesterday",
    "the campfire crackled under the stars",
    "she rearranged all the living room furniture",
    "we spotted a deer near the walking path",
    "the coffee shop added oat milk to the menu",
    "my sister adopted a rescue kitten",
    "the market sells fresh flowers on fridays",
    "we cleaned out the garage over the weekend",
    "the lighthouse looks beautiful at dusk",
]


def pct(n, total):
    return 100.0 * n / total if total else 0.0


def main():
    parser = argparse.ArgumentParser(description="Offline semantic scorer eval")
    parser.add_argument("--config", default="config/scenarios_en.yml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config)) or {}
    sem_cfg = cfg.get("semantic") or {}
    scenarios = cfg.get("scenarios") or []

    proto_input = [
        {
            "id": s["id"],
            "exemplars": (s.get("semantic") or {}).get("exemplars", []),
            "contrast": (s.get("semantic") or {}).get("contrast", []),
        }
        for s in scenarios
        if (s.get("semantic") or {}).get("exemplars")
    ]
    thresholds = {
        s["id"]: float((s.get("semantic") or {}).get("threshold", 0.30)) for s in scenarios
    }
    stems_by_scenario = {s["id"]: [str(x) for x in s.get("stems", [])] for s in scenarios}

    print(f"Loading semantic detector ({sem_cfg.get('model_name')}) ...")
    detector = SemanticDetector(
        proto_input,
        model_name=sem_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        cache_dir=sem_cfg.get("cache_dir", "models/embed"),
    )

    overall_ok = True

    for sid in detector.scenario_ids:
        if sid not in EVAL_SETS:
            print(f"\n[skip] no eval set defined for scenario '{sid}'")
            continue
        thr = thresholds.get(sid, 0.30)
        kw = KeywordDetector(stems_by_scenario.get(sid, []))
        print(f"\n{'='*66}\nScenario: {sid}   threshold={thr}\n{'='*66}")

        cat_results = {}
        for category, sentences in EVAL_SETS[sid].items():
            above = 0
            scores = []
            stem_overlap = 0
            for text in sentences:
                score = detector.score(text).get(sid, float("-inf"))
                scores.append(score)
                if score >= thr:
                    above += 1
                if category.startswith("paraphrase") and kw.scan(text):
                    stem_overlap += 1
            cat_results[category] = (above, len(sentences), scores)
            avg = sum(scores) / len(scores)
            note = ""
            if category.startswith("paraphrase") and stem_overlap:
                note = f"  [!] {stem_overlap} paraphrase items overlap a stem (invalidates test)"
            print(
                f"  {category:16s} {above:2d}/{len(sentences):2d} above  "
                f"(avg={avg:+.3f}, min={min(scores):+.3f}, max={max(scores):+.3f}){note}"
            )

        # --- Acceptance checks ---
        if sid == "politics":
            a, n, _ = cat_results["paraphrase"]
            ok = pct(a, n) >= 80.0
            overall_ok &= ok
            print(f"  -> paraphrase >= 80%: {pct(a, n):.0f}%  {'PASS' if ok else 'FAIL'}")
        elif sid == "ai_future":
            ap, np_, _ = cat_results["paraphrase_pos"]
            an, nn, _ = cat_results["paraphrase_neg"]
            comb_ok = pct(ap + an, np_ + nn) >= 90.0
            pos_ok = pct(ap, np_) >= 90.0
            neg_ok = pct(an, nn) >= 90.0
            overall_ok &= comb_ok and pos_ok and neg_ok
            print(f"  -> paraphrase combined >= 90%: {pct(ap+an, np_+nn):.0f}%  {'PASS' if comb_ok else 'FAIL'}")
            print(f"     positive valence >= 90%:    {pct(ap, np_):.0f}%  {'PASS' if pos_ok else 'FAIL'}")
            print(f"     negative valence >= 90%:    {pct(an, nn):.0f}%  {'PASS' if neg_ok else 'FAIL'}")

        ha, hn, _ = cat_results["hesitation"]
        hes_ok = ha == 0
        overall_ok &= hes_ok
        print(f"  -> hesitation == 0 above: {ha}/{hn}  {'PASS' if hes_ok else 'FAIL (near-misses acceptable)'}")

    # --- Stem safety across the neutral corpus ---
    print(f"\n{'='*66}\nStem safety (neutral corpus, {len(NEUTRAL_CORPUS)} sentences)\n{'='*66}")
    stem_safe = True
    for sid, stems in stems_by_scenario.items():
        kw = KeywordDetector(stems)
        hits = []
        for text in NEUTRAL_CORPUS:
            for stem, _ in kw.find_matches(text):
                hits.append((stem, text))
        if hits:
            stem_safe = False
            print(f"  {sid}: {len(hits)} FALSE HITS")
            for stem, text in hits[:10]:
                print(f"      '{stem}' in \"{text}\"")
        else:
            print(f"  {sid}: 0 false hits  PASS")
    overall_ok &= stem_safe

    print(f"\n{'='*66}\nOVERALL: {'PASS' if overall_ok else 'FAIL — tune exemplars/thresholds'}\n{'='*66}")
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
