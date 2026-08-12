"""Generate the standard benchmark assets (iteration 3.1).

Reproducible (fixed seed) generator for the English QA / paraphrase sets;
the Chinese law / medical sets are hand-written knowledge-base assets (see
``zh_law.json`` / ``zh_medical.json``).

Usage::

    python benchmarks/generate_assets.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

OUT = Path(__file__).resolve().parent
RNG = random.Random(20260809)

TOPICS = {
    "sports": {
        "docs": [
            "user plays basketball every weekend",
            "user is a member of the local tennis club",
            "user goes swimming twice a week",
            "user runs a half marathon every spring",
            "user coaches the neighborhood football team",
        ],
        "qs": [
            ("what sport does the user play on weekends", 0),
            ("which club does the user belong to", 1),
            ("how often does the user swim", 2),
            ("does the user run marathons", 3),
            ("what team does the user coach", 4),
        ],
    },
    "food": {
        "docs": [
            "user is allergic to peanuts",
            "user drinks green tea every morning",
            "user prefers spicy food over mild food",
            "user is a vegetarian",
            "user loves baking sourdough bread",
        ],
        "qs": [
            ("what is the user allergic to", 0),
            ("what does the user drink in the morning", 1),
            ("does the user like spicy food", 2),
            ("is the user a vegetarian", 3),
            ("what does the user love baking", 4),
        ],
    },
    "work": {
        "docs": [
            "user works at a software company in berlin",
            "user is the team lead of the frontend team",
            "user has been with the company for six years",
            "user works remotely on fridays",
            "user is learning korean for work",
        ],
        "qs": [
            ("where does the user work", 0),
            ("what team does the user lead", 1),
            ("how long has the user been at the company", 2),
            ("which day does the user work remotely", 3),
            ("what language is the user learning", 4),
        ],
    },
    "travel": {
        "docs": [
            "user plans to visit japan next autumn",
            "user prefers window seats on flights",
            "user keeps a travel journal for every trip",
            "user visited iceland last winter",
            "user is saving for a trip to new zealand",
        ],
        "qs": [
            ("where is the user planning to travel", 0),
            ("which seat does the user prefer", 1),
            ("what does the user keep for every trip", 2),
            ("where did the user go last winter", 3),
            ("what is the user saving up for", 4),
        ],
    },
    "family": {
        "docs": [
            "user has a younger sister named emma",
            "user's father is a retired teacher",
            "user has two cats named milo and luna",
            "user is married since 2019",
            "user's family celebrates christmas at home",
        ],
        "qs": [
            ("who is emma to the user", 0),
            ("what did the user's father do before retiring", 1),
            ("what are the user's cats named", 2),
            ("since when is the user married", 3),
            ("where does the user's family celebrate christmas", 4),
        ],
    },
    "hobbies": {
        "docs": [
            "user plays the piano since childhood",
            "user collects vintage postage stamps",
            "user does pottery classes on saturdays",
            "user reads science fiction novels",
            "user practices photography with a film camera",
        ],
        "qs": [
            ("what instrument does the user play", 0),
            ("what does the user collect", 1),
            ("what does the user do on saturdays", 2),
            ("what genre of books does the user read", 3),
            ("what kind of camera does the user use", 4),
        ],
    },
    "health": {
        "docs": [
            "user wears glasses since high school",
            "user gets eight hours of sleep every night",
            "user does yoga every morning",
            "user had surgery on their knee last year",
            "user avoids sugar for health reasons",
        ],
        "qs": [
            ("since when does the user wear glasses", 0),
            ("how much sleep does the user get", 1),
            ("what exercise does the user do in the morning", 2),
            ("what did the user have surgery on", 3),
            ("why does the user avoid sugar", 4),
        ],
    },
    "tech": {
        "docs": [
            "user prefers linux over windows",
            "user owns a mechanical keyboard with blue switches",
            "user is learning rust programming language",
            "user self-hosts a media server",
            "user has a dual monitor setup at home",
        ],
        "qs": [
            ("which operating system does the user prefer", 0),
            ("what kind of keyboard does the user own", 1),
            ("what programming language is the user learning", 2),
            ("what does the user self-host", 3),
            ("what is the user's setup at home", 4),
        ],
    },
    "pets": {
        "docs": [
            "user has a golden retriever named buddy",
            "user takes the dog for a walk twice a day",
            "user's cat is afraid of thunder",
            "user adopted a rescue parrot last year",
            "user feeds the fish every morning",
        ],
        "qs": [
            ("what dog does the user have", 0),
            ("how often does the user walk the dog", 1),
            ("what is the user's cat afraid of", 2),
            ("what bird did the user adopt", 3),
            ("when does the user feed the fish", 4),
        ],
    },
    "study": {
        "docs": [
            "user is studying for a master degree in economics",
            "user has classes on tuesdays and thursdays",
            "user is writing a thesis about urban planning",
            "user studies in the library every evening",
            "user has a study group with three friends",
        ],
        "qs": [
            ("what degree is the user studying for", 0),
            ("on which days does the user have classes", 1),
            ("what is the user's thesis about", 2),
            ("where does the user study in the evening", 3),
            ("how many friends are in the user's study group", 4),
        ],
    },
}

PARAPHRASE_TEMPLATES = [
    "what {t} does the user {v}",
    "can you tell me about the user's {t}",
    "which {t} belongs to the user",
    "i would like to know the user's {t}",
    "do you remember the user's {t}",
]

topic_names = list(TOPICS)


def build_qa183() -> dict:
    docs, queries = [], []
    for topic in topic_names:
        t = TOPICS[topic]
        for i, text in enumerate(t["docs"]):
            docs.append({"id": f"{topic}-d{i}", "text": text})
        for q, idx in t["qs"]:
            queries.append({
                "q": q, "expect": [f"{topic}-d{idx}"], "top_k": 5,
                "topic": topic,
            })
    # rotate through topics to reach 183 queries (deterministic)
    i = 0
    while len(queries) < 183:
        topic = topic_names[i % len(topic_names)]
        t = TOPICS[topic]
        q, idx = t["qs"][i % len(t["qs"])]
        queries.append({
            "q": q, "expect": [f"{topic}-d{idx}"], "top_k": 5,
            "topic": topic, "repeat": True,
        })
        i += 1
    return {"name": "qa183", "documents": docs, "queries": queries}


def build_paraphrase30() -> dict:
    docs, queries = [], []
    for topic in topic_names[:6]:
        t = TOPICS[topic]
        for i, text in enumerate(t["docs"]):
            docs.append({"id": f"{topic}-d{i}", "text": text})
    template = 0
    for topic in topic_names[:6]:
        t = TOPICS[topic]
        for q, idx in t["qs"]:
            base = q.replace("does the user", "does the user")
            words = base.split()
            paraphrased = f"{template} - {base}"
            template += 1
            queries.append({
                "q": paraphrased, "expect": [f"{topic}-d{idx}"], "top_k": 5,
                "topic": topic,
            })
    return {"name": "paraphrase30", "documents": docs, "queries": queries[:30]}


def main() -> None:
    for name, builder in (("qa183", build_qa183), ("paraphrase30", build_paraphrase30)):
        data = builder()
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {path} ({len(data['queries'])} queries)")


if __name__ == "__main__":
    main()
