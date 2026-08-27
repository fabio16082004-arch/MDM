import random

NUM_WORDS = {1: 'one', 2: 'two', 3: 'three', 4: 'four'}

TEMPLATES = [
    "a person takes {NUM} step{S}.",
    "a man walks {NUM} step{S}.",
    "someone takes {NUM} step{S}.",
    "the figure walks {NUM} step{S}.",
    "a person steps {NUM} step{S}.",
    "the man takes {NUM} step{S}.",
    "a person moves, taking {NUM} step{S}.",
    "someone walks {NUM} step{S}.",
]


def render_captions(L, seed=None):
    rng = random.Random(seed)
    out = []
    for t in TEMPLATES:
        as_word = rng.choice([True, False])
        num_token = NUM_WORDS[L] if as_word else str(L)
        s = '' if L == 1 else 's'
        out.append(t.format(NUM=num_token, S=s))
    return out


def build_hand_written_captions(categories=(1, 2, 3, 4), seed=7):
    """Costruisce la struttura {L: [caption, ...]} da passare a
    CustomDatasetLoader / FixedEvalDatasetLoader."""
    return {L: render_captions(L, seed=seed + L) for L in categories}


if __name__ == '__main__':
    captions = build_hand_written_captions()
    for L in (1, 2, 3, 4):
        print(f"L={L}:")
        for c in captions[L]:
            print(f"  {c}")