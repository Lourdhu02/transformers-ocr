import re
import numpy as np
import torch


class Encoder:
    def __init__(self, chars: str):
        self.chars       = chars
        self.c2i         = {c: i + 1 for i, c in enumerate(chars)}
        self.i2c         = {i + 1: c for i, c in enumerate(chars)}
        self.blank       = 0
        self.num_classes = len(chars) + 1

    def encode(self, text: str):
        return [self.c2i[c] for c in text if c in self.c2i]

    def decode_greedy(self, seq) -> str:
        out, prev = [], -1
        for i in seq:
            i = int(i)
            if i != 0 and i != prev:
                out.append(self.i2c.get(i, ""))
            prev = i
        return "".join(out)

    def decode_beam(self, log_probs: np.ndarray, beam_width: int = 10,
                    lexicon_pattern: str = r"^\d{4,8}(\.\d{1,2})?$") -> tuple[str, float]:
        T, V = log_probs.shape
        beams = {("",): (0.0, False)}

        for t in range(T):
            new_beams: dict = {}
            for prefix, (score, ends_blank) in beams.items():
                for v in range(V):
                    p = log_probs[t, v]
                    if v == 0:
                        new_prefix  = prefix
                        new_blank   = True
                        new_score   = score + p
                    else:
                        char = self.i2c.get(v, "")
                        last = prefix[-1] if prefix else None
                        if char == last and not ends_blank:
                            new_prefix = prefix
                        else:
                            new_prefix = prefix + (char,)
                        new_blank  = False
                        new_score  = score + p

                    key = new_prefix
                    prev_score, _ = new_beams.get(key, (-1e9, False))
                    if new_score > prev_score:
                        new_beams[key] = (new_score, new_blank)

            beams = dict(sorted(new_beams.items(),
                                key=lambda x: x[1][0], reverse=True)[:beam_width])

        results = []
        for prefix, (score, _) in beams.items():
            text = "".join(prefix)
            results.append((text, float(score)))

        if lexicon_pattern:
            pat     = re.compile(lexicon_pattern)
            matched = [(t, s) for t, s in results if pat.match(t)]
            if matched:
                return max(matched, key=lambda x: x[1])

        return results[0] if results else ("", -1e9)

    def confidence(self, log_probs: torch.Tensor) -> float:
        return log_probs.max(dim=-1).values.mean().exp().item()

