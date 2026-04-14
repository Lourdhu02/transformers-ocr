import re
import numpy as np
import torch


class Encoder:
    """Maps characters <-> integer indices for CTC training and decoding."""

    def __init__(self, chars: str):
        self.chars       = chars
        self.c2i         = {c: i + 1 for i, c in enumerate(chars)}
        self.i2c         = {i + 1: c for i, c in enumerate(chars)}
        self.blank       = 0
        self.num_classes = len(chars) + 1
        self._decoder_cache: dict = {}

    def encode(self, text: str) -> list[int]:
        encoded = [self.c2i[c] for c in text if c in self.c2i]
        unknown = [c for c in text if c not in self.c2i]
        if unknown:
            import warnings
            warnings.warn(
                f"Encoder skipped unknown chars {unknown!r} in label {text!r}. "
                "Check your charset or label file.",
                stacklevel=2,
            )
        return encoded

    def decode_greedy(self, seq) -> str:
        out, prev = [], -1
        for i in seq:
            i = int(i)
            if i != 0 and i != prev:
                out.append(self.i2c.get(i, ""))
            prev = i
        return "".join(out)

    def _build_ctc_decoder(self, beam_width: int):
        """Build a C++ CTCBeamDecoder (parlance/ctcdecode). Falls back to pure Python if not installed."""
        try:
            from ctcdecode import CTCBeamDecoder
            labels = ["_blank_"] + list(self.chars)
            return CTCBeamDecoder(
                labels,
                model_path=None,
                alpha=0,
                beta=0,
                cutoff_top_n=len(labels),
                cutoff_prob=1.0,
                beam_width=beam_width,
                num_processes=4,
                blank_id=0,
                log_probs_input=True,
            )
        except ImportError:
            return None

    def decode_beam(
        self,
        log_probs: np.ndarray,
        beam_width: int = 10,
        lexicon_pattern: str = r"^\d{4,8}(\.\d{1,2})?$",
    ) -> tuple[str, float]:
        """
        Beam search CTC decoder with lexicon filter.
        Uses C++ ctcdecode if installed (~50x faster), else pure Python fallback.
        """
        # ── Fast path: C++ decoder ──────────────────────────────────────────
        if beam_width not in self._decoder_cache:
            self._decoder_cache[beam_width] = self._build_ctc_decoder(beam_width)

        cpp_decoder = self._decoder_cache[beam_width]
        if cpp_decoder is not None:
            lp_t = torch.tensor(log_probs).unsqueeze(0)          # (1, T, V)
            beam_results, beam_scores, _, out_lens = cpp_decoder.decode(lp_t)
            pat = re.compile(lexicon_pattern) if lexicon_pattern else None
            for b in range(beam_results.size(1)):
                length = out_lens[0][b].item()
                ids    = beam_results[0][b][:length].tolist()
                text   = "".join(self.i2c.get(i, "") for i in ids)
                score  = float(beam_scores[0][b])
                if pat is None or pat.match(text):
                    return text, score
            # fallback to top beam if nothing matches lexicon
            length = out_lens[0][0].item()
            ids    = beam_results[0][0][:length].tolist()
            return "".join(self.i2c.get(i, "") for i in ids), float(beam_scores[0][0])

        # ── Slow path: pure Python (no ctcdecode installed) ─────────────────
        T, V = log_probs.shape
        beams: dict[tuple, float] = {((), False): 0.0}

        for t in range(T):
            new_beams: dict[tuple, float] = {}
            for (prefix, ends_blank), score in beams.items():
                for v in range(V):
                    p = float(log_probs[t, v])
                    if v == 0:
                        key       = (prefix, True)
                        new_score = score + p
                    else:
                        char = self.i2c.get(v, "")
                        last = prefix[-1] if prefix else None
                        new_prefix = prefix if (char == last and not ends_blank) else prefix + (char,)
                        key        = (new_prefix, False)
                        new_score  = score + p
                    if new_score > new_beams.get(key, -1e9):
                        new_beams[key] = new_score
            beams = dict(
                sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width]
            )

        best: dict[str, float] = {}
        for (prefix, _), score in beams.items():
            text = "".join(prefix)
            if score > best.get(text, -1e9):
                best[text] = score

        results = sorted(best.items(), key=lambda x: x[1], reverse=True)

        if lexicon_pattern:
            pat     = re.compile(lexicon_pattern)
            matched = [(t, s) for t, s in results if pat.match(t)]
            if matched:
                return max(matched, key=lambda x: x[1])

        return results[0] if results else ("", -1e9)

    def confidence(self, log_probs: torch.Tensor) -> float:
        return log_probs.max(dim=-1).values.mean().exp().item()