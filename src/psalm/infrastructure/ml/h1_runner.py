"""Real H1 arm runner: generators -> tokenizer -> pre-pretrain -> NL pretrain -> eval.

Wires the Phase-1 generators and Phase-2 model/eval into the callable the
``H1Orchestrator`` expects. One runner instance holds the shared tokenizer and
the evaluation sets so every arm is scored identically (a fairness requirement).

Per arm and seed:
  1. assemble the arm's pre-pretraining stream (Pāṇinian / Dyck / none) to budget,
  2. continue on the NL corpus to the arm's NL budget,
  3. evaluate compositional exact-match and report tokens-to-quality.

This module is GPU-bound for real sizes; it is excluded from mypy/coverage and is
driven by scripts under ``scripts/`` and by the orchestrator. The pure decision
logic it feeds is tested separately.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from psalm.application.data.assembly import PrePretrainAssembler
from psalm.domain.experiments.matrix import ExperimentArm
from psalm.domain.experiments.models import PrePretrainSource
from psalm.domain.model.config import ModelConfig
from psalm.domain.model.training import TrainConfig
from psalm.infrastructure.ml.eval_lm import greedy_generate
from psalm.infrastructure.ml.trainer import train_two_phase

#: Diversity-capped structural budget (whitespace tokens), matched across all
#: pre-pretrain arms so B (Pāṇinian, no-repeat) and C (Dyck, fresh) differ only
#: in content. Set from the source's no-repeat token count (ADR-0013 / the
#: structural-diversity finding); 60k is below the current cache's ~71k ceiling.
DEFAULT_PRE_BUDGET_TOKENS = 60_000

#: Downstream checkpoint fractions -> within-run token-savings curve.
DEFAULT_EVAL_FRACS: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)


@dataclass
class EvalSets:
    """Held-out evaluation material, identical across arms."""

    compositional: list[tuple[str, str]]  # (source, gold_target)
    minimal_pairs: list[tuple[str, str]]  # (acceptable, unacceptable)


class H1Runner:
    """Callable runner binding the full train+eval pipeline for one arm/seed."""

    def __init__(
        self,
        *,
        assembler: PrePretrainAssembler,
        nl_lines: Callable[[], Iterable[str]],
        encode: Callable[[str], list[int]],
        eos_id: int,
        model_cfg: ModelConfig,
        train_cfg: TrainConfig,
        eval_sets: EvalSets,
        pretrain_max_new: int = 32,
        decode: Callable[[list[int]], str] | None = None,
        append_eos_to_prompt: bool = True,
        pre_budget_tokens: int = DEFAULT_PRE_BUDGET_TOKENS,
        eval_fracs: tuple[float, ...] = DEFAULT_EVAL_FRACS,
        nl_budget_tokens: int | None = None,
        pre_epochs: int = 1,
        extra_eval_sets: dict[str, list[tuple[str, str]]] | None = None,
        comp_max_new_cap: int | None = None,
        disc_eval_sets: dict[str, list[tuple[str, str]]] | None = None,
        disc_curve_pairs: list[tuple[str, str]] | None = None,
    ) -> None:
        self.assembler = assembler
        self.nl_lines = nl_lines
        self.encode = encode
        self.eos_id = eos_id
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.eval_sets = eval_sets
        self.pretrain_max_new = pretrain_max_new
        self.pre_budget_tokens = pre_budget_tokens
        self.eval_fracs = eval_fracs
        # Downstream token budget. Defaults to the arm's NL budget, but can be
        # overridden (proxy/pilot) so a run does not always cost the full 130M.
        self.nl_budget_tokens = nl_budget_tokens
        # Matched-epoch structural dose: repeat the capped unique set N times for
        # every pre-pretrain arm (ADR-0014), raising dose while staying matched.
        self.pre_epochs = pre_epochs
        # Real benchmarks (SCAN/COGS) need a true detokenize; the default
        # id-join is only meaningful for the char-proxy. For SCAN-as-task the
        # prompt ("IN: cmd OUT:") must NOT carry EOS or the model sees the
        # sequence as already finished.
        self.decode = decode if decode is not None else _decode_ids
        self.append_eos_to_prompt = append_eos_to_prompt
        # Secondary named eval sets (e.g. COGS structural-generalization tier),
        # scored on the *same* trained model so multi-tier reporting costs no
        # extra training (ADR-0014).
        self.extra_eval_sets = extra_eval_sets or {}
        # Role-discrimination minimal pairs (ADR-0015): each entry is a list of
        # (correct_full_text, role_corrupted_full_text). Scored by teacher-forced
        # sequence logprob (no generation) — off-floor by construction (chance=50%).
        self.disc_eval_sets = disc_eval_sets or {}
        # Sample-efficiency learning curve (ADR-0016): when set, within-run
        # checkpoints score role-discrimination accuracy on these pairs (instead
        # of compositional EM), yielding a discrimination-vs-tokens curve.
        self.disc_curve_pairs = disc_curve_pairs
        # Auto-size generation so the model can emit the *longest* gold target in
        # the PRIMARY eval set. Capping below the target length silently forces
        # exact-match to 0 for every long example (a battery-invalidating trap),
        # so we never generate fewer tokens than the longest primary gold + margin.
        # Secondary tiers (e.g. COGS recursion) are graded (token-F1) and may have
        # far longer targets; sizing to them would make generation O(len) for
        # *every* example, so they are deliberately not used here and an explicit
        # ``comp_max_new_cap`` bounds cost (a truncated long target floors EM,
        # which the secondary tier is expected to do anyway — ADR-0014).
        max_gold_len = max(
            (len(self.encode(gold)) for _, gold in eval_sets.compositional), default=0
        )
        self.comp_max_new = max(pretrain_max_new, max_gold_len + 4)
        if comp_max_new_cap is not None:
            self.comp_max_new = min(self.comp_max_new, comp_max_new_cap)

    def _pre_lines(self, arm: ExperimentArm, seed: int) -> Callable[[], Iterable[str]] | None:
        """Structural pre-pretraining stream (None for arms A/G), diversity-capped."""
        if arm.pre_pretrain is PrePretrainSource.NONE:
            return None

        def make() -> Iterable[str]:
            yield from self.assembler.take_until_tokens(
                arm.pre_pretrain, budget_tokens=self.pre_budget_tokens, seed=seed
            )

        return make

    def _predict(
        self, model: object, device: str, pairs: list[tuple[str, str]]
    ) -> tuple[list[str], list[str]]:
        preds: list[str] = []
        golds: list[str] = []
        for source, gold in pairs:
            prompt = list(self.encode(source))
            if self.append_eos_to_prompt:
                prompt.append(self.eos_id)
            out_ids = greedy_generate(
                model,
                prompt,
                max_new_tokens=self.comp_max_new,
                device=device if device == "cpu" else "cuda",
                eos_id=self.eos_id,
            )
            preds.append(self.decode(out_ids).strip())
            golds.append(gold.strip())
        return preds, golds

    def _eval_compositional(self, model: object, device: str) -> float:
        from psalm.domain.eval.metrics import exact_match_accuracy

        preds, golds = self._predict(model, device, self.eval_sets.compositional)
        return exact_match_accuracy(preds, golds) if preds else 0.0

    def __call__(self, arm: ExperimentArm, seed: int) -> dict[str, float]:
        cfg = self.train_cfg.model_copy(update={"seed": seed})
        aux_vocab = self.model_cfg.vocab_size if arm.karaka_aux_loss else 0
        if arm.karaka_aux_loss:
            cfg = cfg.model_copy(update={"aux_loss_weight": max(cfg.aux_loss_weight, 0.5)})

        nl_budget = self.nl_budget_tokens if self.nl_budget_tokens is not None else arm.token_budget
        pre_lines = self._pre_lines(arm, seed)

        # Checkpoint eval: discrimination learning curve (ADR-0016) when curve
        # pairs are provided, else compositional EM (ADR-0014).
        if self.disc_curve_pairs:
            from psalm.domain.eval.metrics import minimal_pair_accuracy
            from psalm.infrastructure.ml.eval_lm import minimal_pair_scores

            _dev = cfg.device if cfg.device == "cpu" else "cuda"
            curve_pairs = self.disc_curve_pairs

            def _checkpoint_eval(m: object) -> float:
                scores = minimal_pair_scores(
                    m, curve_pairs, encode=self.encode, device=_dev, eos_id=self.eos_id
                )
                return minimal_pair_accuracy(scores)
        else:

            def _checkpoint_eval(m: object) -> float:
                return self._eval_compositional(m, cfg.device)

        model, outcome = train_two_phase(
            self.model_cfg,
            cfg,
            pre_make_lines=pre_lines,
            nl_make_lines=self.nl_lines,
            pre_max_tokens=self.pre_budget_tokens if pre_lines is not None else 0,
            nl_max_tokens=nl_budget,
            encode=self.encode,
            eos_id=self.eos_id,
            aux_vocab=aux_vocab,
            pre_epochs=self.pre_epochs,
            eval_fracs=self.eval_fracs,
            eval_fn=_checkpoint_eval,
        )

        accuracy = self._eval_compositional(model, cfg.device)
        metrics = {
            "compositional_accuracy": accuracy,
            "tokens_to_quality": float(outcome.nl.tokens),
            "final_loss": outcome.nl.last_loss,
            "structural_tokens": float(outcome.pre.tokens),
        }
        # Within-run efficiency curve: accuracy at each downstream checkpoint.
        for nl_tokens, metric in outcome.checkpoints:
            metrics[f"acc_at_{nl_tokens}"] = metric
        # Secondary tiers: graded readouts on the same trained model (ADR-0014).
        if self.extra_eval_sets:
            from psalm.domain.eval.metrics import (
                exact_match_accuracy,
                length_binned_accuracy,
                token_f1_score,
            )

            for name, pairs in self.extra_eval_sets.items():
                if not pairs:
                    continue
                preds, golds = self._predict(model, cfg.device, pairs)
                metrics[f"{name}_em"] = exact_match_accuracy(preds, golds)
                metrics[f"{name}_f1"] = token_f1_score(preds, golds)
                for bucket, val in length_binned_accuracy(preds, golds).items():
                    metrics[f"{name}_len_{bucket}"] = val
        # Role-discrimination readout (ADR-0015): minimal-pair likelihood accuracy.
        if self.disc_eval_sets:
            from psalm.domain.eval.metrics import minimal_pair_accuracy
            from psalm.infrastructure.ml.eval_lm import minimal_pair_scores

            device = cfg.device if cfg.device == "cpu" else "cuda"
            for name, pairs in self.disc_eval_sets.items():
                if not pairs:
                    continue
                scores = minimal_pair_scores(
                    model, pairs, encode=self.encode, device=device, eos_id=self.eos_id
                )
                metrics[f"{name}_disc"] = minimal_pair_accuracy(scores)
                metrics[f"{name}_disc_n"] = float(len(pairs))
        return metrics


def _decode_ids(ids: list[int]) -> str:
    return " ".join(str(i) for i in ids)


def none_source_is_empty(arm: ExperimentArm) -> bool:
    """Helper used by scripts/tests: arms A and G have no pre-pretraining."""
    return arm.pre_pretrain is PrePretrainSource.NONE
