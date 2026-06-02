"""Tests for the H1 experiment matrix (arms A-G) and its fairness invariants."""

from __future__ import annotations

import pytest

from psalm.domain.experiments.matrix import (
    DECISIVE_PAIR,
    H1P_DECISIVE_PAIR,
    ExperimentMatrix,
    default_h1_matrix,
    default_h1p_matrix,
)
from psalm.domain.experiments.models import PrePretrainSource, PretrainCorpus


def test_default_matrix_has_eight_arms_a_through_h() -> None:
    m = default_h1_matrix(param_count_m=100.0)
    assert [a.arm_id for a in m.arms] == ["A", "B", "C", "D", "E", "F", "G", "H"]


def test_decisive_pair_is_b_vs_c() -> None:
    assert DECISIVE_PAIR == ("B", "C")
    m = default_h1_matrix(param_count_m=100.0)
    b, c = m.decisive_pair()
    assert b.pre_pretrain is PrePretrainSource.PANINIAN
    assert c.pre_pretrain is PrePretrainSource.DYCK


def test_b_and_c_are_matched_on_budget_arch_and_corpus() -> None:
    m = default_h1_matrix(param_count_m=100.0)
    b, c = m.decisive_pair()
    assert m.matched(b, c)
    # The only thing that may differ between B and C is the structural prior.
    assert b.param_count_m == c.param_count_m
    assert b.token_budget == c.token_budget
    assert b.pretrain_corpus == c.pretrain_corpus
    assert b.pre_pretrain != c.pre_pretrain


def test_low_budget_triplet_efg_is_matched() -> None:
    # E (Paninian/10M) vs F (Dyck/10M) vs G (None/10M): matched low-budget controls.
    m = default_h1_matrix(param_count_m=100.0)
    e = m.arm("E")
    f = m.arm("F")
    g = m.arm("G")
    assert e.token_budget == f.token_budget == g.token_budget
    assert m.matched(e, f) and m.matched(f, g)
    assert e.token_budget < m.arm("B").token_budget  # data-efficiency arm uses less NL


def test_all_arms_share_param_count_and_corpus() -> None:
    m = default_h1_matrix(param_count_m=123.0)
    assert all(a.param_count_m == 123.0 for a in m.arms)
    assert all(a.pretrain_corpus is PretrainCorpus.ENGLISH for a in m.arms)


def test_verify_fairness_passes_for_default_matrix() -> None:
    m = default_h1_matrix(param_count_m=100.0)
    problems = m.verify_fairness()
    assert problems == []


def test_verify_fairness_flags_mismatched_decisive_pair() -> None:
    m = default_h1_matrix(param_count_m=100.0)
    # Corrupt C's budget so B vs C is no longer matched.
    bad = [
        a.model_copy(update={"token_budget": a.token_budget * 2}) if a.arm_id == "C" else a
        for a in m.arms
    ]
    m2 = ExperimentMatrix(arms=bad, seeds=m.seeds)
    problems = m2.verify_fairness()
    assert any("B" in p and "C" in p for p in problems)


def test_seeds_default_to_at_least_three() -> None:
    m = default_h1_matrix(param_count_m=100.0)
    assert len(m.seeds) >= 3


def test_matrix_rejects_duplicate_arm_ids() -> None:
    m = default_h1_matrix(param_count_m=100.0)
    dup = [*m.arms, m.arms[0]]
    with pytest.raises(ValueError, match="duplicate"):
        ExperimentMatrix(arms=dup, seeds=m.seeds)


def test_arm_lookup_unknown_raises() -> None:
    m = default_h1_matrix(param_count_m=100.0)
    with pytest.raises(KeyError):
        m.arm("Z")


def test_run_plan_is_arms_times_seeds() -> None:
    m = default_h1_matrix(param_count_m=100.0, seeds=(0, 1, 2))
    plan = m.run_plan()
    assert len(plan) == 8 * 3  # arms A-H
    assert (m.arm("B"), 1) in plan


def test_default_h1p_matrix_arms_and_decisive_pair() -> None:
    assert H1P_DECISIVE_PAIR == ("h1p:B", "h1p:C")
    m = default_h1p_matrix(param_count_m=100.0, include_l1=True)
    assert [a.arm_id for a in m.arms] == ["h1p:A", "h1p:B", "h1p:C", "h1p:L"]
    b, c = m.arm("h1p:B"), m.arm("h1p:C")
    assert b.pre_pretrain is PrePretrainSource.SHABDABODHA_ALIGNED
    assert c.pre_pretrain is PrePretrainSource.DYCK
    assert m.matched(b, c)


def test_default_h1p_matrix_without_l1() -> None:
    m = default_h1p_matrix(param_count_m=60.0, include_l1=False)
    assert [a.arm_id for a in m.arms] == ["h1p:A", "h1p:B", "h1p:C"]
