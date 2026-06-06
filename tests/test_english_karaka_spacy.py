"""Unit tests for spaCy-based English kāraka role assignment.

Tests the real dependency parser approach against known linguistic examples,
especially the -s verb bug that the surface heuristic had.
"""

import pytest

from psalm.domain.linguistics.english_karaka_real import (
    assign_karaka_roles_spacy,
    parse_and_assign,
    roles_to_dict,
)


@pytest.fixture
def nlp():
    """Load spaCy model once for all tests."""
    import spacy

    return spacy.load("en_core_web_sm")


class TestDepToKaraka:
    """Test dependency relation to kāraka mapping."""

    def test_subject_to_karta(self, nlp):
        """nsubj (nominal subject) → kartā."""
        doc = nlp("The cat sat.")
        roles = assign_karaka_roles_spacy(doc)
        roles_dict = roles_to_dict(roles)
        # "cat" is nsubj (subject)
        assert roles_dict["cat"] == "karta"

    def test_object_to_karma(self, nlp):
        """dobj/obj (direct object) → karma."""
        doc = nlp("The cat ate food.")
        roles = assign_karaka_roles_spacy(doc)
        roles_dict = roles_to_dict(roles)
        # "food" is obj (direct object)
        assert roles_dict["food"] == "karma"

    def test_verb_to_kriya(self, nlp):
        """ROOT / VERB pos → kriyā (action)."""
        doc = nlp("The cat ate food.")
        roles = assign_karaka_roles_spacy(doc)
        roles_dict = roles_to_dict(roles)
        # "ate" is ROOT (main verb)
        assert roles_dict["ate"] == "kriya"

    def test_preposition_with_to_karana(self, nlp):
        """prep=with / "with" text → karaṇa (instrument)."""
        doc = nlp("The cat ate food with a fork.")
        roles = assign_karaka_roles_spacy(doc)
        roles_dict = roles_to_dict(roles)
        # "with" is ADP (adposition/preposition)
        assert roles_dict["with"] == "karana"
        # "fork" is in obl (oblique, attached to "ate" via "with")
        assert roles_dict["fork"] == "karana"

    def test_preposition_in_to_adhikarana(self, nlp):
        """prep=in / "in" text → adhikaraṇa (locus)."""
        doc = nlp("The cat sat on the mat.")
        roles = assign_karaka_roles_spacy(doc)
        roles_dict = roles_to_dict(roles)
        # "on" is ADP
        assert roles_dict["on"] == "adhikarana"
        # "mat" is in obl (locative)
        assert roles_dict["mat"] == "adhikarana"

    def test_preposition_from_to_apadana(self, nlp):
        """prep=from / "from" text → apādāna (source)."""
        doc = nlp("The bird flew from the tree.")
        roles = assign_karaka_roles_spacy(doc)
        roles_dict = roles_to_dict(roles)
        # "from" is ADP
        assert roles_dict["from"] == "apadana"
        # "tree" is in obl (source/ablative)
        assert roles_dict["tree"] == "apadana"

    def test_adjective_to_visesana(self, nlp):
        """amod (adjectival modifier) → viśeṣaṇa."""
        doc = nlp("The red fox ran.")
        roles = assign_karaka_roles_spacy(doc)
        roles_dict = roles_to_dict(roles)
        # "red" is amod (adjectival modifier)
        assert roles_dict["red"] == "visesana"

    def test_determiner_to_separator(self, nlp):
        """det (determiner) → separator."""
        doc = nlp("The cat sat.")
        roles = assign_karaka_roles_spacy(doc)
        roles_dict = roles_to_dict(roles)
        # "The" is det (determiner)
        assert roles_dict["The"] == "separator"


class TestThirdPersonVerbBug:
    """Test the specific bug: 3rd person -s verbs were misclassified.

    The surface heuristic missed "jumps", "runs", "flows" because they end
    in "s" without a suffix. The dependency parser should get them right.
    """

    def test_jumps_is_verb_not_modifier(self, nlp):
        """'jumps' should be kriyā (verb/ROOT), not viśeṣaṇa."""
        doc = nlp("The fox jumps.")
        roles = assign_karaka_roles_spacy(doc)
        roles_dict = roles_to_dict(roles)
        # "jumps" is ROOT (main verb), not a modifier
        assert roles_dict["jumps"] == "kriya"

    def test_runs_is_verb_not_modifier(self, nlp):
        """'runs' should be kriyā (verb), not viśeṣaṇa."""
        doc = nlp("The dog runs.")
        roles = assign_karaka_roles_spacy(doc)
        roles_dict = roles_to_dict(roles)
        assert roles_dict["runs"] == "kriya"

    def test_flows_is_verb_not_modifier(self, nlp):
        """'flows' should be kriyā, not viśeṣaṇa."""
        doc = nlp("Water flows.")
        roles = assign_karaka_roles_spacy(doc)
        roles_dict = roles_to_dict(roles)
        assert roles_dict["flows"] == "kriya"

    def test_eats_is_verb_not_modifier(self, nlp):
        """'eats' should be kriyā."""
        doc = nlp("The cat eats food.")
        roles = assign_karaka_roles_spacy(doc)
        roles_dict = roles_to_dict(roles)
        assert roles_dict["eats"] == "kriya"


class TestBabyLMSentences:
    """Real BabyLM-style sentences."""

    def test_simple_svo(self, nlp):
        """Simple SVO: The cat ate the food."""
        doc = nlp("The cat ate the food.")
        roles = assign_karaka_roles_spacy(doc)
        roles_dict = roles_to_dict(roles)
        assert roles_dict["cat"] == "karta"
        assert roles_dict["ate"] == "kriya"
        assert roles_dict["food"] == "karma"
        assert roles_dict["The"] == "separator"

    def test_intransitive(self, nlp):
        """Intransitive: The dog barked."""
        doc = nlp("The dog barked.")
        roles = assign_karaka_roles_spacy(doc)
        roles_dict = roles_to_dict(roles)
        assert roles_dict["dog"] == "karta"
        assert roles_dict["barked"] == "kriya"

    def test_with_instrument(self, nlp):
        """PP with instrument: ate food with a spoon."""
        doc = nlp("The dog ate food with a spoon.")
        roles = assign_karaka_roles_spacy(doc)
        roles_dict = roles_to_dict(roles)
        assert roles_dict["dog"] == "karta"
        assert roles_dict["ate"] == "kriya"
        assert roles_dict["food"] == "karma"
        assert roles_dict["with"] == "karana"
        assert roles_dict["spoon"] == "karana"

    def test_with_location(self, nlp):
        """PP with location: sat on the mat."""
        doc = nlp("The cat sat on the mat.")
        roles = assign_karaka_roles_spacy(doc)
        roles_dict = roles_to_dict(roles)
        assert roles_dict["cat"] == "karta"
        assert roles_dict["sat"] == "kriya"
        assert roles_dict["on"] == "adhikarana"
        assert roles_dict["mat"] == "adhikarana"

    def test_multiple_adjectives(self, nlp):
        """Multiple adjectives: The large brown dog ran quickly."""
        doc = nlp("The large brown dog ran quickly.")
        roles = assign_karaka_roles_spacy(doc)
        roles_dict = roles_to_dict(roles)
        # Both "large" and "brown" are modifiers
        assert roles_dict["large"] == "visesana"
        assert roles_dict["brown"] == "visesana"
        assert roles_dict["dog"] == "karta"
        assert roles_dict["ran"] == "kriya"
        # "quickly" is advmod (adverbial modifier)
        assert roles_dict["quickly"] == "visesana"

    def test_with_recipient(self, nlp):
        """PP with recipient: gave book to girl."""
        doc = nlp("The boy gave the book to the girl.")
        roles = assign_karaka_roles_spacy(doc)
        roles_dict = roles_to_dict(roles)
        assert roles_dict["boy"] == "karta"
        assert roles_dict["gave"] == "kriya"
        assert roles_dict["book"] == "karma"
        assert roles_dict["to"] == "sampradana"
        assert roles_dict["girl"] == "sampradana"

    def test_ablative(self, nlp):
        """Ablative: flew from the tree."""
        doc = nlp("A bird flew from the tree.")
        roles = assign_karaka_roles_spacy(doc)
        roles_dict = roles_to_dict(roles)
        assert roles_dict["bird"] == "karta"
        assert roles_dict["flew"] == "kriya"
        assert roles_dict["from"] == "apadana"
        assert roles_dict["tree"] == "apadana"


class TestParseAndAssign:
    """Test the end-to-end parse_and_assign function."""

    def test_parse_and_assign_integration(self, nlp):
        """Test parse_and_assign directly."""
        roles = parse_and_assign("The cat jumps.", nlp)
        roles_dict = roles_to_dict(roles)
        assert roles_dict["cat"] == "karta"
        assert roles_dict["jumps"] == "kriya"
        assert roles_dict["The"] == "separator"

    def test_handles_punctuation(self, nlp):
        """Punctuation is handled by spaCy."""
        roles = parse_and_assign("The cat jumps.", nlp)
        # spaCy marks punctuation as punct (dependency) and should get separator role
        assert len(roles) > 0  # At least the three words


class TestRolesToDict:
    """Test the roles_to_dict conversion."""

    def test_simple_conversion(self):
        """Convert TokenRole list to dict."""
        from psalm.domain.linguistics.english_karaka_real import TokenRole

        roles = [TokenRole("cat", "karta"), TokenRole("sat", "kriya")]
        d = roles_to_dict(roles)
        assert d == {"cat": "karta", "sat": "kriya"}

    def test_empty_list(self):
        """Empty list conversion."""
        assert roles_to_dict([]) == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
