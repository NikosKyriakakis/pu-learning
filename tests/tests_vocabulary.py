import pytest

from src.textprep.vocab import Vocabulary


@pytest.fixture
def default_vocabulary():
    return Vocabulary(add_unk=True, add_pad=True)


def test_add_first_token(default_vocabulary):
    assert len(default_vocabulary) == 2
    default_vocabulary.add_token("fake-token")
    assert len(default_vocabulary) == 3


def test_lookup_token_not_found(default_vocabulary):
    assert default_vocabulary.lookup_token("fake-token") == default_vocabulary.unk_index


def test_lookup_token_found(default_vocabulary):
    default_vocabulary.add_token("fake-token")
    assert default_vocabulary.lookup_token("fake-token") == 2
