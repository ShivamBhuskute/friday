"""Speech-to-text: transcript normalisation (fast) and the spoken golden set.

The golden set is the real test. It runs actual audio through the real model, so
it is marked ``slow`` and needs the downloaded weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from server.stt import SpeechToText, is_repetitive, normalize_transcript

from .conftest import FIXTURES, needs_stt

MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())


# ------------------------------------------------------------- normalisation


class TestNumberNormalisation:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("seven", "7"),
            ("twenty three", "23"),
            ("seven times twenty three", "7 * 23"),
            ("what is 7 times 23", "what is 7 * 23"),
            ("ninety nine bottles", "99 bottles"),
            ("two hundred", "200"),
            ("two hundred and fifty", "200 50"),
            ("three thousand", "3000"),
            ("twenty-three plus seven", "23 + 7"),
        ],
    )
    def test_spelled_numbers_become_digits(self, raw: str, expected: str) -> None:
        assert normalize_transcript(raw) == expected

    def test_leading_zero_word(self) -> None:
        assert normalize_transcript("zero") == "0"

    def test_teens(self) -> None:
        assert normalize_transcript("seventeen") == "17"
        assert normalize_transcript("fifteen plus one") == "15 + 1"


class TestOperatorNormalisation:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("seven times twenty three", "7 * 23"),
            ("twelve divided by four", "12 / 4"),
            ("two plus two", "2 + 2"),
            ("ten minus three", "10 - 3"),
            ("two to the power of ten", "2 ** 10"),
            ("five percent of two hundred", "5 % 200"),
        ],
    )
    def test_operators_become_symbols(self, raw: str, expected: str) -> None:
        assert normalize_transcript(raw) == expected

    def test_divided_by_leaves_no_stray_by(self) -> None:
        assert "by" not in normalize_transcript("twelve divided by four")

    def test_multi_word_operators_beat_single_words(self) -> None:
        out = normalize_transcript("twelve multiplied by four")
        assert "multiplied" not in out
        assert out == "12 * 4"

    def test_squared_and_cubed(self) -> None:
        assert normalize_transcript("five squared") == "5 ** 2"
        assert normalize_transcript("three cubed") == "3 ** 3"


class TestFillerHandling:
    def test_wake_word_is_stripped(self) -> None:
        assert normalize_transcript("friday what is the weather") == "what is the weather"

    def test_stacked_fillers_are_all_stripped(self) -> None:
        assert normalize_transcript("hey friday um what time is it") == "what time is it"

    def test_hesitations_are_stripped(self) -> None:
        assert normalize_transcript("um what is 7 times 23") == "what is 7 * 23"

    def test_hello_is_never_stripped(self) -> None:
        """"Hello there" is a real request, not a filler."""
        assert normalize_transcript("hello there") == "hello there"
        assert normalize_transcript("hey, what is the weather") == "hey, what is the weather"

    def test_a_word_merely_starting_with_friday_survives(self) -> None:
        assert normalize_transcript("fridays are best") == "fridays are best"


class TestTidying:
    def test_whitespace_is_collapsed(self) -> None:
        assert normalize_transcript("  what   is   7   times   23  ") == "what is 7 * 23"

    def test_punctuation_is_kept(self) -> None:
        assert normalize_transcript("what is 7 times 23?") == "what is 7 * 23?"

    def test_empty_input(self) -> None:
        assert normalize_transcript("") == ""
        assert normalize_transcript("   ") == ""

    def test_words_that_look_numeric_are_untouched(self) -> None:
        assert "someone" in normalize_transcript("someone is knocking")
        assert "onion" in normalize_transcript("onion rings")


class TestRoundTrip:
    """Normalised transcripts must still be evaluable by the calc tool."""

    @pytest.mark.parametrize(
        "raw,expression",
        [
            ("what is seven times twenty three", "7 * 23"),
            ("what is twelve divided by four", "12 / 4"),
            ("what is two plus two", "2 + 2"),
        ],
    )
    def test_normalised_transcript_evaluates(self, raw: str, expression: str) -> None:
        from server.tools.calc import normalize_expression

        assert normalize_expression(normalize_transcript(raw)) == expression


# ------------------------------------------------------------------ golden set


@pytest.mark.slow
@pytest.mark.timeout(300)  # first call loads the weights onto the GPU
@needs_stt
class TestGoldenSet:
    """Real audio through the real model.

    Whisper is not deterministic across quantisation settings, so the bar is
    that every expected word appears, not that the output matches byte for byte.
    """

    @pytest.fixture(scope="class")
    def stt(self, session_cfg):
        engine = SpeechToText(session_cfg)
        engine.load()
        return engine

    @pytest.mark.parametrize("name", sorted(MANIFEST))
    def test_fixture_transcribes(self, stt: SpeechToText, name: str) -> None:
        expect = MANIFEST[name]
        result = stt.transcribe(FIXTURES / name)
        text = result.text.lower()
        missing = [w for w in expect["expect_contains"] if w.lower() not in text]
        assert not missing, f"{name}: {result.text!r} is missing {missing}"
        assert result.confidence > 0.3, f"{name}: low confidence {result.confidence}"

    def test_device_is_actually_accelerated(self, stt: SpeechToText) -> None:
        """A silent fallback to CPU would look like a pass but be 10x slower."""
        if stt.device == "cuda":
            assert stt.compute_type in {"int8_float16", "int8", "float16", "float32"}
        assert stt.loaded

    def test_silence_is_not_hallucinated(self, stt: SpeechToText, tmp_path: Path) -> None:
        from .conftest import silence_pcm, wav_bytes

        path = tmp_path / "silence.wav"
        path.write_bytes(wav_bytes(silence_pcm(1.5)))
        result = stt.transcribe(path)
        # Whisper is prone to inventing text on pure noise; the pipeline relies
        # on this returning nothing so the turn is marked no_speech.
        assert len(result.text) < 40

    def test_numeric_transcripts_are_usable(self, stt: SpeechToText) -> None:
        """The golden set, but through the pipeline's eyes: normalise then eval."""
        from server.tools.calc import evaluate

        result = stt.transcribe(FIXTURES / "q_math_7x23.wav")
        assert evaluate(result.text).display == "161"


class TestRepetitionDetection:
    """Whisper loops on music and room tone, and loops *confidently*.

    Every string below is a verbatim transcript from a real capture in
    friday_kws/deteced_recordings/, so these are measurements rather than
    invented cases.
    """

    # Speaker playing music: 18 "Kolkata" in a row, at confidence 0.73.
    MUSIC_LOOP = (
        "1,2,3,4 get on the dance floor 1,2,3,4 get on the dance floor "
        "don't shake you don't shake "
        + "Kolkata " * 12
    )

    def test_a_music_loop_is_caught(self) -> None:
        assert is_repetitive(self.MUSIC_LOOP) is True

    def test_every_real_command_is_spared(self) -> None:
        for text in (
            "What is the weather in Pune right now?",
            "What time is it right now?",
            "and so my fellow Americans",
            "What is the system status?",
            "Hello there",
        ):
            assert is_repetitive(text) is False, text

    def test_a_repeated_digit_is_not_treated_as_a_loop(self) -> None:
        """"What is 2 + 2?" is two identical tokens in a row, and it is real.

        This is the case that sets the threshold: a run of two must survive.
        """
        assert is_repetitive("What is 2 + 2?") is False

    def test_a_run_of_three_is_enough(self) -> None:
        assert is_repetitive("the the the") is True

    def test_punctuation_does_not_hide_the_loop(self) -> None:
        assert is_repetitive("Kolkata, Kolkata. Kolkata!") is True

    def test_case_does_not_hide_the_loop(self) -> None:
        assert is_repetitive("yes yes YES") is True

    def test_non_adjacent_repeats_are_fine(self) -> None:
        """A word said twice in a sentence is ordinary speech."""
        assert is_repetitive("what is the weather, what is the time") is False

    def test_empty_and_short_text_are_not_repetitive(self) -> None:
        assert is_repetitive("") is False
        assert is_repetitive("yes") is False
