import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from finetuning.rlbr_utils import (
    compute_corpus_error_rates,
    compute_rlbr_reward,
    resolve_evaluation_prompt,
)
from local.prepare_rlbr_librispeech import (
    main as prepare_rlbr_main,
    make_output_row_from_list,
    read_biasing_reference,
)


class RLBRRewardTest(unittest.TestCase):
    def test_paper_figure_word_reward(self):
        result = compute_rlbr_reward(
            reference="this is a *red* *apple*",
            hypothesis="this is an red *maple*",
            bias_words=["red", "apple"],
            bias_weight=5,
            edit_level="word",
        )

        self.assertEqual(result.edit_distance, 3)
        self.assertEqual(result.bias_edit_distance, 2)
        self.assertEqual(result.reward, -13.0)

    def test_gold_reference_has_zero_reward(self):
        reference = "language English<asr_text>this is a *red* *apple*"
        result = compute_rlbr_reward(
            reference=reference,
            hypothesis=reference,
            bias_words=["red", "apple"],
            bias_weight=5,
            edit_level="char",
        )

        self.assertEqual(result.reward, 0.0)


class ContextualMetricsTest(unittest.TestCase):
    def test_errors_are_partitioned_between_bias_and_unbiased_words(self):
        metrics = compute_corpus_error_rates(
            [
                {
                    "reference": "red apple now",
                    "hypothesis": "read maple now please",
                    "bias_words": ["red", "apple"],
                }
            ]
        )

        self.assertEqual(metrics["bias_errors"], 2)
        self.assertEqual(metrics["unbiased_errors"], 1)
        self.assertEqual(metrics["wer"], 100.0)

    def test_inserted_bias_word_is_a_bias_error(self):
        metrics = compute_corpus_error_rates(
            [
                {
                    "reference": "say quokka now",
                    "hypothesis": "say quokka quokka now",
                    "bias_words": ["quokka"],
                }
            ]
        )

        self.assertEqual(metrics["insertions"], 1)
        self.assertEqual(metrics["bias_errors"], 1)
        self.assertEqual(metrics["unbiased_errors"], 0)


class EvaluationPromptTest(unittest.TestCase):
    def test_direct_asr_discards_contextual_prompt(self):
        row = {"prompt": "do not leak this", "bias_list": ["quokka"]}

        self.assertEqual(resolve_evaluation_prompt(row, "none"), "")

    def test_local_biasing_builds_prompt_from_word_list(self):
        row = {"bias_list": ["quokka", "zymurgy"]}

        self.assertEqual(
            resolve_evaluation_prompt(row, "biasing"),
            "Transcribe the audio clip into text with extra attention to the "
            "following words: [*quokka*, *zymurgy*]",
        )

    def test_local_biasing_prefers_explicit_prompt(self):
        row = {"prompt": "custom local context", "bias_list": ["ignored"]}

        self.assertEqual(
            resolve_evaluation_prompt(row, "biasing"), "custom local context"
        )


class Rare5kDataTest(unittest.TestCase):
    @staticmethod
    def _write_subset(root, subset, text_id, reference):
        directory = root / subset / "1" / "2"
        directory.mkdir(parents=True)
        (directory / "1-2.trans.txt").write_text(
            f"{text_id} {reference}\n", encoding="utf-8"
        )
        (directory / f"{text_id}.flac").touch()

    def test_reads_official_four_column_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test-clean.biasing_2.tsv"
            path.write_text(
                "utt-1\ta rare word\t"
                + json.dumps(["rare"])
                + "\t"
                + json.dumps(["rare", "distractor"])
                + "\n",
                encoding="utf-8",
            )

            rows = read_biasing_reference(path)

        self.assertEqual(rows[0]["bias_words"], ["rare"])
        self.assertEqual(rows[0]["bias_list"], ["rare", "distractor"])

    def test_local_list_contains_only_positive_words(self):
        row = make_output_row_from_list(
            source={
                "text_id": "utt-1",
                "audio": "/tmp/utt-1.flac",
                "reference": "A rare word",
                "subset": "test-clean",
            },
            positives=["rare"],
            bias_list=["rare"],
            language="English",
            condition="local",
        )

        self.assertEqual(row["bias_list"], ["rare"])
        self.assertEqual(row["distractor_count"], 0)
        self.assertEqual(row["marked_reference"], "A *rare* word")

    def test_prepares_local_and_fixed_size_global_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "LibriSpeech"
            self._write_subset(corpus, "train-clean-100", "1-2-0001", "A rare word")
            self._write_subset(corpus, "dev-clean", "1-2-0002", "A rare word")
            self._write_subset(corpus, "test-clean", "1-2-0003", "A rare word")

            benchmark = root / "is21_deep_bias"
            (benchmark / "words").mkdir(parents=True)
            (benchmark / "ref").mkdir()
            (benchmark / "words/common_words_5k.txt").write_text(
                "a\nword\n", encoding="utf-8"
            )
            (benchmark / "words/all_rare_words.txt").write_text(
                "rare\ndistractor\nextra\n", encoding="utf-8"
            )
            (benchmark / "words/all_words.count.txt").write_text(
                "a\t10\nword\t8\nrare\t2\ndistractor\t0\nextra\t0\n",
                encoding="utf-8",
            )
            (benchmark / "ref/test-clean.biasing_2.tsv").write_text(
                "1-2-0003\ta rare word\t[\"rare\"]\t"
                "[\"rare\", \"distractor\"]\n",
                encoding="utf-8",
            )
            output = root / "output"
            argv = [
                "prepare_rlbr_librispeech.py",
                "--librispeech_root",
                str(corpus),
                "--output_dir",
                str(output),
                "--biasing_benchmark_root",
                str(benchmark),
                "--train_subsets",
                "train-clean-100",
                "--dev_subsets",
                "dev-clean",
                "--test_subsets",
                "test-clean",
                "--train_num_positive",
                "1",
                "--train_distractors",
                "1",
                "--eval_bias_sizes",
                "2",
            ]

            with patch("sys.argv", argv):
                prepare_rlbr_main()

            local_row = json.loads(
                (output / "test_clean_local.jsonl").read_text(encoding="utf-8")
            )
            global_row = json.loads(
                (output / "test_clean_n2.jsonl").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(local_row["bias_list"], ["rare"])
        self.assertEqual(global_row["bias_list"], ["rare", "distractor"])
        self.assertEqual(global_row["bias_list_size"], 2)
        self.assertEqual(manifest["biasing_list_source"], "official_is21_deep_bias")


if __name__ == "__main__":
    unittest.main()
