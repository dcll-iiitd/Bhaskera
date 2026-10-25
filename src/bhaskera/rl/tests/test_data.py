from __future__ import annotations

import json

import pytest

from bhaskera.rl.data import Example, PromptSampler, format_prompt, load_examples


def _write(tmp_path, rows):
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps(r) if not isinstance(r, str) else r for r in rows) + "\n")
    return str(p)


class TestLoad:
    def test_loads_and_limits(self, tmp_path):
        path = _write(tmp_path, [{"prompt": f"q{i}", "answer": str(i)} for i in range(5)])
        assert [e.answer for e in load_examples(path)] == ["0", "1", "2", "3", "4"]
        assert len(load_examples(path, limit=2)) == 2

    def test_numeric_answers_become_strings(self, tmp_path):
        assert load_examples(_write(tmp_path, [{"prompt": "q", "answer": 7}]))[0].answer == "7"

    def test_reports_file_and_line_on_bad_row(self, tmp_path):
        path = _write(tmp_path, [{"prompt": "q", "answer": "1"}, {"prompt": "no answer"}])
        with pytest.raises(ValueError, match=r":2:"):
            load_examples(path)

    def test_invalid_json_and_empty_file(self, tmp_path):
        with pytest.raises(ValueError):
            load_examples(_write(tmp_path, ["{not json"]))
        with pytest.raises(ValueError, match="no examples"):
            load_examples(_write(tmp_path, []))


class TestFormatPrompt:
    def test_plain_text_without_chat_template(self, tokenizer):
        assert format_prompt(tokenizer, "Q 1", "SYS") == "SYS\n\nQ 1\n"
        assert format_prompt(tokenizer, "Q 1") == "Q 1\n"

    def test_uses_chat_template_when_present(self, tokenizer):
        tokenizer.chat_template = (
            "{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}{% endfor %}"
            "{% if add_generation_prompt %}<assistant>{% endif %}"
        )
        try:
            rendered = format_prompt(tokenizer, "hi", "be brief")
            assert rendered == "<system>be brief<user>hi<assistant>"
        finally:
            tokenizer.chat_template = None


class TestPromptSampler:
    ex = [Example(f"q{i}", str(i)) for i in range(5)]

    def test_each_pass_covers_every_example_once(self):
        s = PromptSampler(self.ex, seed=0)
        assert sorted(e.prompt for e in s.next_batch(5)) == sorted(e.prompt for e in self.ex)
        assert s.epoch == 0          # the last example handed out is still from pass 0
        s.next_batch(1)
        assert s.epoch == 1

    def test_deterministic_for_a_seed_and_different_across_seeds(self):
        a = [e.prompt for e in PromptSampler(self.ex, 1).next_batch(12)]
        assert a == [e.prompt for e in PromptSampler(self.ex, 1).next_batch(12)]
        assert a != [e.prompt for e in PromptSampler(self.ex, 2).next_batch(12)]

    def test_passes_are_reshuffled(self):
        s = PromptSampler(self.ex, 0)
        first, second = s.next_batch(5), s.next_batch(5)
        assert [e.prompt for e in first] != [e.prompt for e in second]

    def test_resuming_from_position_continues_the_same_stream(self):
        full = [e.prompt for e in PromptSampler(self.ex, 3).next_batch(14)]
        resumed = PromptSampler(self.ex, 3, position=6)
        assert [e.prompt for e in resumed.next_batch(8)] == full[6:]

    def test_batches_wrap_across_epoch_boundaries(self):
        s = PromptSampler(self.ex, 0)
        assert len(s.next_batch(3)) == 3 and len(s.next_batch(4)) == 4 and s.position == 7
