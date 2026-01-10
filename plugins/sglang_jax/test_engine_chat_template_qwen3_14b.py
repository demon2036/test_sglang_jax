"""
Usage:
  python -m unittest plugins.sglang_jax.test_engine_chat_template_qwen3_14b.TestEngineChatTemplateQwen3
"""

import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SGLANG_JAX_PYTHON = PROJECT_ROOT / "sglang-jax" / "python"
if SGLANG_JAX_PYTHON.exists() and str(SGLANG_JAX_PYTHON) not in sys.path:
    sys.path.insert(0, str(SGLANG_JAX_PYTHON))

from plugins.sglang_jax.chat_template_plugin import build_chat_prompt

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.srt.hf_transformers_utils import get_tokenizer


class TestEngineChatTemplateQwen3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = os.environ.get("SGLANG_JAX_MODEL", "Qwen/Qwen3-14B")
        tp_size = int(os.environ.get("SGLANG_JAX_TP_SIZE", "4"))
        mem_fraction = float(os.environ.get("SGLANG_JAX_MEM_FRACTION", "0.8"))

        cls.engine = Engine(
            model_path=cls.model_path,
            trust_remote_code=True,
            tp_size=tp_size,
            device="tpu",
            mem_fraction_static=mem_fraction,
            download_dir="/tmp",
            dtype="bfloat16",
            skip_server_warmup=True,
        )
        cls.tokenizer = get_tokenizer(cls.model_path)

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def test_ask_who_are_you(self):
        messages = [{"role": "user", "content": "你是谁？"}]
        prompt = build_chat_prompt(self.tokenizer, messages)

        sampling_params = {"max_new_tokens": 64, "temperature": 0}
        if self.tokenizer.eos_token_id is not None:
            sampling_params["stop_token_ids"] = [self.tokenizer.eos_token_id]

        output = self.engine.generate(prompt=prompt, sampling_params=sampling_params)
        item = output[0] if isinstance(output, list) else output
        answer = item.get("text", "")
        print(f"Model response: {answer}")
        self.assertTrue(answer.strip())


if __name__ == "__main__":
    unittest.main()

