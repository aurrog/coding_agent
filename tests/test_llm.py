from types import SimpleNamespace
import unittest

from agent.llm import OpenAICompatibleLLM


class LLMNormalizationTests(unittest.TestCase):
    def test_omits_tools_field_for_finalization_request(self):
        class FakeCompletions:
            def __init__(self):
                self.request = None

            def create(self, **request):
                self.request = request
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="Final answer",
                                tool_calls=[],
                            ),
                            finish_reason="stop",
                        )
                    ],
                    usage=None,
                )

        completions = FakeCompletions()
        llm = OpenAICompatibleLLM.__new__(OpenAICompatibleLLM)
        llm._model = "test-model"
        llm._client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )

        response = llm.complete(messages=[], tools=[])

        self.assertEqual(response.content, "Final answer")
        self.assertNotIn("tools", completions.request)

    def test_normalizes_provider_token_usage(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Done",
                        tool_calls=[],
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=120,
                completion_tokens=30,
                total_tokens=150,
                prompt_tokens_details=SimpleNamespace(
                    cached_tokens=40,
                ),
                completion_tokens_details=SimpleNamespace(
                    reasoning_tokens=12,
                ),
            ),
        )

        normalized = OpenAICompatibleLLM._normalize_response(response)

        self.assertTrue(normalized.usage.reported)
        self.assertEqual(normalized.usage.prompt_tokens, 120)
        self.assertEqual(normalized.usage.completion_tokens, 30)
        self.assertEqual(normalized.usage.total_tokens, 150)
        self.assertEqual(normalized.usage.cached_tokens, 40)
        self.assertEqual(normalized.usage.reasoning_tokens, 12)

    def test_usage_is_marked_unavailable_when_provider_omits_it(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Done",
                        tool_calls=[],
                    ),
                    finish_reason="stop",
                )
            ],
        )

        normalized = OpenAICompatibleLLM._normalize_response(response)

        self.assertFalse(normalized.usage.reported)
        self.assertEqual(normalized.usage.total_tokens, 0)


if __name__ == "__main__":
    unittest.main()
