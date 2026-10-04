import unittest
from io import StringIO

from agent.approval import ConsoleApprovalProvider
from core.models import ApprovalDecision, ApprovalRequest


class ConsoleApprovalProviderTests(unittest.TestCase):
    def test_approves_explicit_yes_and_shows_exact_command(self):
        output = StringIO()
        provider = ConsoleApprovalProvider(
            input_stream=StringIO("да\n"),
            output_stream=output,
        )

        decision = provider.request(
            ApprovalRequest(
                tool_name="run_tests",
                title="Запустить тесты?",
                fingerprint="abc123",
                command=("uv", "run", "python", "-m", "unittest"),
                working_directory="/workspace",
            )
        )

        self.assertEqual(decision, ApprovalDecision.APPROVED)
        self.assertIn("uv run python -m unittest", output.getvalue())
        self.assertIn("/workspace", output.getvalue())

    def test_denies_empty_answer(self):
        provider = ConsoleApprovalProvider(
            input_stream=StringIO("\n"),
            output_stream=StringIO(),
        )
        request = ApprovalRequest(
            tool_name="run_linter",
            title="Запустить линтер?",
            fingerprint="def456",
        )

        self.assertEqual(
            provider.request(request),
            ApprovalDecision.DENIED,
        )


if __name__ == "__main__":
    unittest.main()
