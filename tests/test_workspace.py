import tempfile
from pathlib import Path
import unittest

from security.workspace import (
    Workspace,
    WorkspaceConflict,
    WorkspaceViolation,
)


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_directory.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "app.py").write_text(
            "def health():\n    return 'ok'\n",
            encoding="utf-8",
        )
        (self.root / ".env").write_text(
            "SECRET=value",
            encoding="utf-8",
        )
        self.workspace = Workspace(self.root)

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_rejects_path_outside_workspace(self):
        with self.assertRaises(WorkspaceViolation):
            self.workspace.resolve("../outside.txt")

    def test_rejects_absolute_path_outside_workspace(self):
        outside = self.root.parent / "outside.txt"
        with self.assertRaises(WorkspaceViolation):
            self.workspace.resolve(outside)

    def test_rejects_symlink_outside_workspace(self):
        outside = self.root.parent / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        link = self.root / "outside-link"
        try:
            link.symlink_to(outside)
            with self.assertRaises(WorkspaceViolation):
                self.workspace.resolve("outside-link")
        finally:
            outside.unlink(missing_ok=True)

    def test_rejects_forbidden_file(self):
        with self.assertRaises(WorkspaceViolation):
            self.workspace.read_text(".env")

    def test_resolves_only_directories_inside_workspace(self):
        self.assertEqual(
            self.workspace.resolve_directory("src"),
            self.root.resolve() / "src",
        )
        with self.assertRaises(WorkspaceViolation):
            self.workspace.resolve_directory("src/app.py")

    def test_reads_line_range_and_hash(self):
        result = self.workspace.read_text(
            "src/app.py",
            start_line=1,
            end_line=1,
        )

        self.assertEqual(result["content"], "def health():\n")
        self.assertEqual(result["total_lines"], 2)
        self.assertEqual(len(result["sha256"]), 64)
        self.assertTrue(result["truncated"])

    def test_creates_new_text_file(self):
        result = self.workspace.create_text(
            "src/new.py",
            "value = 'new'\n",
        )

        self.assertTrue(result["created"])
        self.assertEqual(result["path"], "src/new.py")
        self.assertEqual(
            (self.root / "src" / "new.py").read_text(encoding="utf-8"),
            "value = 'new'\n",
        )
        self.assertEqual(len(result["sha256"]), 64)

    def test_create_never_overwrites_existing_file(self):
        with self.assertRaises(WorkspaceConflict):
            self.workspace.create_text("src/app.py", "overwritten\n")

        self.assertIn(
            "def health",
            (self.root / "src" / "app.py").read_text(encoding="utf-8"),
        )

    def test_create_requires_existing_parent_directory(self):
        with self.assertRaises(WorkspaceViolation):
            self.workspace.create_text("missing/new.py", "content\n")

        self.assertFalse((self.root / "missing").exists())

    def test_create_rejects_forbidden_path(self):
        with self.assertRaises(WorkspaceViolation):
            self.workspace.create_text(".git/new.py", "content\n")

    def test_create_rejects_content_over_size_limit(self):
        workspace = Workspace(self.root, max_file_size=5)

        with self.assertRaises(WorkspaceViolation):
            workspace.create_text("src/large.py", "123456")

        self.assertFalse((self.root / "src" / "large.py").exists())

    def test_edits_exact_text_using_latest_hash(self):
        current = self.workspace.read_text("src/app.py")

        result = self.workspace.edit_text(
            "src/app.py",
            old_text="return 'ok'",
            new_text="return 'healthy'",
            expected_sha256=current["sha256"],
        )

        self.assertTrue(result["edited"])
        self.assertEqual(result["replacements"], 1)
        self.assertIn(
            "return 'healthy'",
            (self.root / "src" / "app.py").read_text(encoding="utf-8"),
        )
        self.assertNotEqual(result["sha256"], current["sha256"])

    def test_edit_rejects_stale_hash_without_changing_file(self):
        original = (self.root / "src" / "app.py").read_text(
            encoding="utf-8"
        )

        with self.assertRaises(WorkspaceConflict):
            self.workspace.edit_text(
                "src/app.py",
                old_text="return 'ok'",
                new_text="return 'stale'",
                expected_sha256="0" * 64,
            )

        self.assertEqual(
            (self.root / "src" / "app.py").read_text(encoding="utf-8"),
            original,
        )

    def test_edit_rejects_unexpected_occurrence_count(self):
        current = self.workspace.read_text("src/app.py")

        with self.assertRaises(WorkspaceConflict):
            self.workspace.edit_text(
                "src/app.py",
                old_text="not present",
                new_text="replacement",
                expected_sha256=current["sha256"],
            )

    def test_applies_multiple_edits_atomically(self):
        current = self.workspace.read_text("src/app.py")

        result = self.workspace.edit_text_many(
            "src/app.py",
            edits=[
                {
                    "old_text": "health",
                    "new_text": "readiness",
                },
                {
                    "old_text": "return 'ok'",
                    "new_text": "return 'ready'",
                },
            ],
            expected_sha256=current["sha256"],
        )

        self.assertEqual(result["edits_applied"], 2)
        self.assertEqual(result["replacements"], 2)
        self.assertEqual(
            (self.root / "src" / "app.py").read_text(encoding="utf-8"),
            "def readiness():\n    return 'ready'\n",
        )

    def test_failed_batch_edit_writes_nothing(self):
        current = self.workspace.read_text("src/app.py")
        original = current["content"]

        with self.assertRaises(WorkspaceConflict):
            self.workspace.edit_text_many(
                "src/app.py",
                edits=[
                    {"old_text": "health", "new_text": "readiness"},
                    {"old_text": "missing", "new_text": "value"},
                ],
                expected_sha256=current["sha256"],
            )

        self.assertEqual(
            (self.root / "src" / "app.py").read_text(encoding="utf-8"),
            original,
        )

    def test_rewrites_existing_file_using_latest_hash(self):
        current = self.workspace.read_text("src/app.py")

        result = self.workspace.write_text(
            "src/app.py",
            content="print('rewritten')\n",
            expected_sha256=current["sha256"],
        )

        self.assertTrue(result["written"])
        self.assertEqual(
            (self.root / "src" / "app.py").read_text(encoding="utf-8"),
            "print('rewritten')\n",
        )

    def test_writes_reject_symlink_targets(self):
        link = self.root / "app-link.py"
        link.symlink_to(self.root / "src" / "app.py")
        current = self.workspace.read_text("src/app.py")

        with self.assertRaises(WorkspaceViolation):
            self.workspace.edit_text(
                "app-link.py",
                old_text="return 'ok'",
                new_text="return 'unsafe'",
                expected_sha256=current["sha256"],
            )

    def test_lists_entries_with_relative_paths(self):
        result = self.workspace.list_entries(".", max_depth=2)
        paths = {
            entry["path"]
            for entry in result["entries"]
        }

        self.assertIn("src", paths)
        self.assertIn("src/app.py", paths)
        self.assertNotIn(".env", paths)

    def test_searches_text(self):
        result = self.workspace.search_text(
            "health",
            glob="*.py",
        )

        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(
            result["matches"][0]["path"],
            "src/app.py",
        )

    def test_search_skips_forbidden_directories(self):
        forbidden_directory = self.root / ".git"
        forbidden_directory.mkdir()
        (forbidden_directory / "secret.py").write_text(
            "health = 'secret'\n",
            encoding="utf-8",
        )

        result = self.workspace.search_text(
            "health",
            glob="*.py",
        )

        self.assertEqual(len(result["matches"]), 1)


if __name__ == "__main__":
    unittest.main()
