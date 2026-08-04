from collections.abc import Iterator
import hashlib
import os
from pathlib import Path
import stat
import tempfile


DEFAULT_FORBIDDEN_NAMES = {
    ".env",
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
}


class WorkspaceViolation(Exception):
    pass


class WorkspaceConflict(WorkspaceViolation):
    """The requested change conflicts with the current workspace state."""


class Workspace:
    def __init__(
        self,
        root: str | Path,
        forbidden_names: set[str] | None = None,
        max_file_size: int = 1_000_000,
    ):
        self._root = Path(root).expanduser().resolve()
        self._forbidden_names = (
            set(forbidden_names)
            if forbidden_names is not None
            else set(DEFAULT_FORBIDDEN_NAMES)
        )
        self._max_file_size = max_file_size

        if not self._root.exists():
            raise WorkspaceViolation(
                f"Workspace does not exist: {self._root}"
            )
        if not self._root.is_dir():
            raise WorkspaceViolation(
                f"Workspace is not a directory: {self._root}"
            )
        if max_file_size <= 0:
            raise ValueError("max_file_size must be greater than zero")

    @property
    def root(self) -> Path:
        return self._root

    def resolve(self, user_path: str | Path) -> Path:
        raw_path = Path(user_path).expanduser()
        candidate = (
            raw_path.resolve()
            if raw_path.is_absolute()
            else (self._root / raw_path).resolve()
        )

        if not candidate.is_relative_to(self._root):
            raise WorkspaceViolation("Path is outside workspace")

        relative = candidate.relative_to(self._root)
        if any(
            part in self._forbidden_names
            for part in relative.parts
        ):
            raise WorkspaceViolation("Path is forbidden")

        return candidate

    def relative_path(self, path: Path) -> str:
        return path.relative_to(self._root).as_posix() or "."

    def _resolve_write_target(self, user_path: str | Path) -> Path:
        """Resolve a write target and reject every symlink in its path."""
        raw_path = Path(user_path).expanduser()
        unresolved = (
            raw_path
            if raw_path.is_absolute()
            else self._root / raw_path
        )
        lexical_path = Path(os.path.abspath(unresolved))

        if not lexical_path.is_relative_to(self._root):
            raise WorkspaceViolation("Path is outside workspace")

        current = self._root
        for part in lexical_path.relative_to(self._root).parts:
            current /= part
            if current.is_symlink():
                raise WorkspaceViolation("Write path must not contain symlinks")

        return self.resolve(user_path)

    def read_text(
        self,
        user_path: str,
        *,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> dict:
        path = self.resolve(user_path)
        self._validate_readable_file(path)

        if start_line < 1:
            raise WorkspaceViolation("start_line must be at least 1")
        if end_line is not None and end_line < start_line:
            raise WorkspaceViolation(
                "end_line must be greater than or equal to start_line"
            )

        try:
            raw_bytes = path.read_bytes()
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspaceViolation(
                "File is not valid UTF-8 text"
            ) from exc

        lines = text.splitlines(keepends=True)
        selected = lines[start_line - 1:end_line]
        actual_end = min(
            end_line or len(lines),
            len(lines),
        )

        return {
            "path": self.relative_path(path),
            "content": "".join(selected),
            "start_line": start_line,
            "end_line": actual_end,
            "total_lines": len(lines),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "truncated": end_line is not None and end_line < len(lines),
        }

    def create_text(
        self,
        user_path: str,
        content: str,
    ) -> dict:
        path = self._resolve_write_target(user_path)
        encoded_content = self._encode_writable_text(content)

        if path.exists():
            raise WorkspaceConflict("File already exists")
        if not path.parent.exists():
            raise WorkspaceViolation("Parent directory does not exist")
        if not path.parent.is_dir():
            raise WorkspaceViolation("Parent path is not a directory")

        created_identity: tuple[int, int] | None = None
        try:
            with path.open("x", encoding="utf-8", newline="") as file:
                file_status = os.fstat(file.fileno())
                created_identity = (file_status.st_dev, file_status.st_ino)
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
        except FileExistsError as exc:
            raise WorkspaceConflict("File already exists") from exc
        except OSError as exc:
            if created_identity is not None:
                # Remove a partially written file instead of exposing it as
                # the successful result of this operation. Check its identity
                # first so a concurrent replacement is never removed.
                try:
                    current_status = path.lstat()
                    current_identity = (
                        current_status.st_dev,
                        current_status.st_ino,
                    )
                    if current_identity == created_identity:
                        path.unlink()
                except OSError:
                    pass
            raise WorkspaceViolation("Cannot create file") from exc

        return {
            "path": self.relative_path(path),
            "created": True,
            "bytes_written": len(encoded_content),
            "sha256": hashlib.sha256(encoded_content).hexdigest(),
        }

    def edit_text(
        self,
        user_path: str,
        *,
        old_text: str,
        new_text: str,
        expected_sha256: str,
        expected_replacements: int = 1,
    ) -> dict:
        path = self._resolve_write_target(user_path)
        self._validate_readable_file(path)
        self._validate_sha256(expected_sha256)

        if not old_text:
            raise WorkspaceViolation("old_text must not be empty")
        if not 1 <= expected_replacements <= 1_000:
            raise WorkspaceViolation(
                "expected_replacements must be between 1 and 1000"
            )

        try:
            original_bytes = path.read_bytes()
            original_text = original_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspaceViolation(
                "File is not valid UTF-8 text"
            ) from exc
        except OSError as exc:
            raise WorkspaceViolation("Cannot read file") from exc

        actual_sha256 = hashlib.sha256(original_bytes).hexdigest()
        if actual_sha256 != expected_sha256.lower():
            raise WorkspaceConflict(
                "File changed since it was read; read it again before editing"
            )

        replacements = original_text.count(old_text)
        if replacements != expected_replacements:
            raise WorkspaceConflict(
                "old_text occurrence count does not match "
                f"expected_replacements: expected {expected_replacements}, "
                f"found {replacements}"
            )

        updated_text = original_text.replace(
            old_text,
            new_text,
            expected_replacements,
        )
        updated_bytes = self._encode_writable_text(updated_text)
        self._atomic_replace(path, updated_text, original_bytes)

        return {
            "path": self.relative_path(path),
            "edited": True,
            "replacements": replacements,
            "bytes_written": len(updated_bytes),
            "sha256": hashlib.sha256(updated_bytes).hexdigest(),
        }

    def list_entries(
        self,
        user_path: str = ".",
        *,
        max_depth: int = 3,
        max_entries: int = 500,
    ) -> dict:
        root = self.resolve(user_path)
        if not root.is_dir():
            raise WorkspaceViolation("Requested path is not a directory")
        if not 0 <= max_depth <= 10:
            raise WorkspaceViolation(
                "max_depth must be between 0 and 10"
            )
        if not 1 <= max_entries <= 5_000:
            raise WorkspaceViolation(
                "max_entries must be between 1 and 5000"
            )

        entries: list[dict[str, str]] = []
        truncated = False

        def walk(directory: Path, depth: int) -> None:
            nonlocal truncated
            if truncated:
                return

            try:
                children = sorted(
                    directory.iterdir(),
                    key=lambda item: item.name.casefold(),
                )
            except OSError as exc:
                raise WorkspaceViolation(
                    f"Cannot list directory: {self.relative_path(directory)}"
                ) from exc

            for item in children:
                if item.name in self._forbidden_names:
                    continue
                if len(entries) >= max_entries:
                    truncated = True
                    return

                try:
                    safe_item = self.resolve(item)
                except WorkspaceViolation:
                    continue

                if item.is_symlink():
                    item_type = "symlink"
                elif safe_item.is_dir():
                    item_type = "directory"
                else:
                    item_type = "file"

                entries.append(
                    {
                        "path": self.relative_path(safe_item),
                        "type": item_type,
                    }
                )

                if (
                    item_type == "directory"
                    and depth < max_depth
                ):
                    walk(safe_item, depth + 1)

        walk(root, 0)
        return {
            "root": self.relative_path(root),
            "entries": entries,
            "truncated": truncated,
        }

    def search_text(
        self,
        query: str,
        *,
        user_path: str = ".",
        glob: str = "*.py",
        case_sensitive: bool = False,
        max_results: int = 50,
    ) -> dict:
        if not query:
            raise WorkspaceViolation("Search query must not be empty")
        if not 1 <= max_results <= 500:
            raise WorkspaceViolation(
                "max_results must be between 1 and 500"
            )

        matches: list[dict[str, object]] = []
        needle = query if case_sensitive else query.casefold()

        for path in self._iter_files(user_path, glob):
            try:
                self._validate_readable_file(path)
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, WorkspaceViolation, OSError):
                continue

            for line_number, line in enumerate(
                text.splitlines(),
                start=1,
            ):
                haystack = line if case_sensitive else line.casefold()
                if needle not in haystack:
                    continue
                matches.append(
                    {
                        "path": self.relative_path(path),
                        "line": line_number,
                        "text": line[:500],
                    }
                )
                if len(matches) >= max_results:
                    return {
                        "query": query,
                        "matches": matches,
                        "truncated": True,
                    }

        return {
            "query": query,
            "matches": matches,
            "truncated": False,
        }

    def _iter_files(
        self,
        user_path: str,
        glob: str,
    ) -> Iterator[Path]:
        root = self.resolve(user_path)
        if root.is_file():
            yield root
            return
        if not root.is_dir():
            raise WorkspaceViolation("Search path does not exist")

        for directory, directory_names, file_names in os.walk(
            root,
            topdown=True,
            followlinks=False,
        ):
            current_directory = Path(directory)
            directory_names[:] = [
                name
                for name in directory_names
                if (
                    name not in self._forbidden_names
                    and not (current_directory / name).is_symlink()
                )
            ]

            for file_name in sorted(
                file_names,
                key=str.casefold,
            ):
                if file_name in self._forbidden_names:
                    continue
                candidate = current_directory / file_name
                relative = candidate.relative_to(root)
                if not relative.match(glob):
                    continue
                if candidate.is_symlink():
                    continue
                try:
                    yield self.resolve(candidate)
                except WorkspaceViolation:
                    continue

    def _validate_readable_file(self, path: Path) -> None:
        if not path.exists():
            raise WorkspaceViolation("File does not exist")
        if not path.is_file():
            raise WorkspaceViolation("Requested path is not a file")
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise WorkspaceViolation("Cannot inspect file") from exc
        if size > self._max_file_size:
            raise WorkspaceViolation(
                f"File exceeds the {self._max_file_size} byte limit"
            )

    def _encode_writable_text(self, content: str) -> bytes:
        encoded_content = content.encode("utf-8")
        if len(encoded_content) > self._max_file_size:
            raise WorkspaceViolation(
                f"Content exceeds the {self._max_file_size} byte limit"
            )
        return encoded_content

    @staticmethod
    def _validate_sha256(value: str) -> None:
        hexadecimal_characters = "0123456789abcdefABCDEF"
        if len(value) != 64 or any(
            character not in hexadecimal_characters
            for character in value
        ):
            raise WorkspaceViolation(
                "expected_sha256 must be a SHA-256 hash"
            )

    def _atomic_replace(
        self,
        path: Path,
        content: str,
        expected_bytes: bytes,
    ) -> None:
        temporary_path: Path | None = None
        try:
            file_descriptor, raw_temporary_path = tempfile.mkstemp(
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
            )
            temporary_path = Path(raw_temporary_path)
            with os.fdopen(
                file_descriptor,
                "w",
                encoding="utf-8",
                newline="",
            ) as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())

            temporary_path.chmod(stat.S_IMODE(path.stat().st_mode))

            # Recheck immediately before replacement to reduce the chance of
            # overwriting a concurrent update made after the first read.
            if path.read_bytes() != expected_bytes:
                raise WorkspaceConflict(
                    "File changed while the edit was being prepared"
                )

            os.replace(temporary_path, path)
            temporary_path = None
        except WorkspaceConflict:
            raise
        except OSError as exc:
            raise WorkspaceViolation("Cannot edit file") from exc
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass


# Compatibility alias for the earlier prototype spelling.
WorkSpace = Workspace
