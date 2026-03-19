"""Tests for discover_groups() in geminiocr.cli."""

from pathlib import Path

from geminiocr.cli import discover_groups


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


class TestSubdirectoryLayout:
    def test_discovers_groups_from_subdirs(self, tmp_path):
        _touch(tmp_path / "receipt_1" / "page1.jpg")
        _touch(tmp_path / "receipt_1" / "page2.png")
        _touch(tmp_path / "receipt_2" / "photo.webp")

        groups = discover_groups(tmp_path)

        assert set(groups.keys()) == {"receipt_1", "receipt_2"}
        assert len(groups["receipt_1"]) == 2
        assert len(groups["receipt_2"]) == 1

    def test_ignores_non_image_files_in_subdirs(self, tmp_path):
        _touch(tmp_path / "group" / "image.jpg")
        _touch(tmp_path / "group" / "notes.txt")
        _touch(tmp_path / "group" / "data.csv")

        groups = discover_groups(tmp_path)

        assert len(groups["group"]) == 1
        assert groups["group"][0].name == "image.jpg"

    def test_skips_subdirs_with_no_images(self, tmp_path):
        _touch(tmp_path / "empty_group" / "readme.txt")
        _touch(tmp_path / "valid_group" / "photo.png")

        groups = discover_groups(tmp_path)

        assert "empty_group" not in groups
        assert "valid_group" in groups


class TestFlatLayout:
    def test_groups_by_prefix(self, tmp_path):
        _touch(tmp_path / "receipt_1.jpg")
        _touch(tmp_path / "receipt_2.jpg")
        _touch(tmp_path / "receipt_3.jpg")

        groups = discover_groups(tmp_path)

        # All three share prefix "receipt" (numeric suffix is stripped)
        assert set(groups.keys()) == {"receipt"}
        assert len(groups["receipt"]) == 3

    def test_groups_multiple_images_by_prefix(self, tmp_path):
        _touch(tmp_path / "doc_1.jpg")
        _touch(tmp_path / "doc_2.jpg")
        _touch(tmp_path / "other_1.png")

        groups = discover_groups(tmp_path)

        assert set(groups.keys()) == {"doc", "other"}
        assert len(groups["doc"]) == 2
        assert len(groups["other"]) == 1

    def test_no_numeric_suffix_uses_stem(self, tmp_path):
        _touch(tmp_path / "photo.jpg")
        _touch(tmp_path / "scan.png")

        groups = discover_groups(tmp_path)

        assert set(groups.keys()) == {"photo", "scan"}

    def test_ignores_non_image_files(self, tmp_path):
        _touch(tmp_path / "image_1.jpg")
        _touch(tmp_path / "notes.txt")
        _touch(tmp_path / "data.json")

        groups = discover_groups(tmp_path)

        assert list(groups.keys()) == ["image"]


class TestEdgeCases:
    def test_empty_directory(self, tmp_path):
        assert discover_groups(tmp_path) == {}

    def test_all_image_extensions(self, tmp_path):
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            _touch(tmp_path / f"img{ext}")

        groups = discover_groups(tmp_path)

        # Same stem "img" with no numeric suffix → single group with 4 files
        assert len(groups) == 1
        assert len(groups["img"]) == 4
