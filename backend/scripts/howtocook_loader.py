from __future__ import annotations

import re
from pathlib import Path
from uuid import UUID, uuid5

from app.rag.document import ParsedDocument


DOC_NAMESPACE = UUID("739fb78e-22a4-4aa4-b54a-79f5dffdfcc1")

CATEGORY_MAPPING = {
    "aquatic": "水产",
    "breakfast": "早餐",
    "condiment": "调味品",
    "dessert": "甜品",
    "drink": "饮品",
    "meat_dish": "荤菜",
    "semi-finished": "半成品",
    "soup": "汤品",
    "staple": "主食",
    "vegetable_dish": "素菜",
}


class HowToCookLoader:
    """Load recipe knowledge from HowToCook's dishes directory."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.dishes_dir = self.root / "dishes" if (self.root / "dishes").exists() else self.root

    def load(self) -> list[ParsedDocument]:
        if not self.dishes_dir.exists():
            raise FileNotFoundError(f"HowToCook dishes directory not found: {self.dishes_dir}")
        if not self.dishes_dir.is_dir():
            raise NotADirectoryError(f"HowToCook dishes path is not a directory: {self.dishes_dir}")
        recipe_docs = [
            self._parse_recipe_file(path)
            for path in sorted(self.dishes_dir.rglob("*.md"))
            if self._is_recipe_markdown(path)
        ]
        if not recipe_docs:
            raise ValueError(f"no HowToCook recipe documents found in: {self.dishes_dir}")
        recipe_docs.extend(self._build_index_documents(recipe_docs))
        return recipe_docs

    def _is_recipe_markdown(self, path: Path) -> bool:
        if "template" in path.parts:
            return False
        try:
            relative = path.relative_to(self.dishes_dir)
        except ValueError:
            return False
        return bool(relative.parts) and relative.parts[0] in CATEGORY_MAPPING

    def _parse_recipe_file(self, path: Path) -> ParsedDocument:
        content = path.read_text(encoding="utf-8")
        relative_path = path.relative_to(self.dishes_dir)
        category_key = self._category_key(path)
        dish_name = path.stem
        source = str(relative_path)
        return ParsedDocument(
            doc_id=str(uuid5(DOC_NAMESPACE, source)),
            dish_name=dish_name,
            category=CATEGORY_MAPPING.get(category_key, category_key),
            difficulty=self._extract_difficulty(content),
            content=content,
            source=source,
            data_source="recipes",
            source_type="recipes",
            user_id="GLOBAL",
            is_dish_index=False,
        )

    def _category_key(self, path: Path) -> str:
        try:
            relative = path.relative_to(self.dishes_dir)
            return relative.parts[0]
        except ValueError:
            return path.parent.name

    @staticmethod
    def _extract_difficulty(content: str) -> str:
        match = re.search(r"难度[：:]\s*([★☆]+)", content)
        stars = match.group(1).count("★") if match else content.count("★")
        if stars <= 2:
            return "简单"
        if stars >= 4:
            return "困难"
        return "普通"

    def _build_index_documents(self, documents: list[ParsedDocument]) -> list[ParsedDocument]:
        grouped: dict[tuple[str, str], list[str]] = {}
        for document in documents:
            grouped.setdefault((document.category, document.difficulty), []).append(
                document.dish_name
            )

        indexes: list[ParsedDocument] = []
        for (category, difficulty), dish_names in sorted(grouped.items()):
            title = f"{category}{difficulty}菜谱索引"
            content = (
                f"# {title}\n\n"
                f"以下菜谱属于{category}分类，难度为{difficulty}：\n\n"
                + "\n".join(f"- {name}" for name in sorted(dish_names))
            )
            source = f"indexes/{category}-{difficulty}.md"
            indexes.append(
                ParsedDocument(
                    doc_id=str(uuid5(DOC_NAMESPACE, source)),
                    dish_name=title,
                    category=category,
                    difficulty=difficulty,
                    content=content,
                    source=source,
                    data_source="recipes",
                    source_type="recipes",
                    user_id="GLOBAL",
                    is_dish_index=True,
                )
            )
        return indexes

