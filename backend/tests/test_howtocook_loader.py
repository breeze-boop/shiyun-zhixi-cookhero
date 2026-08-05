from pathlib import Path

from scripts.howtocook_loader import HowToCookLoader


def test_loader_rejects_missing_howtocook_dishes_directory(tmp_path) -> None:
    missing = tmp_path / "HowToCook" / "dishes"

    try:
        HowToCookLoader(missing).load()
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("missing HowToCook dishes directory was accepted")


def test_loader_rejects_empty_howtocook_dishes_directory(tmp_path) -> None:
    dishes_dir = tmp_path / "HowToCook" / "dishes"
    dishes_dir.mkdir(parents=True)

    try:
        HowToCookLoader(tmp_path / "HowToCook").load()
    except ValueError as exc:
        assert "no HowToCook recipe documents found" in str(exc)
    else:
        raise AssertionError("empty HowToCook dishes directory was accepted")


def test_loader_reads_only_dishes_and_maps_metadata() -> None:
    docs = HowToCookLoader(Path("../data/sample_recipes/dishes")).load()
    recipes = [doc for doc in docs if not doc.is_dish_index]

    assert {doc.dish_name for doc in recipes} == {"番茄炒蛋", "红烧肉", "紫菜蛋花汤"}
    tomato = next(doc for doc in recipes if doc.dish_name == "番茄炒蛋")
    assert tomato.category == "素菜"
    assert tomato.difficulty == "简单"
    assert tomato.source.endswith("vegetable_dish/番茄炒蛋.md")
    assert any(doc.is_dish_index for doc in docs)



def test_loader_generates_same_ids_from_repo_root_or_dishes_root(tmp_path) -> None:
    recipe = tmp_path / "HowToCook" / "dishes" / "vegetable_dish" / "西红柿炒鸡蛋" / "西红柿炒鸡蛋.md"
    recipe.parent.mkdir(parents=True)
    recipe.write_text("# 西红柿炒鸡蛋\n\n预估烹饪难度：★★\n", encoding="utf-8")

    from_repo_root = [doc for doc in HowToCookLoader(tmp_path / "HowToCook").load() if not doc.is_dish_index]
    from_dishes_root = [
        doc for doc in HowToCookLoader(tmp_path / "HowToCook" / "dishes").load() if not doc.is_dish_index
    ]

    assert len(from_repo_root) == 1
    assert len(from_dishes_root) == 1
    assert from_repo_root[0].source == "vegetable_dish/西红柿炒鸡蛋/西红柿炒鸡蛋.md"
    assert from_repo_root[0].source == from_dishes_root[0].source
    assert from_repo_root[0].doc_id == from_dishes_root[0].doc_id


def test_loader_skips_non_recipe_markdown_outside_known_dish_categories(tmp_path) -> None:
    dishes_dir = tmp_path / "HowToCook" / "dishes"
    dishes_dir.mkdir(parents=True)
    (dishes_dir / "README.md").write_text("# 菜谱目录\n\n这里不是具体菜谱。", encoding="utf-8")
    recipe = dishes_dir / "vegetable_dish" / "西红柿炒鸡蛋" / "西红柿炒鸡蛋.md"
    recipe.parent.mkdir(parents=True)
    recipe.write_text("# 西红柿炒鸡蛋\n\n预估烹饪难度：★★\n\n## 操作\n\n- 炒熟。\n", encoding="utf-8")

    recipes = [
        document
        for document in HowToCookLoader(tmp_path / "HowToCook").load()
        if not document.is_dish_index
    ]

    assert [document.dish_name for document in recipes] == ["西红柿炒鸡蛋"]
    assert recipes[0].category == "素菜"
