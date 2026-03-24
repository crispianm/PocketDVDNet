from __future__ import annotations

from pathlib import Path

from models import PUBLISHED_POCKETDVDNET_RECIPE, PocketDVDNet, PocketDVDNetRecipe

from .config import dump_json, load_json
from .runtime import count_parameters, model_sparsity


def build_recipe_artifact(source_model=None, source_checkpoint: str | None = None) -> dict:
    recipe = PUBLISHED_POCKETDVDNET_RECIPE
    student = PocketDVDNet(recipe)
    artifact = {
        "recipe": recipe.to_dict(),
        "model_name": "PocketDVDNet",
        "parameter_count": count_parameters(student),
        "derivation": "published_pocketdvdnet_recipe",
    }
    if source_checkpoint:
        artifact["source_checkpoint"] = source_checkpoint
    if source_model is not None:
        artifact["source_sparsity"] = model_sparsity(source_model)
    return artifact


def save_recipe_artifact(
    path: str | Path, source_model=None, source_checkpoint: str | None = None
) -> dict:
    artifact = build_recipe_artifact(source_model=source_model, source_checkpoint=source_checkpoint)
    dump_json(path, artifact)
    return artifact


def load_recipe(path: str | Path) -> dict:
    payload = load_json(path)
    if "recipe" in payload:
        return payload["recipe"]
    return PocketDVDNetRecipe.from_dict(payload).to_dict()
