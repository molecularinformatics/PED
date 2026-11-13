"""Register the PED and Roshambo2 scoring components into an installed reinvent-scoring.

Copies the component modules into ``reinvent_scoring/scoring/score_components/ped/``
and adds their names to the component enum and the component factory. Idempotent.

    python install.py                 # install into the active environment
    python install.py --check         # report what is and is not registered
    python install.py --uninstall     # remove the components and the registrations
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
PACKAGE = "ped"

COMPONENTS = {
    # enum constant         enum value              module                     class
    "GEODIFF_DISTANCE":   ("geodiff_distance",   "geodiff_distance",      "GeoDiffDistance"),
    "MOLFORMER_DISTANCE": ("molformer_distance", "molformer_distance",    "MolformerDistance"),
    "ROSHAMBO2_SIMILARITY": ("roshambo2_similarity", "roshambo2_similarity", "Roshambo2Similarity"),
}
EXTRA_MODULES = ["geodiff_utils.py"]

ENUM_REL = "scoring/enums/scoring_function_component_enum.py"
FACTORY_REL = "scoring/score_components/score_component_factory.py"
MARK = "# --- PED components (generation/scoring_components/install.py) ---"


def scoring_root() -> pathlib.Path:
    spec = importlib.util.find_spec("reinvent_scoring")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("reinvent_scoring is not importable in this environment")
    return pathlib.Path(spec.submodule_search_locations[0])


def copy_modules(root: pathlib.Path) -> None:
    target = root / "scoring" / "score_components" / PACKAGE
    target.mkdir(parents=True, exist_ok=True)
    names = [f"{m}.py" for _, m, _ in COMPONENTS.values()] + EXTRA_MODULES
    for name in names:
        shutil.copy2(HERE / name, target / name)
    imports = [f"from .{m} import {c}" for _, m, c in COMPONENTS.values()]
    (target / "__init__.py").write_text("\n".join(imports) + "\n")
    print(f"  copied {len(names)} modules into {target}")


def patch_enum(root: pathlib.Path) -> None:
    path = root / ENUM_REL
    text = path.read_text()
    if MARK in text:
        print("  enum already registered"); return
    lines = [MARK] + [f'    {k} = "{v}"' for k, (v, _, _) in COMPONENTS.items()]
    # Append inside the class: the enum file is a flat dataclass of string constants.
    path.write_text(text.rstrip("\n") + "\n" + "\n".join("    " + l if l is lines[0] else l
                                                          for l in lines) + "\n")
    print(f"  registered {len(COMPONENTS)} names in {ENUM_REL}")


def patch_factory(root: pathlib.Path) -> None:
    path = root / FACTORY_REL
    text = path.read_text()
    if MARK in text:
        print("  factory already registered"); return
    classes = ", ".join(c for _, _, c in COMPONENTS.values())
    anchor = "class ScoreComponentFactory"
    text = text.replace(
        anchor,
        f"{MARK}\nfrom reinvent_scoring.scoring.score_components.{PACKAGE} import {classes}\n\n\n{anchor}",
        1,
    )
    entries = "".join(f"            enum.{k}: {c},\n" for k, (_, _, c) in COMPONENTS.items())
    marker = "        }"  # close of the component dictionary
    index = text.index("enum.TANIMOTO_SIMILARITY")
    close = text.index(marker, index)
    text = text[:close] + entries + text[close:]
    path.write_text(text)
    print(f"  registered {len(COMPONENTS)} components in {FACTORY_REL}")


def check(root: pathlib.Path) -> None:
    for rel in (ENUM_REL, FACTORY_REL):
        state = "registered" if MARK in (root / rel).read_text() else "NOT registered"
        print(f"  {rel}: {state}")
    target = root / "scoring" / "score_components" / PACKAGE
    print(f"  {target}: {'present' if target.is_dir() else 'absent'}")
    for key, (value, _, cls) in COMPONENTS.items():
        try:
            from reinvent_scoring.scoring.score_components import score_component_factory  # noqa
            print(f"  {value}: importable")
        except Exception as exc:
            print(f"  {value}: import failed ({exc})")
            break


def uninstall(root: pathlib.Path) -> None:
    shutil.rmtree(root / "scoring" / "score_components" / PACKAGE, ignore_errors=True)
    for rel in (ENUM_REL, FACTORY_REL):
        path = root / rel
        kept = [l for l in path.read_text().split("\n")
                if MARK not in l and not any(f"enum.{k}" in l or f'{k} = "' in l
                                             for k in COMPONENTS)]
        kept = [l for l in kept if f"score_components.{PACKAGE} import" not in l]
        path.write_text("\n".join(kept))
    print("  removed components and registrations")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--uninstall", action="store_true")
    args = ap.parse_args()

    root = scoring_root()
    print(f"reinvent_scoring at {root}")
    if args.check:
        check(root)
    elif args.uninstall:
        uninstall(root)
    else:
        copy_modules(root)
        patch_enum(root)
        patch_factory(root)
        print("done; rerun with --check to confirm")


if __name__ == "__main__":
    main()
