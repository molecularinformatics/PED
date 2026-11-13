#!/usr/bin/env bash
# Fetch and patch the external model repositories, and place the checkpoints.
#
#   ./setup.sh              # repos + checkpoints
#   ./setup.sh repos        # clone and patch only
#   ./setup.sh checkpoints  # place and verify checkpoints only
#   ./setup.sh verify       # re-check SHAs, patch state and imports
#
# Checkpoints come from $PED_CKPT_SRC (a local directory) when set, otherwise they
# are downloaded from $PED_CKPT_URL. Either way the SHA-256 is verified.

set -euo pipefail
cd "$(dirname "$0")"
HERE="$PWD"

GEODIFF_URL="https://github.com/MinkaiXu/GeoDiff.git"
GEODIFF_BRANCH="pretrain"
GEODIFF_SHA="d76991dd2edfda98631f001e542ecb933d310803"

SDE_URL="https://github.com/chao1224/MoleculeSDE.git"
SDE_BRANCH="main"
SDE_SHA="06f84c873ba8196e32b4fd4f7436c2bbc3511915"

: "${PED_CKPT_SRC:=}"
: "${PED_CKPT_URL:=}"

log() { printf '[setup] %s\n' "$*"; }
die() { printf '[setup] error: %s\n' "$*" >&2; exit 1; }

clone_and_patch() {
  local dir="$1" url="$2" branch="$3" sha="$4" patch="$5"
  if [ ! -d "$dir/.git" ]; then
    log "cloning $url ($branch) into $dir"
    git clone -q --branch "$branch" "$url" "$dir"
  fi
  git -C "$dir" fetch -q origin "$branch"
  git -C "$dir" checkout -q "$sha"
  if git -C "$dir" apply --check "$HERE/$patch" 2>/dev/null; then
    git -C "$dir" apply "$HERE/$patch"
    log "$dir: pinned at ${sha:0:12} and patched"
  elif git -C "$dir" apply --reverse --check "$HERE/$patch" 2>/dev/null; then
    log "$dir: already patched"
  else
    die "$dir: $patch neither applies nor is already applied; the checkout may be dirty"
  fi
}

place_checkpoints() {
  mkdir -p checkpoints/geodiff/model/checkpoints checkpoints/moleculesde
  local targets=(
    "checkpoints/geodiff/model/checkpoints/drugs_default.pt"
    "checkpoints/geodiff/model/drugs_default.yml"
    "checkpoints/moleculesde/model_complete.pth"
  )
  for target in "${targets[@]}"; do
    local name; name="$(basename "$target")"
    [ -f "$target" ] && continue
    if [ -n "$PED_CKPT_SRC" ] && [ -f "$PED_CKPT_SRC/$name" ]; then
      cp "$PED_CKPT_SRC/$name" "$target"
      log "copied $name from \$PED_CKPT_SRC"
    elif [ -n "$PED_CKPT_URL" ]; then
      curl -fsSL "$PED_CKPT_URL/$name" -o "$target"
      log "downloaded $name"
    else
      die "$name is missing; set PED_CKPT_SRC to a local directory or PED_CKPT_URL to the archive"
    fi
  done
}

verify_checksums() {
  [ -f checksums.txt ] || die "checksums.txt not found"
  ( cd checkpoints && sha256sum -c "$HERE/checksums.txt" --quiet ) \
    && log "checkpoint checksums match" \
    || die "checkpoint checksums do NOT match"
}

verify_imports() {
  local py="${PED_PYTHON:-python}"
  PYTHONPATH="$HERE:${PYTHONPATH:-}" "$py" - <<'PY'
from geodiff.utils.chem import BOND_TYPES
from geodiff.models.epsnet import get_model
from geodiff.utils.transforms import Compose, CountNodesPerGraph, AddHigherOrderEdges
from moleculesde.Geom3D.models.molecule_gnn_model import GNN
from moleculesde.Geom3D.models.schnet import SchNet
print("[setup] imports OK")
PY
}

case "${1:-all}" in
  repos)
    clone_and_patch geodiff     "$GEODIFF_URL" "$GEODIFF_BRANCH" "$GEODIFF_SHA" geodiff.patch
    clone_and_patch moleculesde "$SDE_URL"     "$SDE_BRANCH"     "$SDE_SHA"     moleculesde.patch
    ;;
  checkpoints)
    place_checkpoints; verify_checksums
    ;;
  verify)
    verify_checksums; verify_imports
    ;;
  all)
    clone_and_patch geodiff     "$GEODIFF_URL" "$GEODIFF_BRANCH" "$GEODIFF_SHA" geodiff.patch
    clone_and_patch moleculesde "$SDE_URL"     "$SDE_BRANCH"     "$SDE_SHA"     moleculesde.patch
    place_checkpoints; verify_checksums
    ;;
  *) die "unknown stage '${1}'; use repos, checkpoints, verify or all" ;;
esac
