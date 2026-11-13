#!/usr/bin/env bash
# Fetch the generation stack: REINVENT 3.2, reinvent-scoring, Roshambo2 and the
# public prior checkpoints, each pinned to a recorded commit.
#
#   ./setup.sh            # everything
#   ./setup.sh repos      # REINVENT, reinvent-scoring, Roshambo2 (patched)
#   ./setup.sh priors     # the two public priors
#   ./setup.sh verify     # re-check the pins and the patch state
#
# Roshambo2 and reinvent-scoring still need installing into their environments
# afterwards; see ../README.md.

set -euo pipefail
cd "$(dirname "$0")"
HERE="$PWD"

REINVENT_URL="https://github.com/MolecularAI/Reinvent.git"
REINVENT_SHA="b7324d222a49d18b08335a01649abdb0ac66a734"          # tag v3.2

SCORING_URL="https://github.com/MolecularAI/reinvent-scoring.git"
SCORING_SHA="9e405b5adddd5bf58721465dfd7c31cc432649bc"           # branch reinvent.3.2

ROSHAMBO2_URL="https://github.com/molecularinformatics/roshambo2.git"
ROSHAMBO2_SHA="7ff8237c4f21372e8fbfb4e842b1b5eee87a7aeb"

# The two priors are taken from a pinned commit rather than a raw URL, so the
# commit itself is the integrity guarantee.
PRIORS_URL="https://github.com/MolecularAI/ReinventCommunity.git"
PRIORS_SHA="e4c1b19c6d9bf7d74f8b8cffb239ca9cc28b4729"            # branch reinvent.3.0
PRIOR_FILES=("notebooks/models/augmented.prior" "notebooks/models/library_design.prior")

log() { printf '[setup] %s\n' "$*"; }
die() { printf '[setup] error: %s\n' "$*" >&2; exit 1; }

clone_at() {
  local dir="$1" url="$2" sha="$3" patch="${4:-}"
  if [ ! -d "$dir/.git" ]; then
    log "cloning $url into $dir"
    git clone -q "$url" "$dir"
  fi
  git -C "$dir" fetch -q origin
  git -C "$dir" checkout -q "$sha"
  if [ -n "$patch" ]; then
    if git -C "$dir" apply --check "$HERE/$patch" 2>/dev/null; then
      git -C "$dir" apply "$HERE/$patch"
      log "$dir: pinned at ${sha:0:12} and patched"
    elif git -C "$dir" apply --reverse --check "$HERE/$patch" 2>/dev/null; then
      log "$dir: already patched"
    else
      die "$dir: $patch neither applies nor is already applied"
    fi
  else
    log "$dir: pinned at ${sha:0:12}"
  fi
}

fetch_priors() {
  mkdir -p priors
  if [ ! -d priors/.src/.git ]; then
    log "fetching priors from $PRIORS_URL (about 110 MB)"
    git clone -q --filter=blob:none --no-checkout "$PRIORS_URL" priors/.src
  fi
  git -C priors/.src fetch -q origin "$PRIORS_SHA" || git -C priors/.src fetch -q origin
  git -C priors/.src sparse-checkout set --no-cone notebooks/models >/dev/null 2>&1 || true
  git -C priors/.src checkout -q "$PRIORS_SHA"
  for rel in "${PRIOR_FILES[@]}"; do
    cp "priors/.src/$rel" "priors/$(basename "$rel")"
    log "placed priors/$(basename "$rel")"
  done
}

verify() {
  for d in reinvent reinvent-scoring roshambo2; do
    [ -d "$d/.git" ] || die "$d is missing; run ./setup.sh repos"
    printf '[setup] %-18s %s\n' "$d" "$(git -C "$d" rev-parse HEAD)"
  done
  git -C roshambo2 apply --reverse --check "$HERE/roshambo2.patch" 2>/dev/null \
    && log "roshambo2: patch applied" || die "roshambo2: patch is NOT applied"
  for rel in "${PRIOR_FILES[@]}"; do
    f="priors/$(basename "$rel")"
    [ -f "$f" ] || die "$f is missing; run ./setup.sh priors"
    printf '[setup] %-18s %s MB\n' "$(basename "$rel")" \
      "$(( $(stat -c%s "$f") / 1048576 ))"
  done
}

case "${1:-all}" in
  repos)
    clone_at reinvent        "$REINVENT_URL"  "$REINVENT_SHA"
    clone_at reinvent-scoring "$SCORING_URL"  "$SCORING_SHA"
    clone_at roshambo2       "$ROSHAMBO2_URL" "$ROSHAMBO2_SHA" roshambo2.patch
    ;;
  priors) fetch_priors ;;
  verify) verify ;;
  all)
    clone_at reinvent        "$REINVENT_URL"  "$REINVENT_SHA"
    clone_at reinvent-scoring "$SCORING_URL"  "$SCORING_SHA"
    clone_at roshambo2       "$ROSHAMBO2_URL" "$ROSHAMBO2_SHA" roshambo2.patch
    fetch_priors
    ;;
  *) die "unknown stage '${1}'; use repos, priors, verify or all" ;;
esac
