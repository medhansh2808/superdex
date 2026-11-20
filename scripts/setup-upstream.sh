#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
upstream="$root/.upstream"
mkdir -p -- "$upstream"

checkout_pinned() (
    name=$1
    revision=$2
    path="$upstream/$name"

    if [[ -e "$path" || -L "$path" ]]; then
        if [[ ! -L "$path" && -d "$path/.git" ]] &&
            [[ $(git -C "$path" rev-parse HEAD) == "$revision" ]] &&
            [[ -z $(git -C "$path" status --porcelain --untracked-files=no) ]]; then
            return
        fi
        printf 'Expected a clean checkout of %s at %s in %s.\n' "$name" "$revision" "$path" >&2
        exit 1
    fi

    staging=$(mktemp -d "$upstream/$name.XXXXXX")
    trap 'rm -rf -- "$staging"' EXIT
    git -C "$staging" init --quiet
    git -C "$staging" fetch --depth=1 "https://github.com/HybridRobotics/$name.git" "$revision"
    git -C "$staging" checkout --quiet --detach "$revision"
    mv -T -- "$staging" "$path"
)

checkout_pinned supermimic 698ea39bc901d8fe733b3c8621df3199d241f7e4
checkout_pinned mjlab_research 6ea3e574b08605b79a290a6d464a812a26acd6fd

# SuperMimic's lockfile resolves mjlab_research as a sibling checkout.
uv sync --project "$upstream/supermimic" --locked --no-dev --python "$(cat "$root/.python-version")"
