#!/usr/bin/env bash
# Download one large file with N parallel HTTP range requests (resumable).
# Useful when the server limits per-connection speed (e.g. the NYU .mat from outside the US).
# Usage: bash scripts/parallel_fetch.sh <url> <output> [connections=16]
set -euo pipefail
url=$1; out=$2; n=${3:-16}

size=$(curl -sIL "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{v=$2} END{print v}')
[ -n "$size" ] || { echo "server did not report Content-Length" >&2; exit 1; }
if [ -f "$out" ] && [ "$(stat -c %s "$out")" = "$size" ]; then echo "skip  $out"; exit 0; fi

seg=$(( (size + n - 1) / n ))
dir="$out.segments"
mkdir -p "$(dirname "$out")"
# Reuse a partial single-connection download as the start of segment 0
if [ -f "$out.part" ] && [ ! -d "$dir" ]; then
  mkdir -p "$dir"; mv "$out.part" "$dir/0"
  [ "$(stat -c %s "$dir/0")" -gt "$seg" ] && truncate -s "$seg" "$dir/0"
fi
mkdir -p "$dir"

fetch_segment() {
  local i=$1
  local start=$(( i * seg )) end=$(( (i + 1) * seg - 1 ))
  (( end >= size )) && end=$(( size - 1 ))
  local f="$dir/$i" want=$(( end - start + 1 ))
  touch "$f"
  for _ in $(seq 1 100); do
    local have; have=$(stat -c %s "$f")
    (( have >= want )) && return 0
    curl -sL --fail --max-time 1800 -r "$(( start + have ))-$end" "$url" >> "$f" || sleep 3
  done
  echo "segment $i did not finish" >&2; return 1
}

echo "fetch $out ($size bytes, $n connections)"
for i in $(seq 0 $(( n - 1 ))); do fetch_segment "$i" & done
wait

parts=(); for i in $(seq 0 $(( n - 1 ))); do parts+=("$dir/$i"); done
cat "${parts[@]}" > "$out.tmp"
if [ "$(stat -c %s "$out.tmp")" = "$size" ]; then
  mv "$out.tmp" "$out"; rm -rf "$dir"; echo "done  $out"
else
  echo "size mismatch for $out; rerun to resume" >&2; rm -f "$out.tmp"; exit 1
fi
