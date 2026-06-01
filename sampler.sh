#!/usr/bin/env bash
# sampler.sh — periodic machine-load sampler for the community benchmark (DefDiff-benchmark#2).
#
# Records CPU demand / memory / thermal state while a benchmark run is in flight, so a run taken
# under background load can be detected after the fact (replication cancels random noise but not
# systematic contamination — see issue #2). Record-only; no gating.
#
# Usage: sampler.sh <output.jsonl> [cadence_seconds]
#   Emits one JSON object per line to <output.jsonl>, every cadence seconds (default 2), until
#   killed (SIGTERM/SIGINT → clean exit). Each line:
#     {"t_offset_s":N,"loadavg_1m":N,"loadavg_5m":N,"loadavg_15m":N,
#      "free_mb":N,"active_mb":N,"wired_mb":N,"compressed_mb":N,"cpu_speed_limit":N|null}
#
# No-sudo, macOS-first. Cheapest signals only (sysctl/vm_stat/pmset are ~instant) so the sampler
# does not perturb the very measurement it accompanies (Heisenberg).

set -u

OUT="${1:?usage: sampler.sh <output.jsonl> [cadence_seconds]}"
CADENCE="${2:-2}"

# Clean exit on signal so the parent's kill produces a graceful stop, not a dangling loop.
trap 'exit 0' TERM INT

# Page size: Apple Silicon is 16384 (16 KB), NOT 4096. Read it — never hardcode.
PAGE_SIZE=$(sysctl -n hw.pagesize 2>/dev/null || echo 4096)

t0=$(date +%s)

# Pull free/active/wired/compressed page counts from vm_stat in one pass → MB.
# vm_stat lines look like "Pages free:    42733." — strip label + trailing dot.
read_mem_mb() {
  vm_stat 2>/dev/null | awk -v ps="$PAGE_SIZE" '
    /Pages free:/                  { gsub(/\./,"",$3); free=$3 }
    /Pages active:/                { gsub(/\./,"",$3); active=$3 }
    /Pages wired down:/            { gsub(/\./,"",$4); wired=$4 }
    /Pages occupied by compressor:/{ gsub(/\./,"",$5); comp=$5 }
    END {
      mb = ps / 1048576.0
      printf "%.1f %.1f %.1f %.1f", free*mb, active*mb, wired*mb, comp*mb
    }'
}

# CPU_Speed_Limit from pmset -g therm. Absent line (common when NOT throttled) → empty → null.
read_speed_limit() {
  pmset -g therm 2>/dev/null | awk -F'= ' '/CPU_Speed_Limit/ {gsub(/ /,"",$2); print $2; found=1} END {if(!found) print ""}'
}

while :; do
  now=$(date +%s)
  t_offset=$(( now - t0 ))

  # loadavg: "{ 24.72 27.49 24.90 }" → three floats
  read -r l1 l5 l15 < <(sysctl -n vm.loadavg 2>/dev/null | tr -d '{}' )

  read -r free_mb active_mb wired_mb comp_mb < <(read_mem_mb)
  speed=$(read_speed_limit)
  if [ -z "$speed" ]; then speed_json="null"; else speed_json="$speed"; fi

  printf '{"t_offset_s":%d,"loadavg_1m":%s,"loadavg_5m":%s,"loadavg_15m":%s,"free_mb":%s,"active_mb":%s,"wired_mb":%s,"compressed_mb":%s,"cpu_speed_limit":%s}\n' \
    "$t_offset" "${l1:-0}" "${l5:-0}" "${l15:-0}" \
    "${free_mb:-0}" "${active_mb:-0}" "${wired_mb:-0}" "${comp_mb:-0}" "$speed_json" \
    >> "$OUT"

  sleep "$CADENCE"
done
