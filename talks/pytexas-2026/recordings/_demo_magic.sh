#!/usr/bin/env bash
# Minimal "demo magic" helper for asciinema recordings.
#
# Provides three functions:
#   pe "<cmd>"   — print a fake prompt, simulate typing the command, then run it
#   p  "<text>"  — same, but type-only (no execution; useful for comments)
#   pause [sec]  — sleep for ``sec`` seconds (default 1); shows up as dead air
#                  on playback but asciinema --idle-time-limit caps it at 2s.
#
# Why simulate typing instead of just echoing the command?
#   asciinema faithfully records terminal output with timestamps. If we just
#   ``echo`` the command, it appears instantly on playback, which undermines
#   the "I'm typing live" feel the audience expects from a fallback recording.
#   Simulating at ~20 chars/sec reads as "fast but human."
#
# Override the typing speed with: ``TYPE_SPEED=30 bash record_03_rich_cli.sh``

# --- configuration -----------------------------------------------------------
TYPE_SPEED="${TYPE_SPEED:-20}"     # characters per second
PROMPT="${PROMPT:-\$ }"

# --- internals ---------------------------------------------------------------
_delay_per_char() {
    awk -v s="${TYPE_SPEED}" 'BEGIN { printf "%.3f", 1/s }'
}

_type_out() {
    local text="$1"
    local delay
    delay=$(_delay_per_char)
    printf '%s' "${PROMPT}"
    # ``read -r -n1`` iterates character-by-character. IFS= keeps whitespace.
    local i ch
    for (( i=0; i<${#text}; i++ )); do
        ch="${text:i:1}"
        printf '%s' "${ch}"
        sleep "${delay}"
    done
    printf '\n'
}

# --- public API --------------------------------------------------------------
# Print + run.
pe() {
    _type_out "$1"
    eval "$1"
}

# Print only — no execution. Good for comment lines like ``p "# note: ..."``
p() {
    _type_out "$1"
}

# Pause — dead air the viewer reads as "presenter is explaining this bit."
pause() {
    sleep "${1:-1}"
}
