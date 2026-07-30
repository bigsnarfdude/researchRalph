#!/bin/bash
# meta_artifacts.sh — shared helpers for meta-blackboard generation (v4.10).
#
# Sourced by generate-meta-blackboard.sh (one-shot, between generations) and
# meta-loop.sh (periodic, during a generation). Both used to assemble artifacts
# and write the result inline, and had drifted apart: different prompt-passing
# conventions, and both would destroy the existing meta-blackboard whenever the
# model call failed or returned nothing.
#
# meta-blackboard.md is the ONLY cross-generation memory channel. Losing it is
# worse than not refreshing it, so a failed refresh must leave the old file alone.

# append_domain_artifacts DOMAIN_DIR PROMPT_FILE
#
# Appends whatever run artifacts actually exist. Every read is guarded: domains
# have different shapes (SAE domains have best/sae.py, Lean domains have neither
# sae.py nor a best/ dir at all), and an unguarded `cat` contributes nothing but
# a stderr error.
append_domain_artifacts() {
    local domain_dir="$1" prompt_file="$2"
    local f name

    for name in blackboard.md results.tsv; do
        if [ -s "$domain_dir/$name" ]; then
            echo "### $name" >> "$prompt_file"
            cat "$domain_dir/$name" >> "$prompt_file"
            echo "" >> "$prompt_file"
        fi
    done

    append_best_artifacts "$domain_dir" "$prompt_file"
}

# append_best_artifacts DOMAIN_DIR PROMPT_FILE
#
# best/ holds the current champion artifacts. Shape is domain-specific, so take
# every text file rather than hardcoding config.yaml + sae.py — Lean and
# nirenberg domains have neither, and some have no best/ dir at all. Skip
# config_hash (opaque digest) and binaries; truncate long files.
append_best_artifacts() {
    local domain_dir="$1" prompt_file="$2"
    local f

    [ -d "$domain_dir/best" ] || return 0
    for f in "$domain_dir/best"/*; do
        [ -f "$f" ] || continue
        case "$(basename "$f")" in
            config_hash|*.pt|*.bin|*.safetensors|*.npy|*.pkl) continue ;;
        esac
        echo "### best/$(basename "$f")" >> "$prompt_file"
        head -400 "$f" >> "$prompt_file"
        echo "" >> "$prompt_file"
    done
}

# commit_meta_blackboard DOMAIN_DIR TMP_FILE CLAUDE_EXIT
#
# Replaces meta-blackboard.md only if the refresh actually produced a usable
# cheat sheet. Otherwise the previous version survives and we say so loudly.
# Returns 0 on commit, 1 on rejection.
commit_meta_blackboard() {
    local domain_dir="$1" tmp_file="$2" claude_exit="$3"
    local target="$domain_dir/meta-blackboard.md"
    local headings reason=""

    # grep -c prints its count AND exits 1 on zero matches, so capture the
    # number only — a `|| echo 0` fallback would append a second line.
    headings=$(grep -c '^## ' "$tmp_file" 2>/dev/null | tr -dc '0-9')
    headings="${headings:-0}"

    # Structure is the real signal, not length: a valid cheat sheet for a thin
    # domain can be short, while a failed call is either empty, an error string,
    # or an apology paragraph — none of which carry '## ' sections.
    if [ "$claude_exit" -ne 0 ]; then
        reason="claude exited $claude_exit"
    elif ! grep -q '[^[:space:]]' "$tmp_file" 2>/dev/null; then
        reason="output was empty"
    elif [ "$headings" -lt 2 ]; then
        reason="output has $headings '## ' sections — looks like an error or refusal, not a cheat sheet"
    fi

    if [ -n "$reason" ]; then
        echo "[meta] REFRESH FAILED: $reason" >&2
        if [ -s "$target" ]; then
            echo "[meta] Kept existing meta-blackboard.md ($(wc -l < "$target" | tr -d ' ') lines) — NOT overwritten." >&2
        else
            echo "[meta] No usable meta-blackboard.md exists; cross-generation memory is EMPTY for this domain." >&2
        fi
        mv "$tmp_file" "$target.rejected" 2>/dev/null
        echo "[meta] Rejected output saved to $target.rejected for inspection." >&2
        return 1
    fi

    # Keep one generation of history so a bad-but-passing refresh is recoverable.
    [ -s "$target" ] && cp "$target" "$target.prev"
    mv "$tmp_file" "$target"
    echo "[meta] Wrote meta-blackboard.md ($(wc -l < "$target" | tr -d ' ') lines, $headings sections)"
    return 0
}
