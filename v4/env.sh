#!/bin/bash
# env.sh — portable PATH setup for v4 scripts
#
# Sources this to ensure claude CLI is on PATH.
# Checks common install locations. Override with CLAUDE_BIN env var.

if command -v claude &>/dev/null; then
    return 0 2>/dev/null || true
fi

# Check common locations
for dir in \
    "$HOME/.local/bin" \
    "$HOME/.nvm/versions/node/"*/bin \
    "$HOME/.local/share/fnm/node-versions/"*/installation/bin \
    "/usr/local/bin" \
; do
    if [ -x "$dir/claude" ] 2>/dev/null; then
        export PATH="$dir:$PATH"
        return 0 2>/dev/null || true
    fi
done

# Custom override
if [ -n "${CLAUDE_BIN:-}" ] && [ -x "$CLAUDE_BIN" ]; then
    export PATH="$(dirname "$CLAUDE_BIN"):$PATH"
    return 0 2>/dev/null || true
fi

echo "Warning: claude CLI not found. Set CLAUDE_BIN or install claude." >&2
