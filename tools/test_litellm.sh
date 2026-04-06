#!/bin/bash
# test_litellm.sh — Verify LiteLLM proxy + Ollama + claude CLI integration
#
# Run this on nigel AFTER:
#   1. ollama pull gemma4:26b   (or gemma4:12b)
#   2. litellm --config tools/litellm_config.yaml --port 4000 &
#
# Usage:
#   bash tools/test_litellm.sh [port]      # default port 4000
#   bash tools/test_litellm.sh 4000

PORT="${1:-4000}"
BASE_URL="http://localhost:$PORT"

export ANTHROPIC_BASE_URL="$BASE_URL"
export ANTHROPIC_API_KEY="sk-litellm"

# Find claude CLI
CLAUDE_BIN=""
for dir in "$HOME/.local/bin" "$HOME/.nvm/versions/node/"*/bin "/usr/local/bin"; do
    if [ -x "$dir/claude" ] 2>/dev/null; then
        CLAUDE_BIN="$dir/claude"
        export PATH="$dir:$PATH"
        break
    fi
done
if [ -z "$CLAUDE_BIN" ]; then
    echo "ERROR: claude CLI not found. Checked ~/.local/bin and /usr/local/bin"
    exit 1
fi
echo "Claude: $CLAUDE_BIN"

echo "=== LiteLLM + Ollama + claude CLI test ==="
echo "Proxy: $BASE_URL"
echo ""

PASS=0
FAIL=0

check() {
    local label="$1"
    local result="$2"
    local expect="$3"
    if echo "$result" | grep -qi "$expect"; then
        echo "  PASS  $label"
        PASS=$((PASS+1))
    else
        echo "  FAIL  $label"
        echo "        expected: $expect"
        echo "        got: $(echo "$result" | head -3)"
        FAIL=$((FAIL+1))
    fi
}

# 1. Proxy health (401 = auth required = proxy is up)
echo "--- 1. Proxy health ---"
HEALTH=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health" 2>&1)
check "proxy responds" "$HEALTH" "200\|401"

# 2. Single-shot prompt
echo ""
echo "--- 2. Single-shot prompt (gemma4) ---"
OUT=$(claude --model gemma4 -p "Reply with exactly: LITELLM_OK" 2>&1)
check "single-shot response" "$OUT" "LITELLM_OK"

# 3. Tool use — file read
echo ""
echo "--- 3. Tool use: file read ---"
echo "hello from litellm test" > /tmp/litellm_test.txt
OUT=$(claude --model gemma4 \
    --allowedTools "Read" \
    -p "Read the file /tmp/litellm_test.txt and tell me what it says." 2>&1)
check "file read tool" "$OUT" "hello from litellm test"
rm -f /tmp/litellm_test.txt

# 4. Tool use — bash execution
echo ""
echo "--- 4. Tool use: bash execution ---"
OUT=$(claude --model gemma4 \
    --allowedTools "Bash" \
    -p "Run: echo BASH_WORKS and tell me what it printed." 2>&1)
check "bash tool" "$OUT" "BASH_WORKS"

# 5. Multi-turn (max-turns 3)
echo ""
echo "--- 5. Multi-turn (--max-turns 3) ---"
OUT=$(claude --model gemma4 \
    --max-turns 3 \
    -p "Count from 1 to 3, one number per message." 2>&1)
check "multi-turn" "$OUT" "1\|2\|3"

# 6. stream-json output format
echo ""
echo "--- 6. stream-json output format ---"
OUT=$(claude --model gemma4 \
    --output-format stream-json \
    -p "Say hello." 2>&1)
check "stream-json events" "$OUT" "content\|delta\|type"

# 7. Haiku passthrough (cloud) — only if ANTHROPIC_API_KEY_CLOUD is set
echo ""
echo "--- 7. Haiku cloud passthrough ---"
if [ -n "$ANTHROPIC_API_KEY_CLOUD" ]; then
    OUT=$(claude --model haiku -p "Reply with exactly: HAIKU_OK" 2>&1)
    check "haiku passthrough" "$OUT" "HAIKU_OK"
else
    echo "  SKIP  haiku passthrough (set ANTHROPIC_API_KEY_CLOUD to test)"
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
if [ "$FAIL" -eq 0 ]; then
    echo "All good. Wire up the harness:"
    echo ""
    echo "  export ANTHROPIC_BASE_URL=http://localhost:$PORT"
    echo "  export ANTHROPIC_API_KEY=sk-litellm"
    echo "  RRMA_MODEL=gemma4 bash v4/launch-agents-chaos-v2.sh domains/nirenberg-1d-chaos-haiku 2 '1' 50 5"
else
    echo "Fix failures before running the harness."
    exit 1
fi
