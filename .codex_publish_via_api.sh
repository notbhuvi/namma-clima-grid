#!/bin/zsh
set -euo pipefail

repo="notbhuvi/namma-clima-grid"
branch="main"
api="https://api.github.com/repos/$repo"
GH_BIN="/opt/homebrew/bin/gh"
JQ_BIN="/usr/bin/jq"
CURL_BIN="/usr/bin/curl"
BASE64_BIN="/usr/bin/base64"
TR_BIN="/usr/bin/tr"
RM_BIN="/bin/rm"
MV_BIN="/bin/mv"

token="$("$GH_BIN" auth token)"

auth_header="Authorization: Bearer $token"
accept_header="Accept: application/vnd.github+json"

tmp_dir="$(mktemp -d)"
tree_file="$tmp_dir/tree.json"
printf '[]' > "$tree_file"

cleanup() {
  "$RM_BIN" -rf "$tmp_dir"
}
trap cleanup EXIT

api_call() {
  local method="$1"
  local url="$2"
  local data="${3:-}"

  if [[ -n "$data" ]]; then
    local payload_file="$tmp_dir/payload.json"
    printf '%s' "$data" > "$payload_file"
    "$CURL_BIN" -fsSL -X "$method" \
      -H "$auth_header" \
      -H "$accept_header" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "$url" \
      --data-binary "@$payload_file"
  else
    "$CURL_BIN" -fsSL -X "$method" \
      -H "$auth_header" \
      -H "$accept_header" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "$url"
  fi
}

if ! api_call GET "$api/git/ref/heads/$branch" >/dev/null 2>&1; then
  bootstrap_content="$("$BASE64_BIN" < .gitignore | "$TR_BIN" -d '\n')"
  bootstrap_payload="$("$JQ_BIN" -n \
    --arg message "Bootstrap branch" \
    --arg content "$bootstrap_content" \
    --arg branch "$branch" \
    '{message: $message, content: $content, branch: $branch}')"
  api_call PUT "$api/contents/.gitignore" "$bootstrap_payload" >/dev/null
fi

ref_json="$(api_call GET "$api/git/ref/heads/$branch")"
parent_sha="$(printf '%s' "$ref_json" | "$JQ_BIN" -r '.object.sha')"
commit_json="$(api_call GET "$api/git/commits/$parent_sha")"
base_tree_sha="$(printf '%s' "$commit_json" | "$JQ_BIN" -r '.tree.sha')"

while IFS=$'\t' read -r meta path; do
  mode="${meta%% *}"
  file_b64="$("$BASE64_BIN" < "$path" | "$TR_BIN" -d '\n')"
  blob_payload="$(printf '%s' "$file_b64" | "$JQ_BIN" -Rs '{content: ., encoding: "base64"}')"
  blob_json="$(api_call POST "$api/git/blobs" "$blob_payload")"
  blob_sha="$(printf '%s' "$blob_json" | "$JQ_BIN" -r '.sha')"

  "$JQ_BIN" --arg path "$path" --arg mode "$mode" --arg sha "$blob_sha" \
    '. += [{path: $path, mode: $mode, type: "blob", sha: $sha}]' \
    "$tree_file" > "$tree_file.tmp"
  "$MV_BIN" "$tree_file.tmp" "$tree_file"
done < <(git ls-tree -r HEAD)

tree_payload="$("$JQ_BIN" -n \
  --arg base_tree "$base_tree_sha" \
  --slurpfile tree "$tree_file" \
  '{base_tree: $base_tree, tree: $tree[0]}')"
tree_json="$(api_call POST "$api/git/trees" "$tree_payload")"
tree_sha="$(printf '%s' "$tree_json" | "$JQ_BIN" -r '.sha')"

commit_payload="$("$JQ_BIN" -n \
  --arg message "Initial commit" \
  --arg tree "$tree_sha" \
  --arg parent "$parent_sha" \
  '{message: $message, tree: $tree, parents: [$parent]}')"
new_commit_json="$(api_call POST "$api/git/commits" "$commit_payload")"
new_commit_sha="$(printf '%s' "$new_commit_json" | "$JQ_BIN" -r '.sha')"

update_payload="$("$JQ_BIN" -n --arg sha "$new_commit_sha" '{sha: $sha, force: false}')"
api_call PATCH "$api/git/refs/heads/$branch" "$update_payload" >/dev/null

printf 'Published %s at commit %s\n' "$repo" "$new_commit_sha"
