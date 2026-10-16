#!/bin/bash
# Build versioned runtime images locally. Never pushes images or model files.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
if [[ $# != 2 ]]; then echo "Usage: $0 REGISTRY/NAMESPACE VERSION" >&2; exit 2; fi
namespace=$1
version=$2
base="$namespace-base:$version"
vision="$namespace-vision:$version"
docker build -f docker/typing-public.Dockerfile -t "$base" .
docker build -f docker/vision-ppo.Dockerfile --build-arg "SIM_IMAGE=$base" -t "$vision" .
docker image inspect --format '{{.RepoTags}} {{.Id}}' "$base" "$vision"
echo 'Images built locally; nothing uploaded. Record registry digests after explicit publication.'
