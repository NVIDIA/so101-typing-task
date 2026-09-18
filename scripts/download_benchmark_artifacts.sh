#!/usr/bin/env bash
# Download the released SO-101 typing checkpoints and their matched contracts.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/download_benchmark_artifacts.sh [--historical | --all | --calibrated] [OUTPUT_DIR]

Defaults to the September fitted-scene policy, matching calibration, training,
simulation and physical NEWTON evidence. --calibrated is an explicit alias.
--historical selects the original July AnchorBench 19k checkpoint, environment
and evaluation. --all includes both, plus supported development artifacts.
Downloads immutable artifacts from:
  https://huggingface.co/datasets/VigneshBhavan/so101-keyboard-typing-benchmark

If OUTPUT_DIR is omitted, files are placed under .artifacts/ in this checkout.
Every downloaded file is verified against the release SHA-256 values before it
is installed in the output directory.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

selection=calibrated
case "${1:-}" in
  --all) selection=all; shift ;;
  --calibrated) selection=calibrated; shift ;;
  --historical) selection=historical; shift ;;
esac
[[ $# -le 1 && "${1:-}" != --* ]] || { usage >&2; exit 2; }

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${1:-${SO101_ARTIFACT_DIR:-$repo_root/.artifacts/so101-keyboard-typing-benchmark}}"
output_dir="$(mkdir -p "$output_dir" && realpath "$output_dir")"
base_url="${SO101_HF_BASE_URL:-https://huggingface.co/datasets/VigneshBhavan/so101-keyboard-typing-benchmark/resolve/44d8ad43e2619e0d6e89528fea07f887de93678e}"

command -v sha256sum >/dev/null || { echo "ERROR: sha256sum is required" >&2; exit 1; }
if command -v curl >/dev/null; then
  downloader="curl"
elif command -v wget >/dev/null; then
  downloader="wget"
else
  echo "ERROR: curl or wget is required" >&2
  exit 1
fi

die() {
  echo "ERROR: $*" >&2
  exit 1
}

download_and_verify() {
  local relative_path="$1"
  local expected_sha256="$2"
  local destination="$output_dir/$relative_path"
  local temporary actual_sha256

  mkdir -p "$(dirname "$destination")"
  if [[ -f "$destination" ]]; then
    actual_sha256="$(sha256sum "$destination" | awk '{print $1}')"
    if [[ "$actual_sha256" == "$expected_sha256" ]]; then
      printf '[OK] %s\n' "$relative_path" >&2
      return
    fi
    printf '[INFO] replacing checksum-mismatched %s\n' "$relative_path" >&2
  fi

  temporary="$(mktemp "$destination.partial.XXXXXX")"
  if [[ "$downloader" == "curl" ]]; then
    if ! curl --fail --location --silent --show-error --retry 3 --retry-delay 1 \
      --output "$temporary" "$base_url/$relative_path"; then
      rm -f "$temporary"
      die "download failed: $relative_path"
    fi
  elif ! wget --quiet --output-document="$temporary" "$base_url/$relative_path"; then
    rm -f "$temporary"
    die "download failed: $relative_path"
  fi

  actual_sha256="$(sha256sum "$temporary" | awk '{print $1}')"
  if [[ "$actual_sha256" != "$expected_sha256" ]]; then
    rm -f "$temporary"
    die "checksum mismatch for $relative_path: expected $expected_sha256, got $actual_sha256"
  fi
  chmod 644 "$temporary"
  mv "$temporary" "$destination"
  printf '[OK] %s\n' "$relative_path" >&2
}

# Sanitized historical benchmark and September calibrated-scene release.
# Both the immutable dataset revision above and every payload checksum are pinned.
artifacts=(
  'README.md|11830a4f6296a84c7ce00d31898e81c4634bd0fbd0dd020a35c3746deda79a68'
  'RESULTS.md|675bf6d7bf2afa87682da6d8ddc8902abfa3fbe509508d6edec5c7f80c2c2c2a'
  'checkpoints/mjwarp-anchorbench-19k/model_19000.pt|239d93c99cd282e9add019f4ac952407c97fde2879821103a2aa7bd7a307a15f'
  'checkpoints/mjwarp-usd-19k/model_19000.pt|cc7f128e15a282a6f2fcc68e97f67c2e6f81bf6e03347b6a3cbf0b9a02d072a4'
  'checkpoints/public-p1a-seed1307/model_500.pt|f7fdd19b4c4b971c5363391292b4e3e7bbee602a195d7630814fa2a23d71e7a4'
  'checkpoints/public-p1a-seed1307/params/agent.yaml|a4ab12d5a6bbaa074e12460c397e970a6051de97b04e974ceed40084a23a2cdf'
  'checkpoints/public-p1a-seed1307/params/env.yaml|c61cba29921c2187f43591561ab80beb2980c7436e77709019d3bec04da34405'
  'checkpoints/public-transit15-seed1307/model_549.pt|ea6c940f8a57797acaaca845bfb4b7cf2f1780f3d9571536b20998cbf114e0a8'
  'checkpoints/public-transit15-seed1307/params/agent.yaml|7cc34a174d3f8ace38b1287b87e8a7137b99d151cec1eaadb293ac634434b25f'
  'checkpoints/public-transit15-seed1307/params/env.yaml|4942a720f3c2cc43aae6d66973c260aeec56a6e9891d5bb8b57febdf65e772bd'
  'configs/evaluation/placement-1mm.json|5090b4de25ad592309e31757a0a168c34bf93ee2baf922690f21a1d2a109860e'
  'configs/evaluation/placement-3mm.json|b39ab06dc6a69f24b07e155d82ce5194d26f652fe28631139666f2068483974a'
  'configs/evaluation/placement-zero.json|a0ab687ed2bf7cfb8d98a0b517d8b9db1f5a312132be20c84b72426e2ce54c2b'
  'configs/mjwarp-anchorbench-19k.env.yaml|d4766dacb28a4ffc0fc75efe8e5abed162e656b338b110e8d498cc808e31ef8b'
  'configs/mjwarp-usd-19k.env.yaml|d14d60d19554326d66cbffc175c42166232968380980574ae36c196c9a0f7acb'
  'evaluation/fresh-checkout-software-validation.json|b01f2a04b542dda099153c66b35a0a40e179475e92f4b948a54639d65aeba5d3'
  'evaluation/mjwarp-anchorbench-19k.corpus_100x6_seed1307.json|0598e2f235bcd3e2279c08a9c6e75934b47e84b58553b21dfd0e0089b57ac66a'
  'evaluation/mjwarp-usd-19k.corpus_100x6_seed1307.json|9db2f161f5bebac4b5a4dd5c9457d13fe0eeadf8eb3e2e01b42d8358a3ef0305'
  'evaluation/public-p1a-seed2307.json|5be3c74c0e665c8f70b9f69b6066a5b2a98ba212aefe65d78d8b08c8ddb95aca'
  'evaluation/public-p1a-seed3307.json|119d2b3461d2b0fbc64c78aa49c3b4941c77ebc4a50d2c1420863fe2b2983ee2'
  'evaluation/public-p1a-video-AZ.json|cfa37683482a8f1be632f86589a36a08f7f2754bfe2ff3d7fa3d51cf3e75bec1'
  'evaluation/public-p1a-video-HE.json|9b3fbdf2a757dc268dde798cac50e099af443a7df90bfd5165e71cd087649162'
  'evaluation/public-p1a-video-QZ.json|c4a116cd8b73e819aef6a9c0611d3dfd916fc9c4bcd05201e37301d7e370264d'
  'evaluation/public-transit15-actor-parity.json|96e6a4567e222d6733f53113eef408d7394fc436f07ca64f4b10ddd06a3e0e17'
  'evaluation/public-transit15-placement-1mm.json|63fb156081cb27ef733d844740d5ffd42f4521f4ff2729b0b563ca254639a852'
  'evaluation/public-transit15-placement-3mm.json|428e530369bc581e7f77b664311c3a0a716bf5919b859377695e78f0fe2b0478'
  'evaluation/public-transit15-placement-zero.json|34228a75faf9e1cfe61afbbcd634e190583daeec8de72240f44aff9151d6b29c'
  'evaluation/public-transit15-seed2307.json|6d98c925c847c9976f62e49948a6fc76485921482403ffd4ffb217a254647b4a'
  'evaluation/public-transit15-seed3307.json|e9a14707925073c3d37222505e9141ed1f954e709cf786ff1ab45d2cd1470644'
  'evaluation/public-transit15-video-NVIDIA.json|d83135a8ecffee5627a80ff4d51df1e74bffec04730fcd467e530b52c82dce73'
  'evaluation/run-provenance.json|d340ed3c0dadca6e4f21f1efcce8cca9d93b4274551485c3cf22f2777dc72da2'
  'evaluation/three_way_mjwarp19k_vbd20k_usd19k_orange2.summary.json|bb474e559394be754fed0d677712c4d2995416f2c5fae3d1a77e2ba4e413f274'
  'manifest.json|067a68ed3a3eb170a78f0030d85d0164bd0e7b7f551d86f90d74471543df38f0'
  'training/p1a-iterations.csv|fb666d504182e00e2709fbb22d357e33948961e6f50ec7ce663783f5838d9e9a'
  'training/public-pipeline.json|d5f418789915e521f013884b5cd0b221199ec8f4e7d66319de8137ba3fbd02fa'
  'training/transit15-iterations.csv|adabcd241b39894c855a0a2c50eaded015b1c858abe9efda2b4b49b4896a5e7c'
  'videos/public-p1a-AZ.mp4|221152c32717f3cd064c050d7e03b2b3986a35924c43e1d44af21ea03890a2fd'
  'videos/public-p1a-HE.mp4|af53c7db46af1b393a57b3117f356bc623b085d694d31d56dc60460834adc712'
  'videos/public-p1a-QZ.mp4|d17a7d1c463dee883f86978c06b6194977c6581ddf1d63c1d5998e8e7abc56b2'
  'videos/public-transit15-NVIDIA.mp4|071e726316af4a2f61aad9c3b46679b8de018e87732354c09d3d4ecb5a51d63a'
  'releases/calibrated-20260917/README.md|fa00fe7e8231e26ea8b2a136e427f5da25b0dae794ddfc373739fc01228a7450'
  'releases/calibrated-20260917/SHA256SUMS|ff5816d51e4285912c48fa5f440441bb8f9ed8db1cb337af1e09358035ec8b90'
  'releases/calibrated-20260917/calibration/contact-result.json|391d17dd1f53dc89f39367dab76ad30f2d517287ace07f2ca62c408f456007ac'
  'releases/calibrated-20260917/calibration/mapping-report.json|7ff64fa4dfc8f2c1bd02ce9e802b1052e7ddd707a95e2ee50adecd67839ebb9c'
  'releases/calibrated-20260917/calibration/task.json|3827dd35fa534f903b41d99cf598e74ad8da481baf363ef88ee6c312fef62d7d'
  'releases/calibrated-20260917/evaluation/actor-parity.json|5fd58267178bfbd7f6ed31de044e9ddcbdf4e019a9c059d46dd4d89e9f109d33'
  'releases/calibrated-20260917/evaluation/seed2307.json|fe958df7c5a9821f0c7433689759748fbeb1834c5915f61feb3c04a0f3d03d47'
  'releases/calibrated-20260917/evaluation/seed3307.json|8079e7ac892e1750fda6ededb0ca66418b8426fdcf92d3cea7ec42026af24495'
  'releases/calibrated-20260917/evaluation/video-NEWTON.json|8e3682142c217fc573c45c312d4dd16ae049a4650e027d416dfd8f58fa005fcd'
  'releases/calibrated-20260917/evaluation/video-NVIDIA.json|eeb2ea79edb2badab3a1307f251bcc46616fad2e3fff8c6d1164e8efeabced00'
  'releases/calibrated-20260917/hardware/NEWTON.jsonl|5ce18fffc8b8b36212b5674a7d0f31e9653061101dc28e12e06dc736d09935a9'
  'releases/calibrated-20260917/hardware/result.json|44f3092ebd0921d32ea55067df637596428c167834791f6d423ffc766fb32682'
  'releases/calibrated-20260917/manifest.json|c74e96507efaa7a030eaf5369d3acda661310c2c0b721d765186822aaf876b70'
  'releases/calibrated-20260917/model_2498.pt|6af850866096ec6b6ad6b202c8ac9417681efa0ebe71f4e8e8fdcf57d393f225'
  'releases/calibrated-20260917/params/agent.yaml|f18ef6bb0cfd19edc795875af5c86e4547aec08e9ac1b25000fd8526b3d6da06'
  'releases/calibrated-20260917/params/env.yaml|72bb3f80cd41da4930c0fe5de54707ba3dfa1ee772f9745afc128fcd01f8db32'
  'releases/calibrated-20260917/params/public_task.json|3827dd35fa534f903b41d99cf598e74ad8da481baf363ef88ee6c312fef62d7d'
  'releases/calibrated-20260917/training/provenance.json|24a2bb92f27fafc7c362ddac57af57e97df12dcedd1664c546f38182d265209d'
  'releases/calibrated-20260917/training/status.json|9b5062fb1ff6ddf780d13c402f20bf1db63e99d11228be596c55b8c342fe9627'
  'releases/calibrated-20260917/videos/simulation-NEWTON.mp4|090427894efddc34844a371b0e59e381f3f85ac172698f7e5fe3003a0459d37a'
  'releases/calibrated-20260917/videos/simulation-NVIDIA.mp4|e12649860759a39e9782f4af4c06708dc20875ee23ac79633add4be1b5f45191'
)

selected_checksums="$(mktemp "$output_dir/.download-checksums.XXXXXX")"
trap 'rm -f "$selected_checksums"' EXIT
for specification in "${artifacts[@]}"; do
  IFS='|' read -r relative_path expected_sha256 <<< "$specification"
  if [[ "$selection" == historical ]]; then
    case "$relative_path" in
      checkpoints/mjwarp-anchorbench-19k/model_19000.pt|configs/mjwarp-anchorbench-19k.env.yaml|evaluation/mjwarp-anchorbench-19k.corpus_100x6_seed1307.json) ;;
      *) continue ;;
    esac
  elif [[ "$selection" == calibrated && "$relative_path" != releases/calibrated-20260917/* ]]; then
    continue
  fi
  download_and_verify "$relative_path" "$expected_sha256"
  printf '%s  %s\n' "$expected_sha256" "$relative_path" >> "$selected_checksums"
done
chmod 644 "$selected_checksums"
mv "$selected_checksums" "$output_dir/DOWNLOAD_SHA256SUMS"

printf '%s\n' "$output_dir"
