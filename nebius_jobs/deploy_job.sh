#!/usr/bin/env bash
# Usage: REGION=<region> REGISTRY_PATH=<registry-id-without-registry-prefix> VOLUME="<bucket-or-fs>:/results:rw" [DRY_RUN=1] ./nebius_jobs/deploy_job.sh
# Not deployed by the author (no billing on the dev account). Platform/preset are unverified defaults.
set -euo pipefail
: "${REGION:?set REGION}" "${REGISTRY_PATH:?set REGISTRY_PATH}"
PLATFORM="${PLATFORM:-cpu-d3}"; PRESET="${PRESET:-8vcpu-32gb}"; EPISODES="${EPISODES:-100}"
IMAGE="cr.${REGION}.nebius.cloud/${REGISTRY_PATH}/rescuesim-usar:latest"
docker build --platform linux/amd64 -t "$IMAGE" .
docker push "$IMAGE"
ARGS=(ai job create --name "rescuesim-eval-$(date +%Y%m%d-%H%M)" --image "$IMAGE"
  --platform "$PLATFORM" --preset "$PRESET" --timeout 2h --disk-size 50Gi
  --container-command python
  --args "-m nebius_jobs.batch_policy_eval --episodes ${EPISODES} --workers 8 --output /results/serverless_eval.json")
[ -n "${VOLUME:-}" ] && ARGS+=(--volume "$VOLUME")
[ -n "${DRY_RUN:-}" ] && ARGS+=(--dry-run)
nebius "${ARGS[@]}"