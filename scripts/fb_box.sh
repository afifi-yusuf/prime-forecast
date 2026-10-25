#!/usr/bin/env bash
# Run a live ForecastBench round on one AWS GPU instance, from this laptop.
#
#   scripts/fb_box.sh up               launch an instance (first type/region with capacity), set it up, serve
#   scripts/fb_box.sh smoke            20 live episodes on an old round (pipeline + search check)
#   scripts/fb_box.sh run    <due>     fetch the round and start the live run (background on the box)
#   scripts/fb_box.sh status [<due>]   vLLM, run progress, auto-stop time
#   scripts/fb_box.sh submit <due>     write the submission file on the box, copy it here, show coverage
#   scripts/fb_box.sh upload <due>     upload that file to the ForecastBench bucket (gcloud)
#   scripts/fb_box.sh down             terminate the instance
#   scripts/fb_box.sh ssh              shell on the box
#
# Config (no ids in the repo): ${FB_BOX_ENV:-~/.fb_box.env} sets FB_PROFILE, FB_KEY (key pair FB_KEY_NAME must
# exist in each region, with a security group FB_SG_NAME), FB_ADAPTER (local adapter dir), FB_BUCKET
# (gs://.../binary), FB_GCLOUD_ACCOUNT, FB_ORG, FB_MODEL_NAME, and optionally FB_REGIONS, FB_TYPES, FB_BRANCH,
# FB_HOURS (auto-terminate, default 24), FB_SEARCH (default agentcore,exa). The live instance is kept in ~/.fb_box.state.
#
# Instances are disposable: GPU capacity is often short, so `up` launches the first FB_TYPES x FB_REGIONS pair
# that has some, and `down` terminates. Types must be multi-GPU: vLLM loads the bf16 weights (72 GB) across the
# GPUs before quantizing to FP8, so a single 48 GB GPU runs out of memory while loading.
# Everything lives on the local NVMe, and `up` rebuilds it (idempotent, ~20 min):
# base model download, LoRA merge (scripts/merge_lora.py), vLLM serving the merged model in FP8, this repo.
# Only the keys a live run needs are copied to the box (search + FRED); live mode needs no Bedrock.
set -euo pipefail
_env_hours=${FB_HOURS:-}  # a command-line FB_HOURS=... beats the config file
source "${FB_BOX_ENV:-$HOME/.fb_box.env}"
FB_HOURS=${_env_hours:-${FB_HOURS:-24}}  # the box terminates at this deadline, outputs included
: "${FB_PROFILE:?}" "${FB_KEY:?}" "${FB_KEY_NAME:?}" "${FB_SG_NAME:?}"
FB_BRANCH=${FB_BRANCH:-main} FB_SEARCH=${FB_SEARCH:-agentcore,exa}
FB_REGIONS=${FB_REGIONS:-us-east-1 us-west-2}
FB_TYPES=${FB_TYPES:-g6.12xlarge g5.12xlarge g6.24xlarge g5.24xlarge g6e.12xlarge p4d.24xlarge}
STATE=$HOME/.fb_box.state
FB_INSTANCE="" FB_REGION=""
[ -f "$STATE" ] && source "$STATE"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
N=/opt/dlami/nvme W=/opt/dlami/nvme/work
aws_() { aws --profile "$FB_PROFILE" --region "$FB_REGION" "$@"; }
launch() {
  local myip; myip=$(curl -s https://checkip.amazonaws.com)
  for type in $FB_TYPES; do for region in $FB_REGIONS; do
    local sg ami id
    sg=$(aws --profile "$FB_PROFILE" --region "$region" ec2 describe-security-groups --filters "Name=group-name,Values=$FB_SG_NAME" \
      --query 'SecurityGroups[0].GroupId' --output text)
    aws --profile "$FB_PROFILE" --region "$region" ec2 authorize-security-group-ingress --group-id "$sg" --protocol tcp --port 22 \
      --cidr "$myip/32" >/dev/null 2>&1 || true  # current IP; already allowed is fine
    ami=$(aws --profile "$FB_PROFILE" --region "$region" ssm get-parameter --query Parameter.Value --output text \
      --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id)
    if id=$(aws --profile "$FB_PROFILE" --region "$region" ec2 run-instances --image-id "$ami" --instance-type "$type" \
        --key-name "$FB_KEY_NAME" --security-group-ids "$sg" --instance-initiated-shutdown-behavior terminate \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3,DeleteOnTermination=true}' \
        --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=fb-box},{Key=owner,Value=yusuf},{Key=purpose,Value=forecastbench}]' \
        --query 'Instances[0].InstanceId' --output text 2>/dev/null); then
      printf 'FB_INSTANCE=%s\nFB_REGION=%s\n' "$id" "$region" > "$STATE"; FB_INSTANCE=$id FB_REGION=$region
      echo "launched $type in $region: $id"; return 0
    fi
    echo "no capacity: $type in $region"
  done; done
  echo "no capacity anywhere in: $FB_TYPES / $FB_REGIONS" >&2; return 1
}
ip() { aws_ ec2 describe-instances --instance-ids "$FB_INSTANCE" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text; }
box() { ssh -i "$FB_KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "ubuntu@$(ip)" "$@"; }
put() { scp -i "$FB_KEY" -q "$@"; }

setup_remote() {  # runs on the box
cat <<EOF
set -euo pipefail
export PATH=\$HOME/.local/bin:\$PATH HF_HOME=$N/hf
sudo chown ubuntu:ubuntu $N && mkdir -p $N/models $N/adapters $W && cd $W
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[ -d prime-forecast ] || git clone -q https://github.com/afifi-yusuf/prime-forecast.git
(cd prime-forecast && git fetch -q origin && git checkout -q $FB_BRANCH && git pull -q --ff-only)
[ -x prime-forecast/.venv/bin/python ] || (cd prime-forecast && uv venv -q --python 3.12 .venv \
  && uv pip install -q -p .venv/bin/python -e environments/prime_forecast)
( [ -x vllm/bin/vllm ] || { uv venv -q --python 3.12 vllm && uv pip install -q -p vllm/bin/python "vllm==0.32.0" --torch-backend=auto; } ) > vllm_install.log 2>&1 &
[ -x dl/bin/python ] || { uv venv -q --python 3.12 dl && uv pip install -q -p dl/bin/python "huggingface_hub[hf_xet]" safetensors torch numpy; }
[ -f $N/models/base/model.safetensors.index.json ] || dl/bin/hf download Qwen/Qwen3.5-35B-A3B --local-dir $N/models/base >/dev/null
[ -f $N/models/merged/MERGED_FROM.json ] || dl/bin/python prime-forecast/scripts/merge_lora.py \
  --base $N/models/base --adapter $N/adapters/dyor --out $N/models/merged | tail -1
wait
cat > serve.sh <<'EOS'
export VLLM_API_KEY=local VLLM_USE_FLASHINFER_SAMPLER=0  # FlashInfer's sampler JIT-compiles; no CUDA toolkit here
exec $W/vllm/bin/vllm serve $N/models/merged --served-model-name dyor-v4 \\
  --host 127.0.0.1 --port 8000 --tensor-parallel-size \$(nvidia-smi -L | wc -l) --quantization fp8 \\
  --max-model-len 32768 --gpu-memory-utilization 0.90 \\
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \\
  --limit-mm-per-prompt '{"image":0,"video":0}'
EOS
# Every fd of a backgrounded list must leave the ssh channel, or ssh waits for the server to exit.
pgrep -f "[v]llm/bin/vllm serve" >/dev/null || { setsid nohup bash serve.sh > serve.log 2>&1 < /dev/null & } > /dev/null 2>&1
for i in \$(seq 1 90); do curl -sf -H "Authorization: Bearer local" localhost:8000/v1/models >/dev/null && break
  pgrep -f "[v]llm/bin/vllm serve" >/dev/null || { tail -20 serve.log; exit 1; }; sleep 10; done
curl -s localhost:8000/v1/chat/completions -H "Authorization: Bearer local" -H "content-type: application/json" -d '{"model":"dyor-v4","max_tokens":1024,"messages":[{"role":"user","content":"Will the Fed cut rates in December? Search first."}],"tools":[{"type":"function","function":{"name":"web_search","description":"Search the web","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}]}' \
  | python3 -c "import json,sys; m=json.load(sys.stdin)['choices'][0]['message']; print('vLLM ready; tool call:', (m.get('tool_calls') or [{}])[0].get('function',{}).get('name','NONE'))"
EOF
}

case "${1:-}" in
  up)
    aws --profile "$FB_PROFILE" sts get-caller-identity --query Arn --output text
    state=$([ -n "$FB_INSTANCE" ] && aws_ ec2 describe-instances --instance-ids "$FB_INSTANCE" \
      --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo none)
    [ "$state" = running ] && echo "reusing $FB_INSTANCE ($FB_REGION)" || launch
    aws_ ec2 wait instance-status-ok --instance-ids "$FB_INSTANCE"
    echo "box $(ip)"
    box "sudo shutdown -h +$((FB_HOURS * 60)) 'fb_box auto-stop' 2>&1; sudo chown ubuntu:ubuntu $N; mkdir -p $N/adapters/dyor $W"
    put "$FB_ADAPTER"/adapter_config.json "$FB_ADAPTER"/adapter_model.safetensors "ubuntu@$(ip):$N/adapters/dyor/"
    tmp=$(mktemp); trap 'rm -f "$tmp"' EXIT; chmod 600 "$tmp"
    grep -E "^(export )?(EXA_API_KEY|FRED_API_KEY|AGENTCORE_[A-Z_]+|AWS_REGION)=" "$ROOT/secrets.env" | sed 's/^export //' > "$tmp"
    printf 'PF_SEARCH_BACKEND=%s\nVLLM_API_KEY=local\n' "$FB_SEARCH" >> "$tmp"
    put "$tmp" "ubuntu@$(ip):$W/secrets.env"; box "chmod 600 $W/secrets.env"
    setup_remote | box bash -s ;;
  smoke)
    box "cd $W/prime-forecast && set -a && . ../secrets.env && set +a && rm -rf outputs/forecastbench/runs/2026-08-16/smoke \
      && .venv/bin/python scripts/forecastbench.py fetch --due 2026-08-16 >/dev/null \
      && .venv/bin/python scripts/forecastbench.py run --due 2026-08-16 --mode live --sample 20 --name smoke --max-concurrent 20 2>&1 \
         | grep -E 'episodes to run|batch|retry|coverage'" ;;
  run)
    due=${2:?due date}
    box "cd $W/prime-forecast && .venv/bin/python scripts/forecastbench.py fetch --due $due || exit 1
      set -a; . ../secrets.env; set +a; export PF_TREND_PROB=0  # trend fit without probabilities (FRED test)
      { setsid nohup .venv/bin/python scripts/forecastbench.py run --due $due --mode live --name live --batch 300 \
          --max-concurrent 40 > ../run_$due.log 2>&1 < /dev/null & } > /dev/null 2>&1
      sleep 20; grep -E 'episodes to run' $W/run_$due.log" ;;
  status)
    box "curl -sf -H 'Authorization: Bearer local' localhost:8000/v1/models >/dev/null && echo 'vLLM up' || echo 'vLLM DOWN';
      pgrep -f '[f]orecastbench.py run' >/dev/null && echo 'run: running' || echo 'run: not running';
      [ -n '${2:-}' ] && tr '\r' '\n' < $W/run_${2:-x}.log | grep -E 'episodes to run|batch|retry|coverage|Error' | tail -6;
      sudo shutdown --show 2>&1 | head -1" ;;
  submit)  # file 1: DYOR; file 2 (if the base run exists): DYOR+base averaged. Same post-processing for both.
    due=${2:?due date}; post="--variants anchored --source-prior ${FB_PRIORS:-acled=0.12 yfinance=0.5:0 fred=0.63:0.75} --climatology 0.25 --live-crowd --live-anchor-weight ${FB_LIVE_W:-0.9}"
    box "cd $W/prime-forecast && .venv/bin/python scripts/forecastbench.py submit --due $due --name live --org '${FB_ORG:?}' \
      --model-name '${FB_MODEL_NAME:?}' $post --number 1 && if [ -f outputs/forecastbench/runs/$due/live-base/rollouts.jsonl ]; then \
      .venv/bin/python scripts/forecastbench.py submit --due $due --name live --pool live-base --org '$FB_ORG' \
      --model-name '$FB_MODEL_NAME (ensemble)' $post --pool-weight wikipedia=0.25 --number 2; fi"
    mkdir -p "$ROOT/outputs/forecastbench/runs/$due/live"
    scp -i "$FB_KEY" -q "ubuntu@$(ip):$W/prime-forecast/outputs/forecastbench/runs/$due/live/*" "$ROOT/outputs/forecastbench/runs/$due/live/"
    ls -l "$ROOT/outputs/forecastbench/runs/$due/live/" ;;
  upload)
    due=${2:?due date}
    for f in "$ROOT/outputs/forecastbench/runs/$due/live/$due".*.json; do
      python3 -c "import json,sys; d=json.load(open('$f')); print('$(basename "$f")', '|', d['organization'], '|', d['model'], '|', len(d['forecasts']), 'forecasts')"
      gcloud storage cp "$f" "${FB_BUCKET:?}/$due/" --account="${FB_GCLOUD_ACCOUNT:?}"
    done
    gcloud storage ls -l "$FB_BUCKET/$due/" --account="$FB_GCLOUD_ACCOUNT" ;;
  down)
    : "${FB_INSTANCE:?no instance in $STATE}"
    aws_ ec2 terminate-instances --instance-ids "$FB_INSTANCE" --query 'TerminatingInstances[0].CurrentState.Name' --output text
    aws_ ec2 wait instance-terminated --instance-ids "$FB_INSTANCE" && rm -f "$STATE" && echo terminated ;;
  ssh) box ;;
  *) sed -n '2,12p' "$0"; exit 1 ;;
esac
