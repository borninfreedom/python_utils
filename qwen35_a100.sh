#!/usr/bin/env bash
set -euo pipefail

# ============ Usage ============
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --daemon     Run in background (nohup mode)"
    echo "  -s, --stop       Stop the running server"
    echo "  --status         Check server status"
    echo "  -h, --help       Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  MODEL_NAME              Model to serve (default: /root/models/Qwen3.5-27B)"
    echo "  PORT                    Port to listen on (default: 8005)"
    echo "  API_KEY                 API key (default: token-qwen35)"
    echo "  GPU_MEMORY_UTILIZATION  GPU memory utilization (default: 0.9)"
    echo "  MAX_MODEL_LEN           Max model length (default: 8192)"
    echo "  MAX_NUM_SEQS            Max concurrent sequences per iteration (default: 8)"
    echo "  MAX_NUM_BATCHED_TOKENS  Max batched tokens per iteration (default: 2048)"
    echo "  PERFORMANCE_MODE        vLLM mode: balanced/interactivity/throughput (default: interactivity)"
    echo "  MM_PROCESSOR_KWARGS     Optional JSON for multimodal processor"
    echo "  ALLOWED_LOCAL_MEDIA_PATH  Path for local media files (default: project root)"
    echo "  VENV_PATH               uv venv path (default: /root/.venv_qwen35)"
    echo "  USE_UV_ENV              Enable uv env activation (default: 1)"
    echo "  LD_LIBRARY_PATH_BASE    Base path for libstdc++ (default: /root/miniforge3/envs/lib)"
    exit 0
}

# ============ Configuration ============
# Model configuration
MODEL_REPO="Qwen/Qwen3.5-27B"
MODEL_DIR="/root/models/Qwen3.5-27B"
MODEL_NAME="${MODEL_NAME:-${MODEL_DIR}}"

# Server configuration
PORT="${PORT:-8005}"
API_KEY="${API_KEY:-token-qwen35}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# 27B model needs more memory, 0.9 on 80G A100 is safe (~72GB)
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
PERFORMANCE_MODE="${PERFORMANCE_MODE:-interactivity}"
MM_PROCESSOR_KWARGS="${MM_PROCESSOR_KWARGS:-}"
VENV_PATH="${VENV_PATH:-/root/.venv_qwen35}"
USE_UV_ENV="${USE_UV_ENV:-1}"
LD_LIBRARY_PATH_BASE="${LD_LIBRARY_PATH_BASE:-/root/miniforge3/envs/lib}"

# GPU Configuration
# User requested to use Card 2
export CUDA_VISIBLE_DEVICES=1
export HF_ENDPOINT

# 允许访问本地媒体文件的目录路径（用于 file:// 协议）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-${PROJECT_ROOT}}"

# ============ Log Configuration ============
LOG_DIR="${SCRIPT_DIR}/qwen3_5_logs"
LOG_FILE="${LOG_DIR}/log"
PID_FILE="${LOG_DIR}/server.pid"
ROTATE_PID_FILE="${LOG_DIR}/rotate.pid"
MAX_LOG_SIZE=$((50 * 1024 * 1024))  # 50MB in bytes

# ============ Parse Arguments ============
DAEMON_MODE=false
STOP_MODE=false
STATUS_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--daemon)
            DAEMON_MODE=true
            shift
            ;;
        -s|--stop)
            STOP_MODE=true
            shift
            ;;
        --status)
            STATUS_MODE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# ============ Model Verification & Download ============
# Define the expected files for Qwen3.5-27B
# Based on: https://huggingface.co/Qwen/Qwen3.5-27B/tree/main
REQUIRED_FILES=(
    "config.json"
    "generation_config.json"
    "model.safetensors.index.json"
    "tokenizer.json"
    "tokenizer_config.json"
    "vocab.json"
    "merges.txt"
    "video_preprocessor_config.json"
    "preprocessor_config.json"
    "chat_template.jinja"
    "README.md"
)

# Add shard files (11 shards)
for i in $(seq -f "%05g" 1 11); do
    REQUIRED_FILES+=("model.safetensors-${i}-of-00011.safetensors")
done

verify_safetensors_integrity() {
    echo "Verifying safetensors file integrity..."
    python3 -c "
import os, sys, json, struct

model_dir = '$MODEL_DIR'
failed = False

if not os.path.exists(model_dir):
    print(f'Error: Model directory {model_dir} does not exist.')
    sys.exit(1)

for filename in os.listdir(model_dir):
    if filename.endswith('.safetensors'):
        file_path = os.path.join(model_dir, filename)
        try:
            file_size = os.path.getsize(file_path)
            with open(file_path, 'rb') as f:
                header_size_bytes = f.read(8)
                if len(header_size_bytes) != 8:
                    print(f'Error: File {filename} is too small.')
                    failed = True
                    continue

                header_size = struct.unpack('<Q', header_size_bytes)[0]
                header_bytes = f.read(header_size)
                if len(header_bytes) != header_size:
                    print(f'Error: File {filename} header is truncated.')
                    failed = True
                    continue

                header = json.loads(header_bytes)
                max_offset = 0
                for k, v in header.items():
                    if k != '__metadata__' and 'data_offsets' in v:
                        max_offset = max(max_offset, v['data_offsets'][1])

                expected_size = 8 + header_size + max_offset
                if file_size != expected_size:
                    print(f'Error: File {filename} is truncated/corrupted. Expected {expected_size}, got {file_size}')
                    failed = True
        except Exception as e:
            print(f'Error checking {filename}: {e}')
            failed = True

if failed:
    sys.exit(1)
else:
    print('All safetensors files verified successfully.')
"
}

verify_model() {
    echo "Verifying model integrity in ${MODEL_DIR}..."
    local missing_files=0

    if [[ ! -d "${MODEL_DIR}" ]]; then
        echo "Model directory not found."
        return 1
    fi

    for file in "${REQUIRED_FILES[@]}"; do
        if [[ ! -f "${MODEL_DIR}/${file}" ]]; then
            echo "Missing file: ${file}"
            missing_files=1
        fi
    done

    if [[ ${missing_files} -eq 1 ]]; then
        echo "Integrity check failed: Some files are missing."
        return 1
    fi

    # Check safetensors integrity (size check based on header)
    if ! verify_safetensors_integrity; then
        echo "Integrity check failed: Some safetensors files are corrupted."
        return 1
    fi

    echo "Integrity check passed: All required files are present and valid."
    return 0
}

# Check if model exists and is valid, if not download/repair
if ! verify_model; then
    echo "Downloading/Repairing ${MODEL_REPO} from ${HF_ENDPOINT}..."
    mkdir -p "${MODEL_DIR}"
    # Use hf-transfer for faster downloads if available (optional)
    # HF_HUB_ENABLE_HF_TRANSFER=1 
    huggingface-cli download --resume-download "${MODEL_REPO}" --local-dir "${MODEL_DIR}"

    # Re-verify after download
    if ! verify_model; then
        echo "Error: Model verification failed even after download."
        exit 1
    fi
    echo "Model downloaded and verified successfully."
fi

# ============ Stop Function ============
stop_server() {
    echo "Stopping vllm Qwen3.5 server..."

    # Stop log rotation process
    if [[ -f "${ROTATE_PID_FILE}" ]]; then
        ROTATE_PID=$(cat "${ROTATE_PID_FILE}")
        if kill -0 "${ROTATE_PID}" 2>/dev/null; then
            kill "${ROTATE_PID}" 2>/dev/null || true
            echo "Log rotation process (PID: ${ROTATE_PID}) stopped."
        fi
        rm -f "${ROTATE_PID_FILE}"
    fi

    # Stop vllm server
    if [[ -f "${PID_FILE}" ]]; then
        SERVER_PID=$(cat "${PID_FILE}")
        if kill -0 "${SERVER_PID}" 2>/dev/null; then
            kill "${SERVER_PID}" 2>/dev/null
            echo "Server (PID: ${SERVER_PID}) stopped."
        else
            echo "Server process not running."
        fi
        rm -f "${PID_FILE}"
    else
        echo "PID file not found. Server may not be running."
        # Try to find and kill by port
        PID=$(lsof -t -i:"${PORT}" 2>/dev/null || true)
        if [[ -n "${PID}" ]]; then
            kill "${PID}" 2>/dev/null || true
            echo "Killed process on port ${PORT} (PID: ${PID})"
        fi
    fi
    exit 0
}

# ============ Status Function ============
check_status() {
    echo "=== Qwen3.5 Server Status ==="
    echo "Port: ${PORT}"
    echo "Log directory: ${LOG_DIR}"

    if [[ -f "${PID_FILE}" ]]; then
        SERVER_PID=$(cat "${PID_FILE}")
        if kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "Server status: RUNNING (PID: ${SERVER_PID})"
        else
            echo "Server status: NOT RUNNING (stale PID file)"
        fi
    else
        # Check by port
        PID=$(lsof -t -i:"${PORT}" 2>/dev/null || true)
        if [[ -n "${PID}" ]]; then
            echo "Server status: RUNNING (PID: ${PID}, no PID file)"
        else
            echo "Server status: NOT RUNNING"
        fi
    fi

    if [[ -f "${ROTATE_PID_FILE}" ]]; then
        ROTATE_PID=$(cat "${ROTATE_PID_FILE}")
        if kill -0 "${ROTATE_PID}" 2>/dev/null; then
            echo "Log rotation: RUNNING (PID: ${ROTATE_PID})"
        else
            echo "Log rotation: NOT RUNNING"
        fi
    fi

    # Show recent log entries
    if [[ -f "${LOG_FILE}" ]]; then
        echo ""
        echo "=== Recent Log (last 10 lines) ==="
        tail -n 10 "${LOG_FILE}"
    fi
    exit 0
}

# Handle stop and status commands
if [[ "${STOP_MODE}" == true ]]; then
    stop_server
fi

if [[ "${STATUS_MODE}" == true ]]; then
    check_status
fi

if [[ "${USE_UV_ENV}" == "1" ]]; then
    if [[ -f "${VENV_PATH}/bin/activate" ]]; then
        source "${VENV_PATH}/bin/activate"
    else
        echo "UV env not found at ${VENV_PATH}"
        exit 1
    fi
fi

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH_BASE}:${LD_LIBRARY_PATH:-}"

# ============ Model Download ============
# Check if model exists, if not download
if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "Model not found at ${MODEL_DIR}"
    echo "Downloading ${MODEL_REPO} from ${HF_ENDPOINT}..."
    mkdir -p "${MODEL_DIR}"
    huggingface-cli download --resume-download "${MODEL_REPO}" --local-dir "${MODEL_DIR}"
    echo "Model downloaded successfully."
else
    echo "Model found at ${MODEL_DIR}"
fi

# ============ Prepare Log Directory ============
# Clean old logs and create log directory
rm -rf "${LOG_DIR}"
mkdir -p "${LOG_DIR}"

# ============ Log Rotation Function ============
log_rotate() {
    echo $$ > "${ROTATE_PID_FILE}"
    while true; do
        sleep 30  # Check every 30 seconds
        if [[ -f "${LOG_FILE}" ]]; then
            size=$(stat -c%s "${LOG_FILE}" 2>/dev/null || echo 0)
            if [[ $size -ge $MAX_LOG_SIZE ]]; then
                # Find next available log number
                num=1
                while [[ -f "${LOG_FILE}.${num}" ]]; do
                    ((num++))
                done
                mv "${LOG_FILE}" "${LOG_FILE}.${num}"
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Log rotated to log.${num}" >> "${LOG_FILE}"
            fi
        fi
    done
}

# ============ Build vllm command ============
# Qwen3.5-27B is a VLM (Vision Language Model), so we include allowed-local-media-path
args=(
  vllm serve "${MODEL_NAME}"
  --host 0.0.0.0
  --port "${PORT}"
  --api-key "${API_KEY}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --performance-mode "${PERFORMANCE_MODE}"
  --dtype auto
  --served-model-name "Qwen/Qwen3.5-27B"
  --reasoning-parser qwen3
  --allowed-local-media-path "${ALLOWED_LOCAL_MEDIA_PATH}"
  --limit-mm-per-prompt '{"image": 1, "video": 1}'
  --enable-prefix-caching
  --enable-chunked-prefill
  --mm-processor-cache-type shm
  --tensor-parallel-size 1
  --default-chat-template-kwargs '{"enable_thinking": false}'
)

if [[ -n "${MM_PROCESSOR_KWARGS}" ]]; then
  args+=(--mm-processor-kwargs "${MM_PROCESSOR_KWARGS}")
fi

echo "Executing: ${args[*]}"
echo "Log directory: ${LOG_DIR}"

# ============ Start Server ============
if [[ "${DAEMON_MODE}" == true ]]; then
    # Daemon mode: run in background with nohup
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting vllm Qwen3.5 server in daemon mode..." | tee "${LOG_FILE}"
    echo "Executing: ${args[*]}" >> "${LOG_FILE}"

    # Start log rotation in background
    nohup bash -c "$(declare -f log_rotate); ROTATE_PID_FILE='${ROTATE_PID_FILE}'; LOG_FILE='${LOG_FILE}'; MAX_LOG_SIZE=${MAX_LOG_SIZE}; log_rotate" > /dev/null 2>&1 &

    # Start vllm server in background
    nohup "${args[@]}" >> "${LOG_FILE}" 2>&1 &
    SERVER_PID=$!
    echo "${SERVER_PID}" > "${PID_FILE}"

    echo ""
    echo "=============================================="
    echo "Qwen3.5 Server started in background (PID: ${SERVER_PID})"
    echo "Log file: ${LOG_FILE}"
    echo "PID file: ${PID_FILE}"
    echo ""
    echo "Commands:"
    echo "  Check status:  $0 --status"
    echo "  Stop server:   $0 --stop"
    echo "  View logs:     tail -f ${LOG_FILE}"
    echo "=============================================="
else
    # Foreground mode: run with tee for terminal output
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting vllm Qwen3.5 server..." | tee "${LOG_FILE}"
    echo "Executing: ${args[*]}" >> "${LOG_FILE}"

    # Start background log rotation
    log_rotate &
    ROTATE_PID=$!

    # Cleanup function
    cleanup() {
        kill "${ROTATE_PID}" 2>/dev/null || true
        rm -f "${ROTATE_PID_FILE}" "${PID_FILE}"
    }
    trap cleanup EXIT INT TERM

    # Execute command and redirect output to log file (also show in terminal)
    "${args[@]}" 2>&1 | tee -a "${LOG_FILE}"
fi