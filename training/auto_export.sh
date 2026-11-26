#!/bin/bash
# Auto-export the best checkpoint from freedoom_focus run to submission weights
# Run periodically to keep submitted weights up to date with training progress

cd /home/ayush/Documents/devforge/devforge-doom-template
source .venv/bin/activate

RUN_DIR="${1:-training/runs/v2_from_361}"
EXPORT_PATH="${2:-submission/weights/policy.pt}"

if [ ! -f "$RUN_DIR/latest.pt" ]; then
    echo "No checkpoint found in $RUN_DIR"
    exit 1
fi

# Export latest checkpoint
python3 -c "
import torch, shutil, os
ck = torch.load('$RUN_DIR/latest.pt', map_location='cpu', weights_only=False)
config = ck['config']
update = ck['update']
env_steps = ck['env_steps']
export = {
    'model': {k: v.half().cpu() for k, v in ck['model'].items()},
    'config': config,
    'meta': {'update': update, 'env_steps': env_steps}
}
torch.save(export, '$EXPORT_PATH.tmp')
os.replace('$EXPORT_PATH.tmp', '$EXPORT_PATH')
print(f'Exported update {update}, {env_steps:,} steps -> $EXPORT_PATH')
print(f'Config: {config}')
"
