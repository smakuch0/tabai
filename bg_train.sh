#!/bin/bash

mkdir -p logs
mkdir -p b/checkpoints

. venv/bin/activate

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="logs/train_${TIMESTAMP}.log"

echo "Starting training at $(date)" | tee -a $LOGFILE
echo "Logs: $LOGFILE" | tee -a $LOGFILE
echo "Checkpoints: b/checkpoints/" | tee -a $LOGFILE
echo "================================================" | tee -a $LOGFILE
echo "" | tee -a $LOGFILE

nohup python3 -m b.train \
    --data b/data/GuitarSet \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --checkpoint_dir b/checkpoints \
    >> $LOGFILE 2>&1 &

PID=$!
echo $PID > logs/train.pid
echo "Training started with PID: $PID" | tee -a $LOGFILE
echo "Monitor: tail -f $LOGFILE"
echo "Stop: kill $(cat logs/train.pid)"
