#!/bin/bash

if [ ! -f logs/train.pid ]; then
    echo "No training process found (logs/train.pid missing)"
    exit 1
fi

PID=$(cat logs/train.pid)

if ps -p $PID > /dev/null; then
    echo "  Training running (PID: $PID)"
    echo ""
    
    LATEST_LOG=$(ls -t logs/train_*.log | head -1)
    echo "Latest log: $LATEST_LOG"
    echo "================================================"
    tail -20 $LATEST_LOG
    echo
    echo "================================================"
    echo ""
    echo "Commands:"
    echo "  Monitor live: tail -f $LATEST_LOG"
    echo "  Stop: kill $PID"
    echo "  Check GPU: nvidia-smi"
else
    echo "   Training not running (PID $PID not found)"
    rm logs/train.pid
    exit 1
fi
