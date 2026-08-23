#!/bin/zsh
set -e

cd /Users/lab4099/Desktop/Mujoco/task2_handover
export QT_QPA_PLATFORM=cocoa
export QT_MAC_WANTS_LAYER=1
export GZ_PARTITION=warehouse_preview
export GZ_IP=127.0.0.1
export GZ_SIM_RESOURCE_PATH=/Users/lab4099/Desktop/Mujoco/work/src

WORLD=/Users/lab4099/Desktop/Mujoco/work/src/warehouse_bringup/worlds/two_floor_warehouse.sdf
pixi run gz sim -s -r -v 3 "$WORLD" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null' EXIT INT TERM
sleep 3
pixi run gz sim -g -v 3 --render-engine-gui-api-backend opengl
