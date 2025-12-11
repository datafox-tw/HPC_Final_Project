#!/bin/bash

# 四個版本
SCRIPTS=(
    # "tsmixer_base.py"
    # "tsmixer_1_add_precision.py"
    #"tsmixer_2_add_dataloader.py"
    "tsmixer_3_enlarge_512.py"
    "tsmixer_4_enlarge_1024.py"
)

# 跑五次
RUNS=5

# results.csv 若存在，先提醒
if [ -f results.csv ]; then
    echo "[警告] results.csv 已存在，將會 append!"
fi

for script in "${SCRIPTS[@]}"; do
    echo ""
    echo "==============================="
    echo "Running $script ..."
    echo "==============================="

    for ((i=1; i<=RUNS; i++)); do
        echo ">>> Run $i of $script"

        # 設定 RUN_ID 環境變數給 python 用
        RUN_ID=$i

        # 執行 python
        RUN_ID=$i python "$script"

        echo ">>> Done Run $i"
        echo ""
    done
done

echo "所有實驗執行完畢！資料已寫入 results.csv"
#運行：
# chmod +x run_all.sh
# ./run_all.sh
