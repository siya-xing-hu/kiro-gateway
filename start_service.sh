#!/bin/bash

# 使用 $HOME 变量，systemd 会将其替换为指定用户的家目录
cd "$HOME/kiro-gateway" || exit 1

# 更新代码
git pull

# conda 环境路径（同样使用 $HOME）
ENV_PATH="$HOME/miniconda3/envs/kiro-gateway"

# 安装依赖
"$ENV_PATH/bin/pip" install -r requirements.txt

# 运行主程序
"$ENV_PATH/bin/python" main.py