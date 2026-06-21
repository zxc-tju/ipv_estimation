#!/bin/zsh

PROJECT_DIR="/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation"
CLI_COMMAND="claude"

export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$HOME/bin:$HOME/.npm-global/bin:$HOME/.cargo/bin:$PATH"

for profile_file in "$HOME/.zprofile" "$HOME/.zshrc"; do
  if [[ -r "$profile_file" ]]; then
    source "$profile_file"
  fi
done

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "没有找到项目目录："
  echo "$PROJECT_DIR"
  echo
  echo "按回车关闭这个窗口。"
  read -r _
  exit 1
fi

cd "$PROJECT_DIR" || {
  echo "无法进入项目目录："
  echo "$PROJECT_DIR"
  echo
  echo "按回车关闭这个窗口。"
  read -r _
  exit 1
}

if ! command -v "$CLI_COMMAND" >/dev/null 2>&1; then
  echo "没有找到 $CLI_COMMAND 命令。"
  echo "请确认 Claude CLI 已安装，并且能在普通终端中直接运行：$CLI_COMMAND"
  echo
  echo "当前目录：$PWD"
  echo "当前 PATH：$PATH"
  echo
  echo "按回车关闭这个窗口。"
  read -r _
  exit 127
fi

echo "已进入项目：$PWD"
echo "启动 $CLI_COMMAND ..."
echo
exec "$CLI_COMMAND"
