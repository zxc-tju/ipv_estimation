import stat
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
SCRIPT_SPECS = {
    "launch_claude.command": "claude",
    "launch_codex.command": "codex",
}


class ShortcutScriptTests(unittest.TestCase):
    def test_shortcut_scripts_exist_and_are_executable(self):
        for script_name in SCRIPT_SPECS:
            script_path = SCRIPTS_DIR / script_name

            self.assertTrue(script_path.exists(), f"{script_path} is missing")
            self.assertTrue(
                script_path.stat().st_mode & stat.S_IXUSR,
                f"{script_path} must be user-executable for double-click launch",
            )

    def test_shortcut_scripts_cd_to_project_before_starting_cli(self):
        for script_name, cli_command in SCRIPT_SPECS.items():
            script_path = SCRIPTS_DIR / script_name
            script_text = script_path.read_text(encoding="utf-8")

            self.assertTrue(script_text.startswith("#!/bin/zsh"))
            self.assertIn(f'PROJECT_DIR="{ROOT}"', script_text)
            self.assertIn(f'CLI_COMMAND="{cli_command}"', script_text)
            self.assertIn('command -v "$CLI_COMMAND"', script_text)
            self.assertLess(
                script_text.index('cd "$PROJECT_DIR"'),
                script_text.index('exec "$CLI_COMMAND"'),
            )
            self.assertIn("按回车关闭这个窗口", script_text)


if __name__ == "__main__":
    unittest.main()
