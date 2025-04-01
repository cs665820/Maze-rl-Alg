from gymnasium.utils.env_checker import check_env

from MazeEnv import MazeEnv

if __name__ == "__main__":
    check_env(MazeEnv(), skip_render_check=True, skip_close_check=True)