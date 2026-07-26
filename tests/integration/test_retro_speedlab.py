from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import cv2
import stable_retro
import yaml
from datenwissenschaften.checkpoints.model import atomic_save
from datenwissenschaften.configuration.loader import load_config
from datenwissenschaften.models.agent import load_agent
from datenwissenschaften.retro.environment import build_environment
from datenwissenschaften.training.winning_episode_uploader import WinningEpisodeUploader
from loguru import logger

from tests.integration.retro_speedlab.done import Done
from tests.integration.retro_speedlab.explore import ExploreTarget
from tests.integration.retro_speedlab.target import ApproachTarget
from tests.integration.retro_speedlab.wrapper import AirstrikerWrapper

GAME = "Airstriker-Genesis-v0"
SAVESTATE = "Level1"


class UploadHandler(BaseHTTPRequestHandler):
    payloads: list[bytes] = []

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        self.payloads.append(self.rfile.read(length))
        self.send_response(201)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass


def test_training_and_checkpoint_with_retro_speedlab(tmp_path: Path) -> None:
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    server = ThreadingHTTPServer(("127.0.0.1", 0), UploadHandler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    roms = tmp_path / "roms"
    roms.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths": {"roms": str(roms), "models": str(tmp_path / "models")},
                "training": {"game": GAME, "savestate": SAVESTATE, "num_envs": 1},
                "upload": {"url": f"http://127.0.0.1:{server.server_port}", "api_key": "test"},
                "log_level": "INFO",
            }
        ),
        encoding="utf-8",
    )
    emulator = stable_retro.make(GAME, SAVESTATE, render_mode="rgb_array")
    frame, _ = emulator.reset()
    emulator.close()
    template_path = tmp_path / "target.png"
    template = cv2.cvtColor(frame[:32, :32], cv2.COLOR_RGB2GRAY)
    cv2.imwrite(str(template_path), template)
    for state in (ExploreTarget, ApproachTarget, Done):
        state.template_file = str(template_path)
    environment = build_environment(AirstrikerWrapper, config_path)
    try:
        environment.reset()
        states = []
        scores = []
        for _ in range(3):
            _, rewards, _, infos = environment.step([0])
            states.append(infos[0]["state"])
            scores.append(rewards[0])
        assert states == ["ApproachTarget", "Done", "Done"]
        assert scores[0] > 0.0
        assert scores[1] > 0.0
        assert scores[2] == 0.0
        checkpoint = tmp_path / "models" / GAME / SAVESTATE / "model"
        assert checkpoint.parent.joinpath("targets", "ExploreTarget.json").is_file()
        assert checkpoint.parent.joinpath("targets", "ApproachTarget.json").is_file()
        recording = Path(infos[0]["episode_bk2_path"])
        assert recording.is_file()
        uploader = WinningEpisodeUploader(load_config(config_path))
        uploader.locals = {"dones": [True], "infos": [infos[0]]}
        assert uploader._on_step()
        assert not recording.exists()
        assert b'name="bk2_file"' in UploadHandler.payloads[0]
        assert b"metadata_file" not in UploadHandler.payloads[0]
        model = load_agent(environment, checkpoint)
        model.learn(total_timesteps=model.n_steps * model.n_envs)
        atomic_save(model, checkpoint)
        restored = load_agent(environment, checkpoint)
        action, _ = restored.predict(environment.reset())
        assert environment.action_space.contains(action[0])
        assert restored.num_timesteps == model.num_timesteps
        output = "".join(messages)
        assert "Environments ready" in output
        assert "State transition: ExploreTarget -> ApproachTarget" in output
        assert "Remembered ExploreTarget target" in output
        assert "Episode finished:" in output
        assert "Uploaded winning episode" in output
    finally:
        logger.remove(sink)
        environment.close()
        server.shutdown()
        thread.join()
