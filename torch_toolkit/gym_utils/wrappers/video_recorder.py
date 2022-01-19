__all__ = ['RecordVideoWithText']

import distutils.version
import distutils.spawn
import os
from typing import Callable

import numpy as np
from gym import logger, error
from gym.wrappers import RecordVideo
from gym.wrappers.monitoring.video_recorder import VideoRecorder, ImageEncoder
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
try:
    fnt = ImageFont.truetype((Path(__file__).resolve().parent / 'Inconsolata-Bold.ttf').as_posix(), 10)  # Try to default to a monospace font
except:
    fnt = ImageFont.load_default()

class RecordVideoWithText(RecordVideo):
    def __init__(
        self,
        env,
        video_folder: str,
        episode_trigger: Callable[[int], bool] = None,
        step_trigger: Callable[[int], bool] = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        text_space: int = 0
    ):
        super().__init__(env, video_folder, episode_trigger, step_trigger, video_length, name_prefix)
        self.text_space = text_space
        self.text_to_render = ''

    def start_video_recorder(self):
        self.close_video_recorder()

        video_name = f"{self.name_prefix}-step-{self.step_id}"
        if self.episode_trigger:
            video_name = f"{self.name_prefix}-episode-{self.episode_id}"

        base_path = os.path.join(self.video_folder, video_name)
        self.video_recorder = VideoRecorderWithText(
            env=self.env,
            base_path=base_path,
            metadata={"step_id": self.step_id, "episode_id": self.episode_id},
            text_space=self.text_space
        )

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True

    def step(self, action):
        observations, rewards, dones, infos = super(RecordVideo, self).step(action)
        # increment steps and episodes
        # self.step_id += 1
        if not self.is_vector_env:
            self.step_id += 1
            if dones:
                self.episode_id += 1
        else:
            self.step_id += self.num_envs
            if dones[0]: self.episode_id += 1
        # elif dones[0]:
        #     self.episode_id += 1

        if self.recording:
            self.video_recorder.capture_frame(self.text_to_render)
            self.recorded_frames += 1
            if self.video_length > 0:
                if self.recorded_frames > self.video_length:
                    self.close_video_recorder()
            else:
                if not self.is_vector_env:
                    if dones:
                        self.close_video_recorder()
                elif dones[0]:
                    self.close_video_recorder()

        elif self._video_enabled():
            self.start_video_recorder()

        return observations, rewards, dones, infos

    def set_render_text(self, text: str = ''):
        self.text_to_render = text


class VideoRecorderWithText(VideoRecorder):
    def __init__(self, env, path=None, metadata=None, enabled=True, base_path=None, text_space: int = 40):
        super().__init__(env, path, metadata, enabled, base_path)
        self.text_space = text_space

    def capture_frame(self, text: str = ''):
        """Render the given `env` and add the resulting frame to the video."""
        if not self.functional:
            return
        if self._closed:
            logger.warn(
                "The video recorder has been closed and no frames will be captured anymore."
            )
            return
        logger.debug("Capturing video frame: path=%s", self.path)

        render_mode = "ansi" if self.ansi_mode else "rgb_array"
        frame = self.env.render(mode=render_mode)

        if frame is None:
            if self._async:
                return
            else:
                # Indicates a bug in the environment: don't want to raise
                # an error here.
                logger.warn(
                    "Env returned None on render(). Disabling further rendering for video recorder by marking as disabled: path=%s metadata_path=%s",
                    self.path,
                    self.metadata_path,
                )
                self.broken = True
        else:
            self.last_frame = frame
            if self.ansi_mode:
                self._encode_ansi_frame(frame)
            else:
                self._encode_image_frame(frame, text)

    def _encode_image_frame(self, frame, text: str = ''):
        if not self.encoder:
            self.encoder = TextImageEncoder(
                self.path, frame.shape, self.frames_per_sec, self.output_frames_per_sec, self.text_space
            )
            self.metadata["encoder_version"] = self.encoder.version_info

        try:
            self.encoder.capture_frame(frame, text)
        except error.InvalidFrame as e:
            logger.warn("Tried to pass invalid video frame, marking as broken: %s", e)
            self.broken = True
        else:
            self.empty = False


class TextImageEncoder(ImageEncoder):
    def __init__(self, output_path, frame_shape, frames_per_sec, output_frames_per_sec, text_space_in_pixels: int = 40):
        frame_shape = (frame_shape[0] + text_space_in_pixels, frame_shape[1], frame_shape[2])  # Expand frame to fit text
        self.text_space = text_space_in_pixels
        super().__init__(output_path, frame_shape, frames_per_sec, output_frames_per_sec)
        self.text_h = self.wh[-1] - self.text_space

    def capture_frame(self, frame, text: str = ''):
        if not isinstance(frame, (np.ndarray, np.generic)):
            raise error.InvalidFrame(
                "Wrong type {} for {} (must be np.ndarray or np.generic)".format(
                    type(frame), frame
                )
            )
        if self.text_space:  # Always add the space if we're using it
            frame = np.concatenate((frame, np.zeros((self.text_space, *self.frame_shape[1:]), dtype=np.uint8)), axis=0)  # Add space

        if frame.shape != self.frame_shape:
            raise error.InvalidFrame(
                "Your frame has shape {}, but the VideoRecorder is configured for shape {}.".format(
                    frame.shape, self.frame_shape
                )
            )
        if frame.dtype != np.uint8:
            raise error.InvalidFrame(
                "Your frame has data type {}, but we require uint8 (i.e. RGB values from 0-255).".format(
                    frame.dtype
                )
            )

        if text and self.text_space:  # Draw text if any is provided and we have space
            frame = Image.fromarray(frame)  # Convert to Pillow image
            dr = ImageDraw.Draw(frame)  # In-place drawing context
            dr.text((10, self.text_h), text, font=fnt)  # Draw text

        try:
            if distutils.version.LooseVersion(
                np.__version__
            ) >= distutils.version.LooseVersion("1.9.0") or isinstance(frame, Image.Image):
                self.proc.stdin.write(frame.tobytes())
            else:
                self.proc.stdin.write(frame.tostring())
        except Exception as e:
            stdout, stderr = self.proc.communicate()
            logger.error("VideoRecorder encoder failed: %s", stderr)