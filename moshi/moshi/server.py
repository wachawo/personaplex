# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Load .env.local to configure HuggingFace cache paths before imports
import os
import argparse
import asyncio
from dataclasses import dataclass
import logging
import random
import tarfile
import time
import secrets
import sys
from typing import Callable, cast, Literal, Optional

import aiohttp
from aiohttp import web
from huggingface_hub import hf_hub_download
import numpy as np
import sentencepiece
import sphn
import torch
import random
from pathlib import Path
from dotenv import load_dotenv

from .models import loaders, MimiModel, LMModel, LMGen
from .utils.logger import setup_logging
from .utils.connection import create_ssl_context, get_lan_ip

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
ENV_LOCAL = PROJECT_ROOT / ".env.local"
# Load .env first (base values)
if ENV_FILE.exists():
    load_dotenv(ENV_FILE, override=False)
    logger.info(f"[.env] Loaded base configuration from {ENV_FILE}")
# Load .env.local second (overrides .env)
if ENV_LOCAL.exists():
    load_dotenv(ENV_LOCAL, override=True)
    logger.info(f"[.env.local] Loaded local overrides from {ENV_LOCAL}")
if not ENV_FILE.exists() and not ENV_LOCAL.exists():
    logger.info(f"[env] No .env or .env.local found in {PROJECT_ROOT}, using defaults")

DeviceString = Literal["cuda"] | Literal["cpu"]  # | Literal["mps"]


def torch_auto_device(requested: Optional[DeviceString] = None) -> torch.device:
    """Return a torch.device based on the requested string or availability."""
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #    return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def wrap_with_system_tags(text: str) -> str:
    """Add system tags as the model expects if they are missing.
    Example: "<system> You enjoy having a good conversation. Have a deep conversation about technology. Your name is Jane. <system>"
    """
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


@dataclass
class ServerState:
    mimi: MimiModel
    other_mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    lock: asyncio.Lock

    def __init__(
        self,
        mimi: MimiModel,
        other_mimi: MimiModel,
        text_tokenizer: sentencepiece.SentencePieceProcessor,
        lm: LMModel,
        device: str | torch.device,
        voice_prompt_dir: str | None = None,
        save_voice_prompt_embeddings: bool = False,
        fp8: bool = False,
    ):
        self.mimi = mimi
        self.other_mimi = other_mimi
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.voice_prompt_dir = voice_prompt_dir
        self._fp8 = fp8
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lm_gen = LMGen(
            lm,
            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
            sample_rate=self.mimi.sample_rate,
            device=device,
            frame_rate=self.mimi.frame_rate,
            save_voice_prompt_embeddings=save_voice_prompt_embeddings,
        )

        self.lock = asyncio.Lock()
        # Pre-allocate pinned CPU buffer for non-blocking DtoH audio transfer
        # mimi.decode output shape: [1, 1, 1920] (1920 samples per frame at 24kHz)
        self._pinned_pcm = torch.empty(1920, dtype=torch.float32, pin_memory=True)
        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

    def warmup(self):
        warmup_dtype = torch.float16 if self._fp8 else torch.float32
        for _ in range(4):
            # chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            chunk = torch.zeros(
                1, 1, self.frame_size, dtype=warmup_dtype, device=self.device
            )
            codes = self.mimi.encode(chunk)
            # Skip other_mimi.encode — its output is always discarded
            # _ = self.other_mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:9])
                # Skip other_mimi.encode — its output is always discarded
                # _ = self.other_mimi.decode(tokens[:, 1:9])

        if self.device.type == "cuda":
            torch.cuda.synchronize()

    async def handle_chat(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        peer = request.remote  # IP
        peer_port = request.transport.get_extra_info("peername")[1]  # Port
        logger.info("Incoming connection from %s:%s", peer, peer_port)

        # self.lm_gen.temp = float(request.query["audio_temperature"])
        # self.lm_gen.temp_text = float(request.query["text_temperature"])
        # self.lm_gen.top_k_text = max(1, int(request.query["text_topk"]))
        # self.lm_gen.top_k = max(1, int(request.query["audio_topk"]))

        # Construct full voice prompt path
        requested_voice_prompt_path = None
        voice_prompt_path = None
        if self.voice_prompt_dir is not None:
            voice_prompt_filename = request.query["voice_prompt"]
            requested_voice_prompt_path = None
            if voice_prompt_filename is not None:
                requested_voice_prompt_path = os.path.join(
                    self.voice_prompt_dir, voice_prompt_filename
                )
            # If the voice prompt file does not exist, find a valid (s0) voiceprompt file in the directory
            if requested_voice_prompt_path is None or not os.path.exists(
                requested_voice_prompt_path
            ):
                raise FileNotFoundError(
                    f"Requested voice prompt '{voice_prompt_filename}' not found in '{self.voice_prompt_dir}'"
                )
            else:
                voice_prompt_path = requested_voice_prompt_path

        if self.lm_gen.voice_prompt != voice_prompt_path:
            if voice_prompt_path.endswith(".pt"):
                # Load pre-saved voice prompt embeddings
                self.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
            else:
                self.lm_gen.load_voice_prompt(voice_prompt_path)
        self.lm_gen.text_prompt_tokens = (
            self.text_tokenizer.encode(
                wrap_with_system_tags(request.query["text_prompt"])
            )
            if len(request.query["text_prompt"]) > 0
            else None
        )
        seed = int(request["seed"]) if "seed" in request.query else None

        async def recv_loop():
            nonlocal close
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        logger.error("%s", ws.exception())
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSED:
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSE:
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        logger.error("unexpected message type %s", message.type)
                        continue
                    message = message.data
                    if not isinstance(message, bytes):
                        logger.error("unsupported message type %s", type(message))
                        continue
                    if len(message) == 0:
                        logger.warning("empty message")
                        continue
                    kind = message[0]
                    if kind == 1:  # audio
                        payload = message[1:]
                        opus_reader.append_bytes(payload)
                    else:
                        logger.warning("unknown message kind %s", kind)
            finally:
                close = True
                logger.info("connection closed")

        async def opus_loop():
            all_pcm_data = None
            frame_count = 0
            profile_interval = 100
            input_dtype = torch.float16 if self._fp8 else torch.float32

            # Accumulators for averaging over profile_interval frames
            acc_total = 0.0
            acc_prep = 0.0
            acc_enc = 0.0
            acc_step = 0.0
            acc_dec = 0.0
            acc_codes = 0
            acc_frames = 0

            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                pcm = opus_reader.read_pcm()
                if pcm.shape[-1] == 0:
                    continue
                if all_pcm_data is None:
                    all_pcm_data = pcm
                else:
                    all_pcm_data = np.concatenate((all_pcm_data, pcm))
                while all_pcm_data.shape[-1] >= self.frame_size:
                    torch.cuda.synchronize()
                    frame_start = time.monotonic()
                    chunk = all_pcm_data[: self.frame_size]
                    all_pcm_data = all_pcm_data[self.frame_size :]
                    chunk = torch.from_numpy(chunk)
                    # chunk = chunk.to(device=self.device)[None, None]
                    chunk = chunk.to(device=self.device, dtype=input_dtype)[None, None]
                    torch.cuda.synchronize()
                    prep_end = time.monotonic()
                    codes = self.mimi.encode(chunk)
                    # Skip other_mimi.encode — output always discarded (saves ~6ms)
                    # _ = self.other_mimi.encode(chunk)
                    torch.cuda.synchronize()
                    enc_end = time.monotonic()
                    lm_elapsed = 0.0
                    dec_elapsed = 0.0
                    codes_n = 0
                    for c in range(codes.shape[-1]):
                        torch.cuda.synchronize()
                        lm_start = time.monotonic()
                        tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                        torch.cuda.synchronize()
                        lm_end = time.monotonic()
                        lm_elapsed += lm_end - lm_start
                        if tokens is None:
                            continue
                        codes_n += 1
                        assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
                        torch.cuda.synchronize()
                        dec_start = time.monotonic()
                        main_pcm = self.mimi.decode(tokens[:, 1:9])
                        # Skip other_mimi.decode — output always discarded (saves ~4ms)
                        # _ = self.other_mimi.decode(tokens[:, 1:9])
                        torch.cuda.synchronize()
                        dec_end = time.monotonic()
                        dec_elapsed += dec_end - dec_start
                        # main_pcm = main_pcm.cpu()
                        # opus_writer.append_pcm(main_pcm[0, 0].numpy())
                        # Pinned memory DtoH transfer (saves ~2ms vs .cpu())
                        main_pcm = main_pcm.float()
                        self._pinned_pcm.copy_(main_pcm[0, 0], non_blocking=True)
                        torch.cuda.current_stream().synchronize()
                        opus_writer.append_pcm(self._pinned_pcm.numpy())
                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            _text = self.text_tokenizer.id_to_piece(text_token)  # type: ignore
                            _text = _text.replace("▁", " ")
                            msg = b"\x02" + bytes(_text, encoding="utf8")
                            await ws.send_bytes(msg)
                        # else:
                        #    text_token_map = ['EPAD', 'BOS', 'EOS', 'PAD']
                    frame_end = time.monotonic()

                    # Accumulate per-frame timings (seconds → milliseconds)
                    acc_total += (frame_end - frame_start) * 1000
                    acc_prep += (prep_end - frame_start) * 1000
                    acc_enc += (enc_end - prep_end) * 1000
                    acc_step += lm_elapsed * 1000
                    acc_dec += dec_elapsed * 1000
                    acc_codes += codes_n
                    acc_frames += 1

                    frame_count += 1

                    if acc_frames >= profile_interval:
                        n = acc_frames
                        avg_total = acc_total / n
                        avg_prep = acc_prep / n
                        avg_enc = acc_enc / n
                        avg_step = acc_step / n
                        avg_dec = acc_dec / n
                        avg_other = avg_total - avg_prep - avg_enc - avg_step - avg_dec
                        avg_codes = acc_codes / n
                        frame_dur_ms = self.frame_size / self.mimi.sample_rate * 1000
                        message = (
                            f"frames={frame_count - n}‥{frame_count - 1} "
                            f"dur={frame_dur_ms:.0f}ms avg_total={avg_total:.1f}ms "
                            f"prep={avg_prep:.1f} enc={avg_enc:.1f} "
                            f"lm={avg_step:.1f} dec={avg_dec:.1f} "
                            f"other={avg_other:.1f} codes={avg_codes:.1f}"
                        )
                        if avg_total > frame_dur_ms:
                            logger.warning(message)
                        else:
                            logger.info(message)
                        # Reset accumulators
                        acc_total = 0.0
                        acc_prep = 0.0
                        acc_enc = 0.0
                        acc_step = 0.0
                        acc_dec = 0.0
                        acc_codes = 0
                        acc_frames = 0

        async def send_loop():
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    await ws.send_bytes(b"\x01" + msg)

        logger.info("accepted connection")
        if len(request.query["text_prompt"]) > 0:
            logger.info("text prompt: %s", request.query["text_prompt"])
        if len(request.query["voice_prompt"]) > 0:
            logger.info(
                "voice prompt: %s (requested: %s)",
                voice_prompt_path,
                requested_voice_prompt_path,
            )
        close = False
        async with self.lock:
            if seed is not None and seed != -1:
                seed_all(seed)

            opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)
            self.mimi.reset_streaming()
            self.other_mimi.reset_streaming()
            self.lm_gen.reset_streaming()

            async def is_alive():
                if close or ws.closed:
                    return False
                try:
                    # Check for disconnect without waiting too long
                    msg = await asyncio.wait_for(ws.receive(), timeout=0.01)
                    if msg.type in (
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                    ):
                        return False
                except asyncio.TimeoutError:
                    # No messages → client probably still alive
                    return True
                except aiohttp.ClientConnectionError:
                    return False
                return True

            # Reuse mimi for encoding voice prompt and then reset it before conversation starts
            await self.lm_gen.step_system_prompts_async(self.mimi, is_alive=is_alive)
            self.mimi.reset_streaming()
            logger.info("done with system prompts")
            # Send the handshake.
            if await is_alive():
                await ws.send_bytes(b"\x00")
                logger.info("sent handshake bytes")
                # Clean cancellation manager
                tasks = [
                    asyncio.create_task(recv_loop()),
                    asyncio.create_task(opus_loop()),
                    asyncio.create_task(send_loop()),
                ]
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                # Force-kill remaining tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                await ws.close()
                logger.info("session closed")
                # await asyncio.gather(opus_loop(), recv_loop(), send_loop())
        logger.info("done with connection")
        return ws


def _get_voice_prompt_dir(
    voice_prompt_dir: Optional[str], hf_repo: str
) -> Optional[str]:
    """
    If voice_prompt_dir is None:
      - download voices.tgz from HF
      - extract it once
      - return extracted directory
    If voice_prompt_dir is provided:
      - just return it
    """
    if voice_prompt_dir is not None:
        return voice_prompt_dir

    logger.info("retrieving voice prompts")

    voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
    voices_tgz = Path(voices_tgz)
    voices_dir = voices_tgz.parent / "voices"

    if not voices_dir.exists():
        logger.info(f"extracting {voices_tgz} to {voices_dir}")
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=voices_tgz.parent)

    if not voices_dir.exists():
        raise RuntimeError("voices.tgz did not contain a 'voices/' directory")

    return str(voices_dir)


def _get_static_path(static: Optional[str]) -> Optional[str]:
    if static is None:
        logger.info("retrieving the static content")
        dist_tgz = hf_hub_download("nvidia/personaplex-7b-v1", "dist.tgz")
        dist_tgz = Path(dist_tgz)
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=dist_tgz.parent)
        return str(dist)
    elif static != "none":
        # When set to the "none" string, we don't serve any static content.
        return static
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level (default: INFO)",
    )
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument(
        "--gradio-tunnel", action="store_true", help="Activate a gradio tunnel."
    )
    parser.add_argument(
        "--gradio-tunnel-token",
        help="Provide a custom (secret) token here to keep getting the same URL.",
    )

    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument(
        "--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi."
    )
    parser.add_argument(
        "--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi."
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=loaders.DEFAULT_REPO,
        help="HF repo to look into, defaults PersonaPlex. "
        "Use this to select a different pre-trained model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device on which to run, defaults to 'cuda'.",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Offload LM model layers to CPU when GPU memory is insufficient. "
        "Requires 'accelerate' package.",
    )
    parser.add_argument(
        "--voice-prompt-dir",
        type=str,
        help=(
            "Directory containing voice prompt files. "
            "If omitted, voices.tgz is downloaded from HF and extracted."
            "Voice prompt filenames from client requests will be joined with this directory path."
        ),
    )
    parser.add_argument(
        "--ssl",
        type=str,
        help=(
            "use https instead of http, this flag should point to a directory "
            "that contains valid key.pem and cert.pem files"
        ),
    )
    parser.add_argument(
        "--fp8",
        action="store_true",
        help="Quantize LM weights to FP8 for ~1.4x faster inference (requires SM >= 89)",
    )

    args = parser.parse_args()
    setup_logging(level=getattr(logging, args.log_level))
    args.voice_prompt_dir = _get_voice_prompt_dir(
        args.voice_prompt_dir,
        args.hf_repo,
    )
    if args.voice_prompt_dir is not None:
        assert os.path.exists(
            args.voice_prompt_dir
        ), f"Directory missing: {args.voice_prompt_dir}"
    logger.info(f"voice_prompt_dir = {args.voice_prompt_dir}")

    static_path: None | str = _get_static_path(args.static)
    assert static_path is None or os.path.exists(
        static_path
    ), f"Static path does not exist: {static_path}."
    logger.info(f"static_path = {static_path}")
    device: torch.device = torch_auto_device(args.device)

    seed_all(42424242)

    setup_tunnel: Optional[Callable] = None
    tunnel_token = ""
    if args.gradio_tunnel:
        try:
            from gradio import networking  # type: ignore
        except ImportError:
            logger.error(
                "Cannot find gradio which is required to activate a tunnel. "
                "Please install with `pip install gradio`."
            )
            sys.exit(1)
        setup_tunnel = networking.setup_tunnel
        if args.gradio_tunnel_token is None:
            tunnel_token = secrets.token_urlsafe(32)
        else:
            tunnel_token = args.gradio_tunnel_token

    # Download config.json to increment download counter
    # No worries about double-counting since config.json will be cached the second time
    hf_hub_download(args.hf_repo, "config.json")

    logger.info("loading mimi")
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(args.mimi_weight, device)
    other_mimi = loaders.get_mimi(args.mimi_weight, device)
    if args.fp8:
        # FP16 mimi + torch.compile for faster encode/decode (~3ms saved)
        mimi = mimi.half()
        other_mimi = other_mimi.half()
        mimi.torch_compile_encoder_decoder = True
        mimi = torch.compile(mimi)
        logger.info("mimi loaded (FP16 + compiled)")
    else:
        logger.info("mimi loaded")

    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)  # type: ignore

    logger.info("loading moshi")
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(
        args.moshi_weight, device=device, cpu_offload=args.cpu_offload
    )
    lm.eval()
    if args.fp8:
        from .fp8_quantize import quantize_model

        logger.info("applying FP8 quantization...")
        quantize_model(lm)
        logger.info("FP8 quantization complete")
    logger.info("moshi loaded")
    state = ServerState(
        mimi=mimi,
        other_mimi=other_mimi,
        text_tokenizer=text_tokenizer,
        lm=lm,
        device=device,
        voice_prompt_dir=args.voice_prompt_dir,
        save_voice_prompt_embeddings=False,
        fp8=args.fp8,
    )
    logger.info("warming up the model")
    state.warmup()
    if args.fp8:
        from .fp8_quantize import free_bf16_inproj

        free_bf16_inproj(lm)
    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)
    if static_path is not None:

        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))

        logger.info(f"serving static content from {static_path}")
        app.router.add_get("/", handle_root)
        app.router.add_static(
            "/", path=static_path, follow_symlinks=True, name="static"
        )
    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        ssl_context, protocol = create_ssl_context(args.ssl)
    host_ip = (
        args.host if args.host not in ("0.0.0.0", "::", "localhost") else get_lan_ip()
    )
    logger.info(f"Access the Web UI directly at {protocol}://{host_ip}:{args.port}")
    if setup_tunnel is not None:
        tunnel = cast(Callable, setup_tunnel)("localhost", args.port, tunnel_token, None)
        logger.info(
            f"Tunnel started, if executing on a remote GPU, you can use {tunnel}."
        )
    web.run_app(app, port=args.port, ssl_context=ssl_context)


with torch.no_grad():
    main()
