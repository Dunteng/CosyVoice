import argparse
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Tuple

import torch
import torchaudio

sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel


def _prepare_prompt_audio(prompt_path: str) -> Tuple[str, bool]:
    src = pathlib.Path(prompt_path)
    if not src.exists():
        raise FileNotFoundError(f'Prompt audio not found: {src}')

    try:
        waveform, sample_rate = torchaudio.load(str(src))
    except Exception:
        tmp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp_file.close()
        cmd = ['ffmpeg', '-y', '-i', str(src), '-ac', '1', '-ar', '16000', tmp_file.name]
        run = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if run.returncode != 0:
            os.unlink(tmp_file.name)
            raise RuntimeError(f'Failed to decode prompt audio: {run.stderr}')
        return tmp_file.name, True

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        sample_rate = 16000

    if src.suffix.lower() == '.wav' and waveform.shape[0] == 1 and sample_rate == 16000:
        return str(src), False

    tmp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp_file.close()
    torchaudio.save(tmp_file.name, waveform.to(torch.float32), sample_rate)
    return tmp_file.name, True


def main() -> None:
    parser = argparse.ArgumentParser(description='CosyVoice zero-shot voice cloning')
    parser.add_argument('--model_dir', default='pretrained_models/CosyVoice-300M', help='CosyVoice model directory')
    parser.add_argument('--prompt_wav', required=True, help='Reference speaker audio path (wav/m4a/mp3 supported)')
    parser.add_argument('--prompt_text', required=True, help='Transcript of prompt audio')
    parser.add_argument('--text', required=True, help='Target text to synthesize')
    parser.add_argument('--output', default='outputs/cloned.wav', help='Output wav path')
    parser.add_argument('--stream', action='store_true', help='Enable stream inference')
    args = parser.parse_args()

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_audio_path, should_cleanup = _prepare_prompt_audio(args.prompt_wav)

    try:
        cosyvoice = AutoModel(model_dir=args.model_dir)
        chunks = cosyvoice.inference_zero_shot(
            args.text,
            args.prompt_text,
            prompt_audio_path,
            stream=args.stream,
        )

        first_chunk = next(iter(chunks), None)
        if first_chunk is None:
            raise RuntimeError('No audio generated. Please check input text and prompt audio.')

        torchaudio.save(str(output_path), first_chunk['tts_speech'], cosyvoice.sample_rate)
        print(f'Generated: {output_path.resolve()}')
    finally:
        if should_cleanup and os.path.exists(prompt_audio_path):
            os.unlink(prompt_audio_path)


if __name__ == '__main__':
    main()
