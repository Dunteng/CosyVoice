import argparse
import os
import pathlib

import torchaudio

from clone_tts import _prepare_prompt_audio

import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel


def _read_texts(input_txt: str) -> list[str]:
    lines = pathlib.Path(input_txt).read_text(encoding='utf-8').splitlines()
    texts = []
    for line in lines:
        text = line.strip()
        if not text or text.startswith('#'):
            continue
        texts.append(text)
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description='Batch zero-shot voice cloning with CosyVoice')
    parser.add_argument('--model_dir', default='pretrained_models/CosyVoice-300M', help='CosyVoice model directory')
    parser.add_argument('--prompt_wav', required=True, help='Reference speaker audio path (wav/m4a/mp3 supported)')
    parser.add_argument('--prompt_text', required=True, help='Transcript of prompt audio')
    parser.add_argument('--input_txt', required=True, help='UTF-8 text file, one sentence per line')
    parser.add_argument('--output_dir', default='outputs/batch', help='Output directory')
    parser.add_argument('--prefix', default='line', help='Output file prefix')
    parser.add_argument('--start_index', type=int, default=1, help='Start index for file naming')
    parser.add_argument('--stream', action='store_true', help='Enable stream inference')
    args = parser.parse_args()

    texts = _read_texts(args.input_txt)
    if not texts:
        raise ValueError('No valid text found in input_txt.')

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_audio_path, should_cleanup = _prepare_prompt_audio(args.prompt_wav)

    try:
        cosyvoice = AutoModel(model_dir=args.model_dir)
        for i, text in enumerate(texts, start=args.start_index):
            chunks = cosyvoice.inference_zero_shot(
                text,
                args.prompt_text,
                prompt_audio_path,
                stream=args.stream,
            )
            first_chunk = next(iter(chunks), None)
            if first_chunk is None:
                print(f'Skipped (no audio): {text}')
                continue

            output_path = output_dir / f'{args.prefix}_{i:03d}.wav'
            torchaudio.save(str(output_path), first_chunk['tts_speech'], cosyvoice.sample_rate)
            print(f'Generated: {output_path.resolve()} | text={text}')
    finally:
        if should_cleanup and os.path.exists(prompt_audio_path):
            os.unlink(prompt_audio_path)


if __name__ == '__main__':
    main()
