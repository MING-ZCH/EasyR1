"""
Eval SFT base model on training dataset (parquet) to identify easy samples.
- Reads parquet training data with embedded images
- Runs interleaved multi-round point-to-count evaluation 
- Identifies samples where the base model gets correct answers (easy samples)
- Outputs JSON list of easy sample indices for future dataset balancing
- Supports 2-GPU parallel evaluation via multiprocessing
- Memory-efficient: each worker loads only its own image data on-demand

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python eval_sft_base_on_trainset.py
"""

import os
import io
import json
import re
import time
import glob
import torch
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoConfig, PretrainedConfig
from typing import List, Dict, Tuple


# ==================== Config ====================
class Config:
    # SFT base model (same as training base)
    MODEL_PATH = '/data/workspace/hyleochang/EasyR1-latest/save/StepCount-7B-SFT-30k_v24_mask_reward_v4bok_grpo_hm0_gateoff_bok_grpo_20260412_0456/global_step_213/actor/huggingface'

    # Training data (parquet with embedded images)
    TRAIN_DATA_DIR = '/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/datasets/StepCountQA-RL-Traj_0_10_hard_only/data'

    # Output
    OUTPUT_DIR = '/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/datasets/eval_rl_base_trainset'
    OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, 'eval_rl_base_trainset_v24.json')
    OUTPUT_EASY_IDS_PATH = os.path.join(OUTPUT_DIR, 'easy_sample_ids_v24.json')

    # Eval parameters (match training config)
    HISTORY_MODE = 0
    MAX_NEW_TOKENS = 8192
    MAX_ROUNDS = 11
    MAX_PIXELS = 12845056

    # Load prompts from training prompt files to guarantee exact match
    _PROMPT_DIR = '/data/workspace/hyleochang/EasyR1-latest/examples/format_prompt'

    @classmethod
    def _load_prompt(cls, filename):
        path = os.path.join(cls._PROMPT_DIR, filename)
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    @classmethod
    def get_system_prompt(cls):
        return cls._load_prompt('StepCount_interleaved_system_prompt.txt')

    @classmethod
    def get_first_turn_prompt(cls):
        return cls._load_prompt('StepCount_interleaved_first_turn_prompt.txt')

    @classmethod
    def get_process_prompt(cls):
        return cls._load_prompt('StepCount_interleaved_process_prompt.txt')

    NUM_GPUS = 4


# ==================== Data Loading ====================
def load_sample_metadata(data_dir):
    """Load only metadata (question, answer, file location) without image bytes.
    Returns list of lightweight dicts with parquet file path and row index for lazy loading.
    """
    parquet_files = sorted(glob.glob(os.path.join(data_dir, '*.parquet')))
    print(f"Found {len(parquet_files)} parquet files in {data_dir}")

    samples = []
    global_idx = 0
    for pf in parquet_files:
        print(f"  Scanning {os.path.basename(pf)}...")
        # Only read text columns, skip image bytes
        table = pq.read_table(pf, columns=['problem', 'answer'])
        n_rows = table.num_rows

        for i in range(n_rows):
            problem = table['problem'][i].as_py()
            answer = table['answer'][i].as_py()

            question = problem.replace('<image>', '').strip()
            if question.startswith('\n'):
                question = question[1:].strip()

            samples.append({
                'id': f'train_{global_idx}',
                'question': question,
                'answer': str(answer).strip(),
                'parquet_path': pf,
                'row_index': i,
            })
            global_idx += 1

        del table

    print(f"Total training samples: {len(samples)}")
    return samples


# Global cache for parquet image tables (per-process)
_parquet_cache = {}

def load_image_bytes(parquet_path, row_index):
    """Load image bytes from parquet with per-file caching.
    Each parquet file (~1GB) is loaded once and cached in memory.
    """
    global _parquet_cache
    if parquet_path not in _parquet_cache:
        print(f"  [Cache] Loading images from {os.path.basename(parquet_path)}...")
        _parquet_cache[parquet_path] = pq.read_table(parquet_path, columns=['images'])
    table = _parquet_cache[parquet_path]
    images_col = table['images'][row_index].as_py()
    if images_col and len(images_col) > 0:
        return images_col[0].get('bytes', None)
    return None


# ==================== ModelInference ====================
class ModelInference:
    def __init__(self, model_path, device_id):
        device = f"cuda:{device_id}"
        print(f"Loading model to {device}: {model_path}")

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.processor.tokenizer.padding_side = "left"

        if hasattr(self.processor, 'image_processor') and hasattr(self.processor.image_processor, 'max_pixels'):
            self.max_pixels = self.processor.image_processor.max_pixels
        else:
            self.max_pixels = 262144

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.use_cache = True
        if hasattr(config, 'text_config'):
            if isinstance(config.text_config, dict):
                config.text_config = PretrainedConfig(**config.text_config)
            config.text_config.use_cache = True

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16,
            device_map=device,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).eval()

        if hasattr(self.model, 'generation_config') and self.model.generation_config is not None:
            self.model.generation_config.use_cache = True

        print(f"Model loaded on {device}, max_pixels={self.max_pixels}")

    def calculate_resized_dimensions(self, width, height):
        total_pixels = width * height
        if total_pixels <= self.max_pixels:
            return (width // 28) * 28, (height // 28) * 28
        scale = (self.max_pixels / total_pixels) ** 0.5
        new_w = int(width * scale)
        new_h = int(height * scale)
        return (new_w // 28) * 28, (new_h // 28) * 28

    def generate_response(self, text, images, max_new_tokens=2048):
        if images:
            ow, oh = images[0].size
            original_size = (ow, oh)
            rw, rh = self.calculate_resized_dimensions(ow, oh)
            resized_size = (rw, rh)
        else:
            original_size = (0, 0)
            resized_size = (0, 0)

        inputs = self.processor(
            text=[text],
            images=images if images else None,
            return_tensors="pt",
            padding=True
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        del inputs, generated_ids, generated_ids_trimmed
        return response, original_size, resized_size


# ==================== ImageAnnotator ====================
class ImageAnnotator:
    @staticmethod
    def annotate_image_obj(image, points, accumulated_count=0, original_size=None, model_size=None):
        """Annotate image in-memory. Returns annotated PIL Image."""
        img = image.copy()
        draw = ImageDraw.Draw(img)
        img_width, img_height = img.size

        if original_size:
            img_width, img_height = original_size

        if not model_size:
            model_size = (1024, 1024)
        model_width, model_height = model_size

        scale = img_width / model_width if model_width > 0 else 1.0

        dot_radius = max(4, round(10 * scale))
        font_size = max(10, round(20 * scale))

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        for idx, point in enumerate(points):
            x_coord = point['x']
            y_coord = point['y']

            if 0 <= x_coord <= 1 and 0 <= y_coord <= 1:
                x_model = x_coord * model_width
                y_model = y_coord * model_height
                x = round(x_model * scale)
                y = round(y_model * scale)
            else:
                x = round(x_coord)
                y = round(y_coord)

            if x < 0 or x >= img_width or y < 0 or y >= img_height:
                continue

            point_number = accumulated_count + idx + 1

            draw.ellipse(
                [x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius],
                fill=(255, 0, 0),
                outline=(255, 255, 255),
                width=2
            )

            text_x = x + dot_radius + 5
            text_y = y - (font_size // 2)
            text_bbox = draw.textbbox((text_x, text_y), str(point_number), font=font)
            draw.rectangle(
                [text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2],
                fill=(255, 255, 255)
            )
            draw.text((text_x, text_y), str(point_number), font=font, fill=(0, 0, 0))

        return img


# ==================== ResponseParser ====================
class ResponseParser:
    @staticmethod
    def parse_points(response):
        points = []
        pattern = r'<point>\s*(\{[^}]+\})\s*</point>'
        matches = re.findall(pattern, response)
        for match in matches:
            try:
                point_data = json.loads(match)
                coords = point_data.get('point_2d', [0, 0])
                if isinstance(coords, list) and len(coords) == 2:
                    x = float(coords[0])
                    y = float(coords[1])
                else:
                    continue
                label = point_data.get('label', '')
                count = int(point_data.get('count_number', 0))
                points.append({'x': x, 'y': y, 'label': label, 'count': count})
            except Exception:
                pass
        return points

    @staticmethod
    def has_answer(response):
        return '<answer>' in response.lower() and '</answer>' in response.lower()

    @staticmethod
    def extract_answer(response):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1).strip()
            numbers = re.findall(r'\d+', answer)
            if numbers:
                return numbers[0]
            return answer
        numbers = re.findall(r'\d+', response)
        if numbers:
            return numbers[-1]
        return ""


# ==================== Evaluator ====================
class TrainSetEvaluator:
    def __init__(self, config, device_id=0):
        self.config = config
        self.device_id = device_id
        self.model = ModelInference(config.MODEL_PATH, device_id)
        self.annotator = ImageAnnotator()
        self.parser = ResponseParser()
        # Load prompts once
        self.system_prompt = config.get_system_prompt()
        self.first_turn_template = config.get_first_turn_prompt()
        self.process_template = config.get_process_prompt()

    def evaluate_sample(self, sample):
        """Evaluate a single training sample using interleaved multi-round inference."""
        sample_id = sample['id']
        question = sample['question']
        correct_answer = sample['answer']

        # Lazy load image bytes from parquet
        img_bytes = load_image_bytes(sample['parquet_path'], sample['row_index'])
        if img_bytes is None:
            return {
                'id': sample_id, 'question': question,
                'correct_answer': correct_answer, 'predicted_answer': '',
                'is_correct': False, 'num_rounds': 0, 'error': 'no_image'
            }

        original_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        del img_bytes  # Free immediately
        current_image = original_image.copy()
        accumulated_points_count = 0
        original_image_size = None
        model_image_size = None

        system_msg = {"role": "system", "content": self.system_prompt}
        conversation_history = [system_msg]

        for round_idx in range(1, self.config.MAX_ROUNDS + 1):
            if round_idx == 1:
                prompt_text = self.first_turn_template.replace("{question}", question)
            else:
                prompt_text = self.process_template.replace("{question}", question)

            # HISTORY_MODE=0: only system prompt + current turn
            messages_for_model = [conversation_history[0]]

            current_user_message = {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text}
                ]
            }
            messages_for_model.append(current_user_message)

            text = self.model.processor.apply_chat_template(
                messages_for_model,
                tokenize=False,
                add_generation_prompt=True
            )

            response, original_image_size, model_image_size = self.model.generate_response(
                text, [current_image], self.config.MAX_NEW_TOKENS
            )

            conversation_history.append({"role": "user", "content": prompt_text})
            conversation_history.append({"role": "model", "content": response})

            has_final_answer = self.parser.has_answer(response)
            new_points = self.parser.parse_points(response)

            if new_points:
                current_image = self.annotator.annotate_image_obj(
                    current_image, new_points,
                    accumulated_count=accumulated_points_count,
                    original_size=original_image_size,
                    model_size=model_image_size
                )
                accumulated_points_count += len(new_points)

                if has_final_answer:
                    break
                if round_idx >= self.config.MAX_ROUNDS:
                    break
            else:
                if has_final_answer:
                    break
                break  # No points, no answer -> stop

        final_response = conversation_history[-1]['content'] if conversation_history[-1].get('role') == 'model' else ""
        predicted_answer = self.parser.extract_answer(final_response)
        if not predicted_answer:
            predicted_answer = str(accumulated_points_count)

        is_correct = str(predicted_answer).strip() == str(correct_answer).strip()
        num_rounds = len([m for m in conversation_history if m.get('role') == 'model'])

        del original_image, current_image

        return {
            'id': sample_id,
            'question': question,
            'correct_answer': correct_answer,
            'predicted_answer': predicted_answer,
            'is_correct': is_correct,
            'num_rounds': num_rounds,
            'parquet_file': os.path.basename(sample.get('parquet_path', '')),
            'row_index': sample.get('row_index', -1),
        }

    def evaluate_all(self, samples, temp_json_path):
        """Evaluate all samples assigned to this GPU."""
        results = []
        correct_count = 0

        for i, sample in enumerate(tqdm(samples, desc=f"GPU-{self.device_id}", position=self.device_id)):
            try:
                result = self.evaluate_sample(sample)
                results.append(result)
                if result['is_correct']:
                    correct_count += 1

                if (i + 1) % 50 == 0:
                    acc = correct_count / (i + 1) * 100
                    print(f"[GPU-{self.device_id}] Progress: {i+1}/{len(samples)}, Running accuracy: {acc:.1f}%")

                if (i + 1) % 100 == 0:
                    with open(temp_json_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"[GPU-{self.device_id}] Error on sample {sample.get('id', '?')}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'id': sample.get('id', '?'),
                    'question': sample.get('question', ''),
                    'correct_answer': sample.get('answer', ''),
                    'predicted_answer': '',
                    'is_correct': False,
                    'num_rounds': 0,
                    'error': str(e),
                })

        with open(temp_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"[GPU-{self.device_id}] Done. {correct_count}/{len(samples)} correct ({correct_count/max(len(samples),1)*100:.1f}%)")


# ==================== Worker ====================
def worker(device_id, config, samples_chunk):
    """Per-GPU worker process."""
    temp_json_path = config.OUTPUT_JSON_PATH.replace('.json', f'_part_{device_id}.json')
    print(f"Starting worker on GPU:{device_id}, {len(samples_chunk)} samples...")

    try:
        evaluator = TrainSetEvaluator(config, device_id)
        evaluator.evaluate_all(samples_chunk, temp_json_path)
    except Exception as e:
        print(f"[GPU-{device_id}] Fatal error: {e}")
        import traceback
        traceback.print_exc()


# ==================== Main ====================
def main():
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    if not torch.cuda.is_available():
        print("No CUDA available!")
        return
    num_gpus = min(torch.cuda.device_count(), config.NUM_GPUS)
    print(f"Using {num_gpus} GPUs")

    # Load only metadata (no image bytes) - lightweight
    print("Loading training data metadata...")
    all_samples = load_sample_metadata(config.TRAIN_DATA_DIR)

    # Check for existing progress (resume support)
    evaluated_ids = set()
    if os.path.exists(config.OUTPUT_JSON_PATH):
        try:
            with open(config.OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            evaluated_ids = {r['id'] for r in existing_results}
            print(f"Found {len(evaluated_ids)} already evaluated samples, resuming...")
        except (json.JSONDecodeError, TypeError):
            pass

    remaining = [s for s in all_samples if s['id'] not in evaluated_ids]

    if not remaining:
        print("All samples already evaluated!")
    else:
        print(f"Evaluating {len(remaining)} remaining samples...")

        # Distribute samples across GPUs (metadata only, ~few KB per sample)
        samples_per_gpu = [[] for _ in range(num_gpus)]
        for i, sample in enumerate(remaining):
            samples_per_gpu[i % num_gpus].append(sample)

        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        processes = []
        for i in range(num_gpus):
            if not samples_per_gpu[i]:
                continue
            p = ctx.Process(target=worker, args=(i, config, samples_per_gpu[i]))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

    # Merge results
    print("\nMerging results...")
    final_results_map = {}

    if os.path.exists(config.OUTPUT_JSON_PATH):
        try:
            with open(config.OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
                for r in json.load(f):
                    final_results_map[r['id']] = r
        except (json.JSONDecodeError, TypeError):
            pass

    for i in range(num_gpus):
        temp_path = config.OUTPUT_JSON_PATH.replace('.json', f'_part_{i}.json')
        if os.path.exists(temp_path):
            try:
                with open(temp_path, 'r', encoding='utf-8') as f:
                    for r in json.load(f):
                        final_results_map[r['id']] = r
                os.remove(temp_path)
                print(f"Merged and removed: {temp_path}")
            except (json.JSONDecodeError, TypeError):
                print(f"Warning: could not parse {temp_path}")

    final_results = list(final_results_map.values())

    with open(config.OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    # Statistics
    total = len(final_results)
    correct = sum(1 for r in final_results if r.get('is_correct', False))
    accuracy = correct / total if total > 0 else 0

    easy_samples = [
        {
            'id': r['id'],
            'question': r['question'],
            'correct_answer': r['correct_answer'],
            'parquet_file': r.get('parquet_file', ''),
            'row_index': r.get('row_index', -1),
        }
        for r in final_results if r.get('is_correct', False)
    ]

    with open(config.OUTPUT_EASY_IDS_PATH, 'w', encoding='utf-8') as f:
        json.dump(easy_samples, f, ensure_ascii=False, indent=2)

    # Range-based statistics
    range_0_5 = []
    range_5_plus = []
    for r in final_results:
        try:
            gt = int(r.get('correct_answer', 0))
            if gt <= 5:
                range_0_5.append(r)
            else:
                range_5_plus.append(r)
        except (ValueError, TypeError):
            range_0_5.append(r)

    correct_0_5 = sum(1 for r in range_0_5 if r.get('is_correct', False))
    correct_5_plus = sum(1 for r in range_5_plus if r.get('is_correct', False))

    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    print(f"Correct (easy): {correct}")
    print(f"Wrong (hard): {total - correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"\nRange 0-5:  {len(range_0_5)} samples, {correct_0_5} correct ({correct_0_5/max(len(range_0_5),1)*100:.1f}%)")
    print(f"Range 5+:   {len(range_5_plus)} samples, {correct_5_plus} correct ({correct_5_plus/max(len(range_5_plus),1)*100:.1f}%)")
    print(f"\nEasy sample IDs saved to: {config.OUTPUT_EASY_IDS_PATH}")
    print(f"Full results saved to: {config.OUTPUT_JSON_PATH}")
    print(f"{'='*60}")

    # Save report
    report_path = config.OUTPUT_JSON_PATH.replace('.json', '_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"SFT Base Model Evaluation on Training Set\n")
        f.write(f"{'='*60}\n")
        f.write(f"Model: {config.MODEL_PATH}\n")
        f.write(f"Dataset: {config.TRAIN_DATA_DIR}\n")
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total: {total}\n")
        f.write(f"Correct (easy): {correct} ({accuracy:.2%})\n")
        f.write(f"Wrong (hard): {total - correct}\n\n")
        f.write(f"Range 0-5: {len(range_0_5)} samples, {correct_0_5} correct ({correct_0_5/max(len(range_0_5),1)*100:.1f}%)\n")
        f.write(f"Range 5+:  {len(range_5_plus)} samples, {correct_5_plus} correct ({correct_5_plus/max(len(range_5_plus),1)*100:.1f}%)\n\n")

        from collections import Counter
        easy_gt_dist = Counter(int(s['correct_answer']) for s in easy_samples)
        f.write(f"Easy sample GT distribution:\n")
        for gt in sorted(easy_gt_dist.keys()):
            f.write(f"  GT={gt}: {easy_gt_dist[gt]}\n")

    print(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
