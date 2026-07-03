"""
Pass@K evaluation on hard training data for rejection sampling.
- For each sample, runs K independent rollouts (with temperature sampling)
- Computes pass@1, pass@8, pass@16, pass@32 statistics
- Identifies samples with pass@K > 0 (learnable by RL) vs pass@K = 0 (unlearnable)
- Outputs filtered dataset for RL training data construction
- Supports multi-GPU parallel evaluation with resume capability

Based on eval_sft_base_on_trainset.py architecture.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_pass_at_k_hard_data.py \
        --model_path /path/to/model \
        --data_dir /path/to/hard_data/data \
        --output_dir /path/to/output \
        --num_rollouts 32 \
        --temperature 1.0 \
        --early_stop_threshold 1
"""

import os
import io
import json
import re
import time
import glob
import math
import argparse
import torch
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoConfig, PretrainedConfig
from typing import List, Dict, Optional
from collections import Counter


# ==================== Defaults ====================
DEFAULT_MODEL_PATH = '/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/models/StepCount-7B-SFT-30k-high/checkpoint-3537'
DEFAULT_DATA_DIR = '/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/datasets/StepCountQA-RL-Traj_0_10_hard_only/data'
DEFAULT_OUTPUT_DIR = '/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/datasets/eval_pass_at_k_hard_sft_base'

PROMPT_DIR = '/data/workspace/hyleochang/EasyR1-latest/examples/format_prompt'

# Eval parameters (match training config)
HISTORY_MODE = 0
MAX_NEW_TOKENS = 8192
MAX_ROUNDS = 11
MAX_PIXELS = 12845056


# ==================== Argument Parser ====================
def parse_args():
    parser = argparse.ArgumentParser(description='Pass@K evaluation for hard data rejection sampling')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to the model checkpoint')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                        help='Directory containing parquet training data')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory for results')
    parser.add_argument('--num_rollouts', type=int, default=16,
                        help='Number of rollouts per sample (K)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature for diverse rollouts')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p (nucleus) sampling parameter')
    parser.add_argument('--num_gpus', type=int, default=2,
                        help='Number of GPUs to use')
    parser.add_argument('--early_stop_threshold', type=int, default=0,
                        help='Stop early after this many correct rollouts (0=no early stop)')
    parser.add_argument('--min_pass_rate', type=float, default=0.0,
                        help='Minimum pass@K rate to be considered learnable (for filtering)')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume from previous checkpoint')
    return parser.parse_args()


# ==================== Prompts ====================
def load_prompt(filename):
    path = os.path.join(PROMPT_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()


# ==================== Data Loading ====================
def load_sample_metadata(data_dir):
    """Load metadata only (no images) for memory efficiency."""
    parquet_files = sorted(glob.glob(os.path.join(data_dir, '*.parquet')))
    print(f"Found {len(parquet_files)} parquet files in {data_dir}")

    samples = []
    global_idx = 0
    for pf in parquet_files:
        print(f"  Scanning {os.path.basename(pf)}...")
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

    print(f"Total samples: {len(samples)}")
    return samples


# Per-process parquet cache
_parquet_cache = {}

def load_image_bytes(parquet_path, row_index):
    """Lazy-load image bytes with caching."""
    global _parquet_cache
    if parquet_path not in _parquet_cache:
        print(f"  [Cache] Loading images from {os.path.basename(parquet_path)}...")
        _parquet_cache[parquet_path] = pq.read_table(parquet_path, columns=['images'])
    table = _parquet_cache[parquet_path]
    images_col = table['images'][row_index].as_py()
    if images_col and len(images_col) > 0:
        return images_col[0].get('bytes', None)
    return None


# ==================== Model Inference ====================
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

    def generate_response(self, text, images, max_new_tokens=2048,
                          do_sample=True, temperature=1.0, top_p=0.95):
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

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs['do_sample'] = True
            gen_kwargs['temperature'] = temperature
            gen_kwargs['top_p'] = top_p
        else:
            gen_kwargs['do_sample'] = False

        with torch.no_grad():
            generated_ids = self.model.generate(**gen_kwargs)

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


# ==================== Image / Parser ====================
class ImageAnnotator:
    @staticmethod
    def annotate_image_obj(image, points, accumulated_count=0, original_size=None, model_size=None):
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
            x_coord, y_coord = point['x'], point['y']
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
                fill=(255, 0, 0), outline=(255, 255, 255), width=2
            )
            text_x = x + dot_radius + 5
            text_y = y - (font_size // 2)
            text_bbox = draw.textbbox((text_x, text_y), str(point_number), font=font)
            draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=(255, 255, 255))
            draw.text((text_x, text_y), str(point_number), font=font, fill=(0, 0, 0))
        return img


class ResponseParser:
    @staticmethod
    def parse_points(response):
        points = []
        pattern = r'<point>\s*(\{[^}]+\})\s*</point>'
        for match in re.findall(pattern, response):
            try:
                point_data = json.loads(match)
                coords = point_data.get('point_2d', [0, 0])
                if isinstance(coords, list) and len(coords) == 2:
                    x, y = float(coords[0]), float(coords[1])
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


# ==================== Pass@K Evaluator ====================
class PassAtKEvaluator:
    def __init__(self, model_path, device_id, temperature=1.0, top_p=0.95):
        self.device_id = device_id
        self.temperature = temperature
        self.top_p = top_p
        self.model = ModelInference(model_path, device_id)
        self.annotator = ImageAnnotator()
        self.parser = ResponseParser()
        self.system_prompt = load_prompt('StepCount_interleaved_system_prompt.txt')
        self.first_turn_template = load_prompt('StepCount_interleaved_first_turn_prompt.txt')
        self.process_template = load_prompt('StepCount_interleaved_process_prompt.txt')

    def run_single_rollout(self, question, original_image, do_sample=True):
        """Execute one complete multi-round trajectory rollout.
        Returns (predicted_answer, num_rounds)."""
        current_image = original_image.copy()
        accumulated_points_count = 0
        original_image_size = None
        model_image_size = None

        system_msg = {"role": "system", "content": self.system_prompt}
        conversation_history = [system_msg]

        for round_idx in range(1, MAX_ROUNDS + 1):
            if round_idx == 1:
                prompt_text = self.first_turn_template.replace("{question}", question)
            else:
                prompt_text = self.process_template.replace("{question}", question)

            # HISTORY_MODE=0: system + current turn only
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
                messages_for_model, tokenize=False, add_generation_prompt=True
            )

            response, original_image_size, model_image_size = self.model.generate_response(
                text, [current_image], MAX_NEW_TOKENS,
                do_sample=do_sample, temperature=self.temperature, top_p=self.top_p
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
                if has_final_answer or round_idx >= MAX_ROUNDS:
                    break
            else:
                break  # No points, no answer -> stop

        final_response = conversation_history[-1]['content'] if conversation_history[-1].get('role') == 'model' else ""
        predicted_answer = self.parser.extract_answer(final_response)
        if not predicted_answer:
            predicted_answer = str(accumulated_points_count)

        num_rounds = len([m for m in conversation_history if m.get('role') == 'model'])
        del current_image
        return predicted_answer, num_rounds

    def evaluate_sample_pass_k(self, sample, num_rollouts=32, early_stop_threshold=0):
        """Evaluate a single sample K times. Returns pass@K result dict."""
        sample_id = sample['id']
        question = sample['question']
        correct_answer = sample['answer']

        # Load image once, reuse for all rollouts
        img_bytes = load_image_bytes(sample['parquet_path'], sample['row_index'])
        if img_bytes is None:
            return {
                'id': sample_id, 'question': question,
                'correct_answer': correct_answer,
                'num_rollouts': 0, 'num_correct': 0,
                'pass_at_k': 0.0, 'predictions': [],
                'error': 'no_image'
            }

        original_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        del img_bytes

        predictions = []
        num_correct = 0

        for k in range(num_rollouts):
            try:
                pred_answer, num_rounds = self.run_single_rollout(
                    question, original_image, do_sample=True
                )
                is_correct = str(pred_answer).strip() == str(correct_answer).strip()
                if is_correct:
                    num_correct += 1

                predictions.append({
                    'rollout_idx': k,
                    'predicted_answer': pred_answer,
                    'is_correct': is_correct,
                    'num_rounds': num_rounds,
                })

                # Early stop if we have found enough correct rollouts
                if early_stop_threshold > 0 and num_correct >= early_stop_threshold:
                    break

            except Exception as e:
                predictions.append({
                    'rollout_idx': k,
                    'predicted_answer': '',
                    'is_correct': False,
                    'num_rounds': 0,
                    'error': str(e),
                })

        del original_image

        total_trials = len(predictions)
        pass_rate = num_correct / total_trials if total_trials > 0 else 0.0

        return {
            'id': sample_id,
            'question': question,
            'correct_answer': correct_answer,
            'num_rollouts': total_trials,
            'num_correct': num_correct,
            'pass_at_k': pass_rate,
            'parquet_file': os.path.basename(sample.get('parquet_path', '')),
            'row_index': sample.get('row_index', -1),
            'predictions': predictions,
        }


# ==================== Pass@K Metrics ====================
def compute_pass_at_k_unbiased(n, c, k):
    """Unbiased estimator for pass@k.
    n: total rollouts, c: correct rollouts, k: target k.
    From Chen et al. 'Evaluating Large Language Models Trained on Code' (2021).
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def generate_report(results, output_dir, num_rollouts):
    """Generate comprehensive pass@K statistics report."""
    total = len(results)
    if total == 0:
        return [], []

    valid = [r for r in results if r.get('num_rollouts', 0) > 0]

    k_values = [1, 4, 8, 16, 32]
    k_values = [k for k in k_values if k <= num_rollouts]

    print(f"\n{'='*70}")
    print(f"Pass@K Evaluation Report")
    print(f"{'='*70}")
    print(f"Total samples: {total}, Valid: {len(valid)}")

    for k in k_values:
        pass_k_scores = []
        for r in valid:
            n = r['num_rollouts']
            c = r['num_correct']
            if n >= k:
                pass_k_scores.append(compute_pass_at_k_unbiased(n, c, k))
            else:
                pass_k_scores.append(1.0 if c > 0 else 0.0)

        avg_pass_k = sum(pass_k_scores) / len(pass_k_scores)
        nonzero = sum(1 for s in pass_k_scores if s > 0)
        print(f"\n  pass@{k}: {avg_pass_k:.4f} ({avg_pass_k*100:.2f}%)")
        print(f"    Samples with pass@{k} > 0: {nonzero}/{len(valid)} ({nonzero/len(valid)*100:.1f}%)")

    correct_dist = Counter(r['num_correct'] for r in valid)
    print(f"\n  Correct count distribution (out of {num_rollouts} rollouts):")
    for c in sorted(correct_dist.keys()):
        print(f"    {c} correct: {correct_dist[c]} samples ({correct_dist[c]/len(valid)*100:.1f}%)")

    print(f"\n  Pass@{num_rollouts} by GT:")
    gt_groups = {}
    for r in valid:
        try:
            gt = int(r['correct_answer'])
        except (ValueError, TypeError):
            gt = -1
        if gt not in gt_groups:
            gt_groups[gt] = []
        gt_groups[gt].append(r)

    for gt in sorted(gt_groups.keys()):
        group = gt_groups[gt]
        pass_any = sum(1 for r in group if r['num_correct'] > 0)
        avg_rate = sum(r['pass_at_k'] for r in group) / len(group)
        print(f"    GT={gt}: {len(group)} samples, {pass_any} with pass>0 ({pass_any/len(group)*100:.1f}%), avg_rate={avg_rate:.3f}")

    learnable = [r for r in valid if r['num_correct'] > 0]
    unlearnable = [r for r in valid if r['num_correct'] == 0]
    print(f"\n  Learnable (pass@{num_rollouts} > 0): {len(learnable)} ({len(learnable)/len(valid)*100:.1f}%)")
    print(f"  Unlearnable (pass@{num_rollouts} = 0): {len(unlearnable)} ({len(unlearnable)/len(valid)*100:.1f}%)")

    report_path = os.path.join(output_dir, 'pass_at_k_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Pass@K Evaluation Report\n{'='*70}\n")
        f.write(f"Total: {total}, Valid: {len(valid)}\n")
        f.write(f"Rollouts per sample: {num_rollouts}\n\n")

        for k in k_values:
            scores = []
            for r in valid:
                n, c = r['num_rollouts'], r['num_correct']
                scores.append(compute_pass_at_k_unbiased(n, c, k) if n >= k else (1.0 if c > 0 else 0.0))
            avg = sum(scores) / len(scores)
            nz = sum(1 for s in scores if s > 0)
            f.write(f"pass@{k}: {avg:.4f} ({avg*100:.2f}%), nonzero={nz}/{len(valid)}\n")

        f.write(f"\nLearnable: {len(learnable)} ({len(learnable)/len(valid)*100:.1f}%)\n")
        f.write(f"Unlearnable: {len(unlearnable)} ({len(unlearnable)/len(valid)*100:.1f}%)\n")

        f.write(f"\nCorrect count distribution:\n")
        for c in sorted(correct_dist.keys()):
            f.write(f"  {c}/{num_rollouts}: {correct_dist[c]} samples\n")

        f.write(f"\nPer-GT breakdown:\n")
        for gt in sorted(gt_groups.keys()):
            g = gt_groups[gt]
            pa = sum(1 for r in g if r['num_correct'] > 0)
            ar = sum(r['pass_at_k'] for r in g) / len(g)
            f.write(f"  GT={gt}: {len(g)} samples, pass>0={pa} ({pa/len(g)*100:.1f}%), avg_rate={ar:.3f}\n")

    print(f"\n  Report saved to: {report_path}")
    return learnable, unlearnable


# ==================== Worker ====================
def worker(device_id, args, samples_chunk):
    """Per-GPU worker: evaluate assigned samples with K rollouts each."""
    temp_path = os.path.join(args.output_dir, f'pass_at_k_results_part_{device_id}.json')
    print(f"[GPU-{device_id}] Starting: {len(samples_chunk)} samples, K={args.num_rollouts}")

    try:
        evaluator = PassAtKEvaluator(
            args.model_path, device_id,
            temperature=args.temperature, top_p=args.top_p
        )

        results = []
        for i, sample in enumerate(tqdm(samples_chunk, desc=f"GPU-{device_id}", position=device_id)):
            try:
                result = evaluator.evaluate_sample_pass_k(
                    sample,
                    num_rollouts=args.num_rollouts,
                    early_stop_threshold=args.early_stop_threshold,
                )
                results.append(result)

                if (i + 1) % 10 == 0:
                    pass_any = sum(1 for r in results if r['num_correct'] > 0)
                    print(f"[GPU-{device_id}] {i+1}/{len(samples_chunk)}: "
                          f"{pass_any}/{len(results)} learnable ({pass_any/len(results)*100:.1f}%)")

                # Save checkpoint every 20 samples
                if (i + 1) % 20 == 0:
                    with open(temp_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"[GPU-{device_id}] Error on {sample['id']}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'id': sample['id'], 'question': sample.get('question', ''),
                    'correct_answer': sample.get('answer', ''),
                    'num_rollouts': 0, 'num_correct': 0,
                    'pass_at_k': 0.0, 'predictions': [],
                    'error': str(e),
                })

        # Final save
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        pass_any = sum(1 for r in results if r['num_correct'] > 0)
        print(f"[GPU-{device_id}] Done: {pass_any}/{len(results)} learnable ({pass_any/len(results)*100:.1f}%)")

    except Exception as e:
        print(f"[GPU-{device_id}] Fatal: {e}")
        import traceback
        traceback.print_exc()


# ==================== Main ====================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not torch.cuda.is_available():
        print("No CUDA available!")
        return

    num_gpus = min(torch.cuda.device_count(), args.num_gpus)
    print(f"Using {num_gpus} GPUs, K={args.num_rollouts}, T={args.temperature}")

    print("Loading training data metadata...")
    all_samples = load_sample_metadata(args.data_dir)

    # Resume support
    final_path = os.path.join(args.output_dir, 'pass_at_k_results.json')
    evaluated_ids = set()
    if args.resume and os.path.exists(final_path):
        try:
            with open(final_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            evaluated_ids = {r['id'] for r in existing}
            print(f"Resuming: {len(evaluated_ids)} already evaluated")
        except (json.JSONDecodeError, TypeError):
            pass

    # Also check partial files
    for i in range(num_gpus):
        part_path = os.path.join(args.output_dir, f'pass_at_k_results_part_{i}.json')
        if os.path.exists(part_path):
            try:
                with open(part_path, 'r', encoding='utf-8') as f:
                    for r in json.load(f):
                        evaluated_ids.add(r['id'])
            except (json.JSONDecodeError, TypeError):
                pass

    remaining = [s for s in all_samples if s['id'] not in evaluated_ids]
    print(f"Remaining: {len(remaining)} samples to evaluate")

    if remaining:
        samples_per_gpu = [[] for _ in range(num_gpus)]
        for i, sample in enumerate(remaining):
            samples_per_gpu[i % num_gpus].append(sample)

        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        processes = []
        for i in range(num_gpus):
            if not samples_per_gpu[i]:
                continue
            p = ctx.Process(target=worker, args=(i, args, samples_per_gpu[i]))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

    # Merge all results
    print("\nMerging results...")
    results_map = {}
    if os.path.exists(final_path):
        try:
            with open(final_path, 'r', encoding='utf-8') as f:
                for r in json.load(f):
                    results_map[r['id']] = r
        except (json.JSONDecodeError, TypeError):
            pass

    for i in range(num_gpus):
        part_path = os.path.join(args.output_dir, f'pass_at_k_results_part_{i}.json')
        if os.path.exists(part_path):
            try:
                with open(part_path, 'r', encoding='utf-8') as f:
                    for r in json.load(f):
                        results_map[r['id']] = r
                os.remove(part_path)
                print(f"  Merged: {part_path}")
            except (json.JSONDecodeError, TypeError):
                print(f"  Warning: could not parse {part_path}")

    final_results = list(results_map.values())

    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    print(f"Results saved: {final_path} ({len(final_results)} samples)")

    learnable, unlearnable = generate_report(final_results, args.output_dir, args.num_rollouts)

    if learnable:
        learnable_path = os.path.join(args.output_dir, 'learnable_samples.json')
        learnable_ids = [{
            'id': r['id'], 'question': r['question'],
            'correct_answer': r['correct_answer'],
            'pass_rate': r['pass_at_k'],
            'num_correct': r['num_correct'],
            'num_rollouts': r['num_rollouts'],
            'parquet_file': r.get('parquet_file', ''),
            'row_index': r.get('row_index', -1),
        } for r in learnable]
        with open(learnable_path, 'w', encoding='utf-8') as f:
            json.dump(learnable_ids, f, ensure_ascii=False, indent=2)
        print(f"Learnable samples saved: {learnable_path} ({len(learnable_ids)} samples)")

    if unlearnable:
        unlearnable_path = os.path.join(args.output_dir, 'unlearnable_samples.json')
        unlearnable_ids = [{
            'id': r['id'], 'question': r['question'],
            'correct_answer': r['correct_answer'],
            'parquet_file': r.get('parquet_file', ''),
            'row_index': r.get('row_index', -1),
        } for r in unlearnable]
        with open(unlearnable_path, 'w', encoding='utf-8') as f:
            json.dump(unlearnable_ids, f, ensure_ascii=False, indent=2)
        print(f"Unlearnable samples saved: {unlearnable_path} ({len(unlearnable_ids)} samples)")

    print(f"\n{'='*70}")
    print(f"Pass@K evaluation complete!")
    print(f"  Learnable (pass@{args.num_rollouts} > 0): {len(learnable)}")
    print(f"  Unlearnable (pass@{args.num_rollouts} = 0): {len(unlearnable)}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
