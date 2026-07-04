import os
import json
from PIL import Image
import torch
from io import BytesIO

os.environ["UNSLOTH_ENABLE_FLEX_ATTENTION"] = "0"   # 1. 关掉 flex-attention 补丁
os.environ["UNSLOTH_DISABLE_CAUSAL_MASK_PATCH"] = "1"  # 2. 新版 Unsloth 支持

import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen
from transformers.masking_utils import create_causal_mask as orig
qwen.create_causal_mask = orig

from unsloth import FastVisionModel # FastLanguageModel for LLMs
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset, Dataset
import requests
from datasets import concatenate_datasets
max_seq_length = 16384 # Must be this long for VLMs
lora_rank = 16 # Larger rank = smarter, but slower


with open("output_action.json", "r", encoding="utf-8") as f:
    list_task = json.load(f)
print("Loaded json")



def get_questions1():
    for answer in list_task:
        image_path = answer["images"][0]
        instruction = answer["instruction"]
        yield{
            "prompt":[{ "role": "user",
            "content" : [
                {"type" : "text",  "text": instruction},
                {"type" : "image"} ]
            }
            ],
            # 仅保存路径，避免在 Dataset 中持有 PIL.Image 对象
            "image" : image_path,
            "answer": answer["output"],
            "image_path": image_path,  # 添加图片路径用于查找预生成的caption
        }

dataset1 = Dataset.from_generator(get_questions1)
# print(dataset1[0])
    

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "qwen2_5vl",
    load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
    max_seq_length = max_seq_length,
    fast_inference = False, # 训练阶段禁用 vLLM 快速推理，避免持久缓存导致显存增长
    gpu_memory_utilization = 0.8,

)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # False if not finetuning vision layers
    finetune_language_layers   = True,  # False if not finetuning language layers
    finetune_attention_modules = True,  # False if not finetuning attention layers
    finetune_mlp_modules       = True,  # False if not finetuning MLP layers

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)


train_dataset = dataset1.map(
    lambda example: {
        "prompt": tokenizer.apply_chat_template(
            example["prompt"],
            tokenize = False,
            add_generation_prompt = True, # Must add assistant
        )
    }
)

FastVisionModel.for_training(model) # Enable for training!

# ------------------------------------Reward--------------------------------------------------

import re
import os
from datetime import datetime
from openai import OpenAI
import openai
client = OpenAI(
    # If environment variables are not configured, replace the following line with: api_key="sk-xxx",
    api_key="",
    base_url="",
)
prompt = """
You are an atomic reasoning extraction system.
Your task is to extract Scene Atomic Actions and ToM Atomic Actions from text.
Your output must follow exact symbolic grammar, allowing perfect token-level matching.
Do not use synonyms, extra words, or alternative phrasing.

=== 1. Atomic Schema (Fixed Vocabulary) ===

(A) Scene Atomic Actions
Allowed verbs (physical actions):
{ walk, run, turn, sit, standup, open, close, pick, place, putin, putback, hold,
  puton, switchon, switchoff, lookat, grab, stand, move, sleep, read, write, watch,
  listen, cut, cook }

Format:
action(character, object)
action(character, object, from=location)
action(character, object, to=location)
action(character, object, from=location, to=location)
action(character, location)

This part can only be extracted from <Description> and this part only contains human action, not robot.

(B) ToM Atomic Predicates
Allowed predicates:
{ attribute_belief, hold_true_belief, lack_belief,
  attribute_desire, attribute_intention, form_intention,
  know, unknown, want, plan_action, tell, resolve_misbelief }

Format:
predicate(agent, content)
predicate(agent, content(subject, object, location))

This part can only be extracted from ToM Reasoning.

=== 2. Normalization Rules (Exact Match Policy) ===
- Character IDs: lowercase char0, char1, char2
- Robot Agent: always use robot
- Objects: lowercase, replace spaces with _, remove articles
- Locations: lowercase, replace spaces with _, remove articles
- No articles or pronouns inside actions or content
- Canonical Verb Mapping: grab -> pick, put down -> place, walk into -> walk, turn off -> switchoff
- Action Separation: one verb per line
- Syntax: always use parentheses for parameters, commas to separate, no spaces before/after parentheses
- Parameter Style: always use from= and to= if known, omit if unknown

=== 3. ToM Content Schema (Strict) ===

Belief / Knowledge Predicates:
- attribute_belief(agent, content): searching(object); human_believes(object_on(location)); object_on(location)
- hold_true_belief(agent, content): object_on(location)
- lack_belief(agent, content): object_on(location); goal_of(human, action)
- know(agent, content): object_on(location); goal_of(human, action)
- unknown(agent, content): object_on(location)

Desire / Intention Predicates:
- attribute_desire(agent, content): assist(human, find(object)); resolve_misbelief(human, belief_type); achieve(goal_name)
- form_intention(agent, content): fetch(object, from=location1, to=location2)
- attribute_intention(agent, content): same as form_intention

Action / Plan Predicates:
- plan_action(agent, subaction1, subaction2, ...): each subaction must come from Scene Atomic Action Set, without character
- tell(agent, content)
- resolve_misbelief(agent, content): belief_conflict(human, object_location)

=== 4. Example (Strict Canonical) ===
Input:
<Description>
char0 walked into the kitchen and picked up the mug from the sink. She placed the mug on the kitchen counter.
Later, char1 walked into the kitchen, picked up the mug from the kitchen counter, and walked over to the stove. He placed the mug on the sofa.
Then, char2 walked into the kitchen, picked up the mug from the stove, and walked back to the kitchen counter. She placed the mug on the kitchen counter.
Finally, char0 walked back to the sink, opened the cabinet under the sink, closed it, and walked around the kitchen before returning to the sink.

<Robot Belief> I believe char0 might currently be looking for the mug or intending to use it. I believe char0 thinks the mug is still at the sink. I believe the mug is actually on the kitchen counter.
<Robot Desire> I want to help char0 achieve her goal of locating or using the mug, and I want to resolve the conflict between char0's belief and the real-world state.
<Robot Intention> Bring the mug from the kitchen counter and place it near char0 at the sink.
<Decision> Pick up the mug from the kitchen counter and bring it to char0 at the sink.

Output:
<Scene Atomic Actions>
1. walk(char0, kitchen)
2. pick(char0, mug, from=sink)
3. place(char0, mug, to=kitchen_counter)
4. walk(char1, kitchen)
5. pick(char1, mug, from=kitchen_counter)
6. walk(char1, stove)
7. place(char1, mug, to=sofa)
8. walk(char2, kitchen)
9. pick(char2, mug, from=stove)
10. walk(char2, kitchen_counter)
11. place(char2, mug, to=kitchen_counter)
12. walk(char0, sink)
13. open(char0, cabinet_under_sink)
14. close(char0, cabinet_under_sink)
15. walk(char0, kitchen)
16. walk(char0, sink)

<ToM Atomic Actions>
1. attribute_belief(robot, searching(mug))
2. attribute_belief(robot, human_believes(object_on(sink)))
3. hold_true_belief(robot, object_on(kitchen_counter))
4. attribute_desire(robot, assist(human, find(mug)))
5. attribute_desire(robot, resolve_misbelief(human, belief_type=object_location))
6. form_intention(robot, fetch(mug, from=kitchen_counter, to=sink))
7. plan_action(robot, walk(kitchen_counter), pick(mug), walk(sink), place(mug, sink))
   """


import re
from datetime import datetime
import os

def structure_reward(prompts, completions, **kwargs):
    """
    Reward function for strict binary scoring:
    - 1 if the output has all six tags in correct order with content
    - 0 otherwise
    """

    pattern = re.compile(
        r"<Description>\s*(.+?)\s*"
        r"<Robot Belief>\s*(.+?)\s*"
        r"<Robot Desire>\s*(.+?)\s*"
        r"<Robot Intention>\s*(.+?)\s*"
        r"<Decision>\s*(.+?)\s*"
        r"<Action>\s*(.+)",
        re.DOTALL
    )

    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S")

    for completion in completions:
        text = completion

        match = pattern.fullmatch(text.strip())
        score = 1.0 if match else 0.0
        rewards.append(score)

        # 调试日志
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "structure_reward_log.txt")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"==== {current_time} Structure reward: {score} ====\n{text}\n\n")

    return rewards


# =========================
# 1️⃣ 动作匹配函数
# =========================
# def match_action(a, b):
#     """带参数匹配的动作相似度 [0,1]"""
#     act_a = a.split("(")[0]; act_b = b.split("(")[0]
#     if act_a != act_b:
#         return 0.0
#     args_a = a[a.find("(")+1:a.find(")")].split(",")
#     args_b = b[b.find("(")+1:b.find(")")].split(",")
#     same = sum(1 for x, y in zip(args_a, args_b) if x.strip() == y.strip())
#     return 0.2 + 0.8 * (same / max(len(args_a), len(args_b), 1))
def match_action(a, b):
    """带参数匹配的动作相似度 [0,1]，支持嵌套括号"""
    # 提取动作名称
    act_a = a.split("(")[0].strip()
    act_b = b.split("(")[0].strip()
    if act_a != act_b:
        return 0.0

    # --- 提取最外层括号内的参数字符串 ---
    def get_outer_args(s):
        start = s.find("(")
        if start == -1:
            return []
        depth = 0
        end = None
        for i, c in enumerate(s[start:], start=start):
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is None:
            return []
        return s[start+1:end]

    args_str_a = get_outer_args(a)
    args_str_b = get_outer_args(b)

    # --- 拆分顶层逗号参数，同时保留内部括号 ---
    def split_top_level_args(s):
        args = []
        current = ""
        depth = 0
        for c in s:
            if c == "(":
                depth += 1
                current += c
            elif c == ")":
                depth -= 1
                current += c
            elif c == "," and depth == 0:
                args.append(current.strip())
                current = ""
            else:
                current += c
        if current:
            args.append(current.strip())
        return args

    args_a = split_top_level_args(args_str_a)
    args_b = split_top_level_args(args_str_b)
    # print(args_a, args_b)

    # --- 计算相似度 ---
    same = sum(1 for x, y in zip(args_a, args_b) if x == y)
    return 0.2 + 0.8 * (same / max(len(args_a), len(args_b), 1))
# =========================
# 2️⃣ ROUGE-N (n=1,2)
# =========================
def rouge_n(ref_actions, pred_actions, n=1):
    """
    加权版ROUGE-N (适用于动作序列)
    修正版：避免重复匹配导致F1 > 1
    """
    def ngrams(seq, n):
        return [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]
    
    # === Step 1: 提取 n-grams ===
    ref_ngrams = ngrams(ref_actions, n)
    pred_ngrams = ngrams(pred_actions, n)
    if not ref_ngrams or not pred_ngrams:
        return 0.0
    
    # === Step 2: 匹配每个 ref_ngram 的最佳 pred_ngram ===
    overlap = 0.0
    used = set()  # 记录已匹配的 pred_ngram
    
    for r in ref_ngrams:
        best_score = 0.0
        best_idx = -1
        for j, p in enumerate(pred_ngrams):
            if j in used:  # 避免重复匹配
                continue
            if len(r) == len(p):
                scores = [match_action(a, b) for a, b in zip(r, p)]
                if all(s > 0 for s in scores):  # 动作类型相同才算匹配
                    avg_score = sum(scores) / len(scores)
                    if avg_score > best_score:
                        best_score = avg_score
                        best_idx = j
        if best_idx >= 0:
            used.add(best_idx)
            overlap += best_score

    # === Step 3: 计算 recall / precision / F1 ===
    recall = overlap / len(ref_ngrams)
    precision = overlap / len(pred_ngrams)
    if recall + precision == 0:
        return 0.0
    f1 = 2 * recall * precision / (recall + precision)
    
    # === Step 4: 数值稳定处理 ===
    f1 = max(0.0, min(f1, 1.0))
    
    return f1


# =========================
# 3️⃣ ROUGE-L (顺序敏感)
# =========================
def rouge_l(ref_actions, pred_actions):
    """顺序敏感的ROUGE-L（基于加权LCS）"""
    m, n = len(ref_actions), len(pred_actions)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            match_score = match_action(ref_actions[i], pred_actions[j])
            if match_score > 0.5:  # 动作相似则纳入LCS
                dp[i+1][j+1] = dp[i][j] + match_score
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    lcs_score = dp[m][n]
    recall = lcs_score / max(m, 1)
    precision = lcs_score / max(n, 1)
    f1 = 2 * recall * precision / (recall + precision + 1e-9)
    return f1

# =========================
# 4️⃣ 综合ROUGE计算
# =========================
def compute_structured_rouge(ref_actions, pred_actions, w1=0.2, w2=0.3, wl=0.5):
    """计算ROUGE1,2,L并融合为最终得分"""
    r1 = rouge_n(ref_actions, pred_actions, 1)
    r2 = rouge_n(ref_actions, pred_actions, 2)
    rl = rouge_l(ref_actions, pred_actions)
    final = w1*r1 + w2*r2 + wl*rl
    return {"rouge1": r1, "rouge2": r2, "rougeL": rl, "final": final}

# =========================
# 5️⃣ 从文本提取动作序列
# =========================
def extract_atomic_actions(text):
    """
    提取 <Scene Atomic Actions> 和 <ToM Atomic Actions> 部分
    """
    scene_block = re.search(r"<Scene Atomic Actions>(.*?)<ToM Atomic Actions>", text, re.DOTALL)
    tom_block = re.search(r"<ToM Atomic Actions>(.*)", text, re.DOTALL)

    scene_actions, tom_actions = [], []

    if scene_block:
        scene_lines = re.findall(r'\d+\.\s*(.*)', scene_block.group(1))
        scene_actions = [line.strip() for line in scene_lines if line.strip()]
    if tom_block:
        tom_lines = re.findall(r'\d+\.\s*(.*)', tom_block.group(1))
        tom_actions = [line.strip() for line in tom_lines if line.strip()]

    return scene_actions, tom_actions


import json
from rouge_score import rouge_scorer

def atomic_action_reward_with_weights(prompts, completions, answer, **kwargs):
    """
    Compute weighted ROUGE-based reward for Scene and ToM atomic actions.
    Now extracts plan_action(...) internal sequence for separate ROUGE computation.
    """
    w1=0.2
    w2=0.3
    wl=0.5
    rewards = []
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_path = "reward_log.jsonl"

    for idx, (content, sol) in enumerate(zip(completions, answer)):
        print(content)
        print(sol)
        reward = 0.0
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(content, sol)
        
        r1 = scores['rouge1'].fmeasure
        r2 = scores['rouge2'].fmeasure
        rl = scores['rougeL'].fmeasure

        reward = 0.2 * r1 + 0.3 * r2 + 0.5 * rl

        rewards.append(reward)

        # ✅ 将 reward 追加写入日志文件（JSONL格式）
        # with open(log_path, "a", encoding="utf-8") as f:
        #     json.dump({
        #         "time": current_time,
        #         "idx": idx,
        #         "reward": reward,
        #         "scene_rouge": scene_scores,
        #         "tom_nonplan": tom_nonplan_scores,
        #         "tom_plan": tom_plan_scores
        #     }, f, ensure_ascii=False)
        #     f.write("\n")

    return rewards

#------------------------------------------------------------------------------------------------

training_args = GRPOConfig(
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    log_completions = False,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 2, # Increase to 4 for smoother training
    reward_weights = [0.1, 0.9],
    num_generations = 8, # 降低生成次数，缓解 KV cache 占用
    max_prompt_length = 32000,
    max_completion_length = 512,
    num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 3000,
    save_steps = 200,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs_7b",

    
    importance_sampling_level = "sequence",
    mask_truncated_completions = True,
    loss_type = "dr_grpo",
)

trainer = GRPOTrainer(
    model = model,
    args = training_args,
    # Pass the processor to handle multimodal inputs
    processing_class = tokenizer,
    reward_funcs = [
        structure_reward,
        atomic_action_reward_with_weights
    ],
    train_dataset = train_dataset,
)




trainer_stats = trainer.train()
model.save_pretrained("qwen_lora_model_grpo_7b") 
tokenizer.save_pretrained("qwen_lora_model_grpo_7b")
print("finish")
model.save_pretrained_merged("qwen_unsloth_finetune_grpo_7b", tokenizer,save_method = "merged_16bit_forced")
