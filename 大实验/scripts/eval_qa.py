"""端到端问答评测器（双轨判分 + 带偏捕获 + 一致性）。

场景与判分规则：
- 对话场景（scene=dialogue）：检索记忆 → DeepSeek 回答 → 判分看
  「回答是否正确」（允许多条记忆协同答对，按 answer_keywords 规则匹配，
  争议样本 LLM 复核）；若检索内容含 conflict_text（纠错前旧说法）且
  回答错误 → 记「被带偏」。
- 知识库场景（scene=kb）：检索条款 → 回答 → 判分看「是否正确条款号/
  关键条文」出现在回答中（精确命中，不允许多条协同充数）。

用法（库）::

    from eval_qa import QAEvaluator
    ev = QAEvaluator(llm, tasks)
    result = ev.eval_adapter(adapter, tag="sme_kb_dynamic")

CLI::

    python scripts/eval_qa.py --tasks data/tasks_dialogue.json --adapter sme_rag ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_client import LLMClient  # noqa: E402

ANSWER_PROMPT = (
    "你是 AI 助手。以下是系统记忆中可能与问题相关的内容"
    "（可能过时、有噪音或与问题无关）：\n{memories}\n\n"
    "问题：{question}\n"
    "请直接给出简短回答（一句话），不要解释。如果记忆中没有相关信息，"
    "请只回答「不知道」。"
)

REVIEW_PROMPT = (
    "题目：{question}\n标准答案要点：{keywords}\n助手回答：{answer}\n\n"
    "助手回答是否包含标准答案要点？（只输出 是/否）"
)


def judge_by_keywords(answer: str, keywords: list[str]) -> bool:
    return any(k and k in answer for k in keywords)


class QAEvaluator:
    def __init__(self, llm: LLMClient, tasks: list[dict],
                 review_llm: LLMClient | None = None) -> None:
        self.llm = llm
        self.review_llm = review_llm or llm
        self.tasks = tasks
        self.calls = 0

    # ------------------------------------------------------------------ #
    def _memories_block(self, hits: list) -> str:
        lines = []
        for i, h in enumerate(hits, 1):
            lines.append(f"{i}. (相关度{h.score:.2f}) {h.text}")
        return "\n".join(lines) if lines else "（无）"

    def _ask(self, question: str, memories_block: str) -> str:
        prompt = ANSWER_PROMPT.format(memories=memories_block, question=question)
        self.calls += 1
        return self.llm.chat([{"role": "user", "content": prompt}],
                             temperature=0.2, max_tokens=200)

    def _review(self, question: str, keywords: list[str], answer: str) -> bool:
        prompt = REVIEW_PROMPT.format(question=question,
                                      keywords=" / ".join(keywords),
                                      answer=answer)
        self.calls += 1
        out = self.review_llm.chat([{"role": "user", "content": prompt}],
                                   temperature=0.0, max_tokens=8)
        return out.strip().startswith("是")

    # ------------------------------------------------------------------ #
    def eval_one(self, adapter, task: dict) -> dict:
        question = task["question"]
        scene = task.get("scene", "dialogue")
        top_k = int(task.get("top_k", 5))
        hits = adapter.search(question, top_k=top_k)
        texts = [h.text for h in hits]
        memories_block = self._memories_block(hits)
        answer = self._ask(question, memories_block)

        keywords = task.get("answer_keywords", [])
        if scene == "kb":
            # 知识库：正确条款号或关键条文必须出现
            correct = judge_by_keywords(answer, keywords) or any(
                k in answer for k in task.get("answer_keywords", []))
        else:
            correct = judge_by_keywords(answer, keywords)
            if not correct and keywords and answer.strip() and answer != "不知道":
                correct = self._review(question, keywords, answer)

        led_astray = False
        conflict = task.get("conflict_text")
        if conflict and not correct and any(
            conflict in t for t in texts):
            led_astray = True  # 检索到了旧说法（且可能带偏）

        return {
            "id": task.get("id", question[:20]),
            "scene": scene,
            "question": question,
            "answer": answer,
            "correct": bool(correct),
            "led_astray": led_astray,
            "retrieved": [t[:60] for t in texts[:3]],
            "hit_expect": any(
                task.get("expect_text") and (task["expect_text"] in t or t in task["expect_text"])
                for t in texts),
        }

    # ------------------------------------------------------------------ #
    def eval_adapter(self, adapter, tag: str = "") -> dict:
        results = []
        noise_cache: list[str] | None = None
        for task in self.tasks:
            inject = task.get("noise_inject")
            if inject:
                # 噪音压力题：评测前注入重复模板句（只注入一次）
                if noise_cache is None:
                    noise_cache = inject
                    for line in inject:
                        adapter.store(line, role="user")
                else:
                    pass  # 已注入
            pre = task.get("pre_turns")
            if pre:
                for line in pre:
                    adapter.store(line, role="user")
            try:
                results.append(self.eval_one(adapter, task))
            except Exception as exc:  # noqa: BLE001
                results.append({"id": task.get("id", "?"), "error": str(exc),
                                "correct": False, "led_astray": False})
        n = len(results)
        correct = sum(1 for r in results if r.get("correct"))
        astray = sum(1 for r in results if r.get("led_astray"))
        return {
            "tag": tag,
            "n": n,
            "correct": correct,
            "accuracy": round(correct / n, 4) if n else 0.0,
            "led_astray": astray,
            "details": results,
            "llm_calls": self.calls,
        }

    # ------------------------------------------------------------------ #
    def consistency(self, adapter, question: str, repeats: int = 3,
                    top_k: int = 5) -> dict:
        """同一问题连问 repeats 次，统计答案一致率。"""
        answers = []
        for _ in range(repeats):
            hits = adapter.search(question, top_k=top_k)
            answers.append(self._ask(question, self._memories_block(hits)))
        norm = [a.strip() for a in answers]
        same = sum(1 for a in norm[1:] if a == norm[0])
        return {
            "question": question,
            "answers": answers,
            "consistency": round(same / max(1, len(norm) - 1), 4),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="端到端问答评测")
    parser.add_argument("--tasks", required=True, help="任务集 JSON")
    parser.add_argument("--adapter", required=True, help="基线名（baselines.ALL_ADAPTERS）")
    parser.add_argument("--workspace", default="results/eval_ws",
                        help="基线工作区（记忆库落盘目录）")
    parser.add_argument("--out", default="", help="结果 JSON 输出")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    tasks = json.loads((root / args.tasks).read_text(encoding="utf-8"))
    task_list = tasks["tasks"] if isinstance(tasks, dict) else tasks

    from baselines import ALL_ADAPTERS
    from baselines.embedder import get_embedder

    llm = LLMClient()
    if not llm.configured:
        print("✗ 未设置 SME_LLM_API_KEY")
        sys.exit(1)
    adapter = ALL_ADAPTERS[args.adapter](workspace=str(root / args.workspace),
                                         embedder=get_embedder())
    ev = QAEvaluator(llm, task_list)
    result = ev.eval_adapter(adapter, tag=args.adapter)
    print(json.dumps({k: v for k, v in result.items() if k != "details"},
                     ensure_ascii=False, indent=1))
    if args.out:
        out = root / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print(f"✓ {out}")


if __name__ == "__main__":
    main()
