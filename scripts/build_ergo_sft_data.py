#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

SYSTEM_PROMPT = """You answer questions by iterative drafting. Your response canvas has six
fields. In [THOUGHT], reason about what the query needs and what evidence is
still missing. In [ACTION], write exactly one of: retrieve - you need
information from the document store; memory - you need something from earlier
in this session; finish - the evidence below is sufficient and [ANSWER] is
complete. In [ACTION_INPUT], name what you need to look up. In [CRITIQUE],
judge each numbered evidence item's usefulness for the query before relying
on it. In [ANSWER], draft the final answer. Ground every claim in the
EVIDENCE section; if evidence is insufficient, prefer retrieve over guessing."""

QUESTION_WORDS = {
    'what', 'which', 'who', 'where', 'when', 'why', 'how', 'is', 'are', 'was', 'were', 'did', 'does', 'do',
    'the', 'a', 'an', 'of', 'in', 'on', 'to', 'for', 'and', 'or', 'by', 'with', 'from', 'at', 'that', 'this'
}


def iter_rows(paths: list[Path]):
    for path in paths:
        table = pq.read_table(path, columns=['question', 'answer', 'supporting_facts', 'context'])
        for row in table.to_pylist():
            yield row


def supporting_sentences(row: dict) -> list[tuple[str, str]]:
    titles = row['supporting_facts']['title']
    sent_ids = row['supporting_facts']['sent_id']
    by_title = {title: sents for title, sents in zip(row['context']['title'], row['context']['sentences'])}
    out: list[tuple[str, str]] = []
    for title, sent_id in zip(titles, sent_ids):
        sents = by_title.get(title, [])
        if 0 <= int(sent_id) < len(sents):
            sent = ' '.join(str(sents[int(sent_id)]).split())
            if sent:
                out.append((str(title), sent))
    dedup: list[tuple[str, str]] = []
    seen = set()
    for item in out:
        if item not in seen:
            seen.add(item)
            dedup.append(item)
    return dedup[:4]


def evidence_digest(pairs: list[tuple[str, str]]) -> str:
    if not pairs:
        return '(none)'
    return '\n'.join(
        f'[{i}] {title}: {sent}'
        for i, (title, sent) in enumerate(pairs, start=1)
    )


def title_hints(question: str) -> str:
    toks = re.findall(r"[A-Za-z0-9'\-.]+", question)
    keep = []
    for tok in toks:
        low = tok.lower()
        if low in QUESTION_WORDS:
            continue
        if tok[:1].isupper() or any(ch.isdigit() for ch in tok):
            keep.append(tok)
    if not keep:
        keep = [t for t in toks if t.lower() not in QUESTION_WORDS][:8]
    return ' '.join(keep[:8]) or question


def retrieve_target(question: str) -> str:
    hint = title_hints(question)
    return (
        '[THOUGHT] The question cannot be answered from the current empty evidence. '
        'We should retrieve supporting facts and bridge entities before answering.\n'
        '[ACTION] retrieve\n'
        f'[ACTION_INPUT] {hint}\n'
        '[OBSERVATION] (none)\n'
        '[CRITIQUE] No evidence is available yet, so nothing can be trusted.\n'
        '[ANSWER]'
    )


def finish_target(answer: str, pairs: list[tuple[str, str]]) -> tuple[str, str]:
    steps = []
    for i, (title, sent) in enumerate(pairs, start=1):
        steps.append(f'Step {i}: In {title}, {sent}')
    thought = ' '.join(steps) if steps else 'Step 1: Use the provided evidence to answer the question.'
    critique_lines = []
    for i, (title, _sent) in enumerate(pairs, start=1):
        critique_lines.append(f'[{i}] useful - {title} contains evidence directly relevant to the question.')
    critique = ' '.join(critique_lines) if critique_lines else 'No evidence provided.'
    prior = 'Need evidence before answering.'
    target = (
        f'[THOUGHT] {thought}\n'
        '[ACTION] finish\n'
        '[ACTION_INPUT] none\n'
        f'[OBSERVATION] {evidence_digest(pairs)}\n'
        f'[CRITIQUE] {critique}\n'
        f'[ANSWER] {answer}'
    )
    return prior, target


def prompt(question: str, evidence: str, prior_cycles: str, cycle: int) -> str:
    return (
        f'{SYSTEM_PROMPT}\n\n'
        f'USER QUERY:\n{question}\n\n'
        f'EVIDENCE (cycle {cycle}):\n{evidence}\n\n'
        f'PRIOR CYCLES:\n{prior_cycles}\n\n'
        'RESPONSE:'
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    paths = [
        Path('datasets/hotpot_qa/distractor/train-00000-of-00002.parquet'),
        Path('datasets/hotpot_qa/distractor/train-00001-of-00002.parquet'),
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    n_examples = 0
    with out.open('w') as fh:
        for row in iter_rows(paths):
            if args.limit is not None and n_rows >= args.limit:
                break
            question = ' '.join(str(row['question']).split())
            answer = ' '.join(str(row['answer']).split())
            pairs = supporting_sentences(row)
            if not question or not answer:
                continue
            retrieve = {
                'task': 'retrieve',
                'prompt': prompt(question, '(none)', '(none)', 0),
                'response': retrieve_target(question),
            }
            prior, finish = finish_target(answer, pairs)
            finish_ex = {
                'task': 'finish',
                'prompt': prompt(question, evidence_digest(pairs), f'cycle 0 thought: {prior}', 1),
                'response': finish,
            }
            for ex in (retrieve, finish_ex):
                fh.write(json.dumps(ex) + '\n')
                n_examples += 1
            n_rows += 1
    print(f'wrote {n_examples} examples from {n_rows} hotpotqa rows to {out}')


if __name__ == '__main__':
    main()
