"""Live local reads of frozen retrieved evidence; not an end-to-end benchmark."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from PRAGMOS_benchmark_LongMemEval import anchor_coverage, anchor_group_is_supported, factual_read_eligible
from PRAGMOS_evidence_compiler import parse_semantic_plan
from PRAGMOS_factual_read import protocol_identity, read_operation_obligations, run_factual_read


def restore_turns(record, memories):
    sessions = dict(zip(record['haystack_session_ids'], record['haystack_sessions']))
    turns = {}
    for memory in memories:
        ids = memory.get('source_turn_ids') or []
        if len(ids) != 1:
            continue
        quote = memory.get('source_quote', '')
        for turn in sessions.get(memory.get('source_session_id'), []):
            if quote and quote in turn['content'] and turn['role'] == memory.get('role'):
                turns[ids[0]] = SimpleNamespace(text=turn['content'], turn_id=ids[0],
                    session_id=memory['source_session_id'], role=turn['role'], speaker=memory.get('speaker'),
                    timestamp=memory.get('timestamp'))
                break
    return turns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-file', required=True)
    parser.add_argument('--trace', action='append', required=True)
    parser.add_argument('--question-id', action='append', default=[])
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n-ctx', type=int, default=2048)
    parser.add_argument('--n-threads', type=int, default=8)
    parser.add_argument('--n-gpu-layers', type=int, default=40)
    args = parser.parse_args()
    records = {r['question_id']: r for r in json.loads(Path(args.data_file).read_text())}
    rows = [r for path in args.trace for r in map(json.loads, Path(path).open())
            if not args.question_id or r['question_id'] in args.question_id]
    if args.question_id and set(args.question_id) != {r['question_id'] for r in rows}:
        raise ValueError('Requested probe IDs are missing from traces')
    from llama_cpp import Llama
    llm = Llama(model_path=args.model_path, n_ctx=args.n_ctx, n_threads=args.n_threads,
                n_gpu_layers=args.n_gpu_layers, seed=42, verbose=False)
    context = SimpleNamespace(llm=llm, count_tokens=lambda s: len(llm.tokenize(s.encode(), add_bos=True, special=True)))
    result = {'method': 'live_source_reads_with_frozen_retrieval; not_end_to_end; no_projected_scores',
              'protocol': protocol_identity(), 'model_path': str(Path(args.model_path).resolve()),
              'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                  for name in ['PRAGMOS_factual_read.py', 'PRAGMOS_factual_scope.py', 'PRAGMOS_benchmark_LongMemEval.py']}, 'cases': []}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for row in rows:
            # Gold/reference fields below are serialized only after the reader finishes.
            memories = row['final_memories'] + row['retrieval_candidate_pool']
            turns = restore_turns(records[row['question_id']], memories)
            raw = copy.deepcopy(row.get('semantic_query_plan') or {})
            raw['operands'] = [{**o, 'id': o['operand_id']} for o in raw.get('operands', [])]
            plan = parse_semantic_plan(raw, row['question'])
            profile = copy.deepcopy(row.get('base_query_profile') or {})
            profile['allow_distributed_actor_evidence'] = False
            enabled = factual_read_eligible(SimpleNamespace(pragmos_factual_mode='guarded'), row['query_intent'],
                profile, row['inferred_operation_plan'], row['state_selector_plan'], row['requested_answer_slot'], plan)
            report = {'status': 'NOT_APPLICABLE', 'answer': None, 'reason': 'existing_protected_capability', 'model_calls': 0}
            if enabled:
                report = run_factual_read(context, row['question'], memories, turns,
                    roles=row['query_intent'].get('preferred_source_roles') or ['user'],
                    anchors_supported=lambda text: anchor_coverage(profile, text)['complete'], n_ctx=args.n_ctx,
                    scope_groups=profile.get('required_anchor_groups'), group_supported=anchor_group_is_supported)
                if report['status'] == 'COMPLETE':
                    admission = read_operation_obligations(report, plan, turns)
                    report['source_obligations'] = admission
                    if not admission['accepted']:
                        report.update(status='NOT_PROVEN', answer=None, reason=admission['reason'])
            item = {'question_id': row['question_id'], 'question': row['question'], 'read': report,
                    'reference_for_review_only': row['answer'], 'saved_previous_answer': row['hypothesis']}
            result['cases'].append(item)
            output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + '\n')
            print(row['question_id'], report['status'], repr(report['answer']), report.get('reason'), flush=True)
    finally:
        llm.close()


if __name__ == '__main__':
    main()
