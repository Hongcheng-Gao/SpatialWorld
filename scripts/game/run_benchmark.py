#!/usr/bin/env python3
"""
           
       sokoban            ：
  - maze              :    1-20
  - block3d           :    1-20
  - maze3d_pro        :    1-25
  - rubik             :    1-20
  - snake             :    20  （   ）

           <output-dir>/<game_name>/      
        ，        --parallelism    
"""

import sys
import os
import argparse
import asyncio
import datetime
import time
from typing import List, Dict, Any

#           
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ──────────────────────────────────────────────────────────────────────────────
#      
# ──────────────────────────────────────────────────────────────────────────────
GAMES = [
    {
        "name":   "maze",
        "script": "maze_openai_evaluation.py",
        "mode":   "level",       #     
        "levels": list(range(1, 21)),
    },
    {
        "name":   "block3d",
        "script": "block3d_openai_evaluation.py",
        "mode":   "level",
        "levels": list(range(1, 21)),
    },
    {
        "name":   "maze3d_pro",
        "script": "maze3d_pro_openai_evaluation.py",
        "mode":   "level",
        "levels": list(range(1, 26)),
    },
    {
        "name":   "rubik",
        "script": "rubik_openai_evaluation.py",
        "mode":   "level",
        "levels": list(range(1, 21)),
    },
    {
        "name":   "snake",
        "script": "snake_openai_evaluation.py",
        "mode":   "run",         #      
        "runs":   20,
    },
]


# ──────────────────────────────────────────────────────────────────────────────
#     
# ──────────────────────────────────────────────────────────────────────────────
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='    MLLM         （  sokoban      ）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  :
  #             
  python batch_evaluation_allgames.py --output-dir results/exp1

  #         
  python batch_evaluation_allgames.py --output-dir results/exp1 --model gpt-4o --parallelism 6

  #         （    ）
  python batch_evaluation_allgames.py --output-dir results/exp1 --games maze,rubik

  #    API   
  python batch_evaluation_allgames.py --output-dir results/exp1 \\
      --model gpt-4o --api-base-url http. --api-key sk-xxx
"""
    )

    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='     （  : logs/allgames_<timestamp>）'
    )
    parser.add_argument(
        '--parallelism', type=int, default=4,
        help='          ，          （  : 4）'
    )
    parser.add_argument(
        '--games', type=str, default=None,
        help='       ，       （  :   ） '
             f'  : {", ".join(g["name"] for g in GAMES)}'
    )

    #          
    parser.add_argument('--max-steps', type=int, default=None,
                        help='      （  :         ）')
    parser.add_argument('--model', type=str, default=None,
                        help='    （  :         ）')
    parser.add_argument('--api-base-url', type=str, default=None,
                        help='OpenAI API    URL（  :         ）')
    parser.add_argument('--api-key', type=str, default=None,
                        help='OpenAI API   （  :         ）')
    parser.add_argument('--retry-times', type=int, default=None,
                        help='API       （  :         ）')

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
#     
# ──────────────────────────────────────────────────────────────────────────────
def _append_common_args(cmd: List[str], args: argparse.Namespace):
    if args.max_steps is not None:
        cmd += ['--max-steps', str(args.max_steps)]
    if args.model is not None:
        cmd += ['--model', args.model]
    if args.api_base_url is not None:
        cmd += ['--api-base-url', args.api_base_url]
    if args.api_key is not None:
        cmd += ['--api-key', args.api_key]
    if args.retry_times is not None:
        cmd += ['--retry-times', str(args.retry_times)]


def build_cmd_level(script_path: str, level: int, log_dir: str,
                    args: argparse.Namespace) -> List[str]:
    cmd = [sys.executable, script_path, '--level', str(level), '--log-dir', log_dir]
    _append_common_args(cmd, args)
    return cmd


def build_cmd_run(script_path: str, log_dir: str,
                  args: argparse.Namespace) -> List[str]:
    cmd = [sys.executable, script_path, '--log-dir', log_dir]
    _append_common_args(cmd, args)
    return cmd


# ──────────────────────────────────────────────────────────────────────────────
# Resume   
# ──────────────────────────────────────────────────────────────────────────────
def is_task_completed(log_dir: str) -> bool:
    """         ：log_dir     *_evaluation_summary_*.json          """
    if not os.path.exists(log_dir):
        return False
    for fname in os.listdir(log_dir):
        if '_evaluation_summary_' in fname and fname.endswith('.json'):
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
#     
# ──────────────────────────────────────────────────────────────────────────────
async def run_task(sem: asyncio.Semaphore, cmd: List[str], label: str,
                   log_dir: str) -> Dict[str, Any]:
    """   semaphore           ，       """
    os.makedirs(log_dir, exist_ok=True)

    async with sem:
        start_time = time.time()
        print(f"  [{label}]   ")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await proc.communicate()
            duration = time.time() - start_time

            status = "✅   " if proc.returncode == 0 else "❌   "
            print(f"  [{label}] {status} |   : {duration:.1f}s |    : {proc.returncode}")

            return {
                'label':      label,
                'returncode': proc.returncode,
                'duration':   duration,
                'stdout':     stdout_bytes.decode('utf-8', errors='replace'),
                'stderr':     stderr_bytes.decode('utf-8', errors='replace'),
            }

        except Exception as e:
            duration = time.time() - start_time
            print(f"  [{label}] ❌   : {e}")
            return {
                'label':      label,
                'returncode': -1,
                'duration':   duration,
                'stdout':     '',
                'stderr':     str(e),
            }


# ──────────────────────────────────────────────────────────────────────────────
#    
# ──────────────────────────────────────────────────────────────────────────────
async def main():
    args = parse_arguments()
    scripts_dir = os.path.dirname(os.path.abspath(__file__))

    #        
    if args.output_dir:
        root_output_dir = args.output_dir
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        root_output_dir = os.path.join("logs", f"allgames_{ts}")
    os.makedirs(root_output_dir, exist_ok=True)

    #         
    selected_names = None
    if args.games:
        selected_names = {g.strip() for g in args.games.split(',')}
        invalid = selected_names - {g['name'] for g in GAMES}
        if invalid:
            print(f"❌       : {invalid}")
            print(f"     : {', '.join(g['name'] for g in GAMES)}")
            sys.exit(1)

    games_to_run = [g for g in GAMES if selected_names is None or g['name'] in selected_names]

    #     
    print("=" * 60)
    print("    MLLM     ")
    print("=" * 60)
    print(f"    :   {len(games_to_run)}")
    print(f"   :     {args.parallelism}（  ，      ）")
    print(f"     : {root_output_dir}")
    if args.model:
        print(f"  :       {args.model}")
    if args.max_steps:
        print(f"    :   {args.max_steps}")
    print()
    for g in games_to_run:
        if g['mode'] == 'level':
            print(f"  {g['name']:<14}          {g['levels'][0]}-{g['levels'][-1]}    {len(g['levels'])}  ")
        else:
            print(f"  {g['name']:<14}          {g['runs']}  ")
    print("=" * 60)

    #            ，          
    sem = asyncio.Semaphore(args.parallelism)
    all_tasks: List[Dict] = []          # {'game', 'label', 'log_dir', 'cmd'}
    skipped_games: List[str] = []
    total_start = time.time()

    for game in games_to_run:
        script_path = os.path.join(scripts_dir, game['script'])
        if not os.path.exists(script_path):
            print(f"[{game['name']}] ❌        : {script_path}，   ")
            skipped_games.append(game['name'])
            continue

        game_output_dir = os.path.join(root_output_dir, game['name'])
        os.makedirs(game_output_dir, exist_ok=True)

        if game['mode'] == 'level':
            for lv in game['levels']:
                log_dir = os.path.join(game_output_dir, f"level_{lv:02d}")
                all_tasks.append({
                    'game':    game['name'],
                    'label':   f"{game['name']}/Level_{lv:02d}",
                    'log_dir': log_dir,
                    'cmd':     build_cmd_level(script_path, lv, log_dir, args),
                })
        else:
            for i in range(game['runs']):
                log_dir = os.path.join(game_output_dir, f"run_{i+1:03d}")
                all_tasks.append({
                    'game':    game['name'],
                    'label':   f"{game['name']}/run_{i+1:03d}",
                    'log_dir': log_dir,
                    'cmd':     build_cmd_run(script_path, log_dir, args),
                })

    total_count = len(all_tasks)

    # Resume：         
    pending_tasks = [t for t in all_tasks if not is_task_completed(t['log_dir'])]
    skipped_count = total_count - len(pending_tasks)
    if skipped_count > 0:
        print(f"\n🔄 Resume   ：    {skipped_count}       ，  ；   {len(pending_tasks)}      ")
        for t in all_tasks:
            if is_task_completed(t['log_dir']):
                print(f"    [   ，  ] {t['label']}")
    else:
        print(f"\n         ，  {total_count}    ，      （     : {args.parallelism}）")
    print("=" * 60)

    if not pending_tasks:
        print("✅         ，       ")
        return

    #         
    coros = [run_task(sem, t['cmd'], t['label'], t['log_dir']) for t in pending_tasks]
    all_results_flat = await asyncio.gather(*coros)

    #        （            ）
    game_results: Dict[str, List[Dict]] = {}
    for task, result in zip(pending_tasks, all_results_flat):
        game_results.setdefault(task['game'], []).append(result)

    game_summaries: List[Dict] = []
    for game in games_to_run:
        if game['name'] in skipped_games:
            game_summaries.append({
                'name': game['name'], 'total': 0, 'ok': 0,
                'fail': 0, 'skipped': True,
            })
            continue
        results = game_results.get(game['name'], [])
        ok   = sum(1 for r in results if r['returncode'] == 0)
        fail = len(results) - ok
        errors = [r['label'] for r in results if r['returncode'] != 0]
        if errors:
            print(f"[{game['name']}]     （        resume）: {errors}")

        #                
        if game['mode'] == 'level':
            game_dir = os.path.join(root_output_dir, game['name'])
            pre_done = sum(
                1 for lv in game['levels']
                if is_task_completed(os.path.join(game_dir, f"level_{lv:02d}"))
                and not any(t['label'] == f"{game['name']}/Level_{lv:02d}" for t in pending_tasks)
            )
        else:
            game_dir = os.path.join(root_output_dir, game['name'])
            pre_done = sum(
                1 for i in range(game['runs'])
                if is_task_completed(os.path.join(game_dir, f"run_{i+1:03d}"))
                and not any(t['label'] == f"{game['name']}/run_{i+1:03d}" for t in pending_tasks)
            )

        game_summaries.append({
            'name': game['name'], 'total': len(results) + pre_done,
            'ok': ok + pre_done, 'fail': fail, 'skipped': False,
        })

    total_duration = time.time() - total_start

    print(f"\n{'='*60}")
    print("          ")
    print("=" * 60)
    print(f"   :     {total_duration:.1f}s")
    print(f"     : {root_output_dir}")
    print()
    print("    analyze_and_export.py       ：")
    print()
    analyze_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analyze_and_export.py")
    print(f"  #            &     ")
    print(f"  python {analyze_script} {root_output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
