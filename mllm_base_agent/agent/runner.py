"""Internal Think -> Act -> Evaluate -> Final runner.

This replaces the previous external graph state machine with a plain Python loop while
keeping the public `.invoke()` and `.stream()` shape used by legacy scripts.
"""

from __future__ import annotations

import base64
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

from mllm_base_agent.agent.state import AgentState
from actions.response_parser import parse_vlm_response
from actions.max_steps import resolve_max_steps_from_task
from mllm_base_agent.llm.messages import AIMessage, HumanMessage, SystemMessage
from mllm_base_agent.prompts import get_system_prompt

LOCAL_RETRY_CONFIG = {
    'max_retries': 3,
    'api_max_retries': 5,
    'retry_delay': 2,
    'api_retry_delay': 5,
}
MODEL_HISTORY_TURNS = 29

EXTERNAL_FAILURE_TYPES = {'api_error', 'env_error', 'external_error'}
MODEL_FAILURE_TYPES = {'parse_error', 'action_error', 'model_error'}


class GraphRecursionError(RuntimeError):
    """Compatibility exception for old graph error handling."""


class ParseRetryError(Exception):
    pass


class APIRetryError(Exception):
    pass


def _success_value_for_failure_type(failure_type: Optional[str]) -> Optional[bool]:
    if failure_type in EXTERNAL_FAILURE_TYPES:
        return None
    if failure_type:
        return False
    return None


def _normalize_token_usage(raw_usage: Optional[dict]) -> Dict[str, int]:
    usage = raw_usage or {}

    def to_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    prompt_tokens = to_int(usage.get('prompt_tokens'))
    completion_tokens = to_int(usage.get('completion_tokens'))
    total_tokens = to_int(usage.get('total_tokens')) or prompt_tokens + completion_tokens
    api_calls = to_int(usage.get('api_calls')) or (1 if total_tokens else 0)
    return {
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens': total_tokens,
        'api_calls': api_calls,
    }


def _extract_token_usage_from_response(response: Any) -> Dict[str, int]:
    if response is None:
        return _normalize_token_usage({})
    metadata = getattr(response, 'response_metadata', None) or {}
    if isinstance(metadata, dict) and metadata.get('token_usage'):
        return _normalize_token_usage(metadata['token_usage'])
    usage_metadata = getattr(response, 'usage_metadata', None)
    if isinstance(usage_metadata, dict):
        return _normalize_token_usage(usage_metadata)
    additional_kwargs = getattr(response, 'additional_kwargs', None) or {}
    if isinstance(additional_kwargs, dict):
        return _normalize_token_usage(additional_kwargs.get('token_usage') or additional_kwargs.get('usage'))
    return _normalize_token_usage({})


def _accumulate_token_usage(state: AgentState, token_usage: Dict[str, int]) -> None:
    if 'token_usage' not in state or not isinstance(state.get('token_usage'), dict):
        state['token_usage'] = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'api_calls': 0}
    usage = state['token_usage']
    normalized = _normalize_token_usage(token_usage)
    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'api_calls'):
        usage[key] = int(usage.get(key, 0) or 0) + normalized[key]


def _read_image_as_data_url(image_path: str, max_retries: int, retry_delay: int) -> str:
    last_error: Optional[BaseException] = None
    for attempt in range(max_retries):
        try:
            with open(image_path, 'rb') as handle:
                image_data = base64.b64encode(handle.read()).decode('utf-8')
            return f'data:image/png;base64,{image_data}'
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    raise OSError(f'Failed to read image after {max_retries} attempts: {last_error}')


def _build_messages(state: AgentState, image_url: str) -> list:
    config = state.get('config') or {}
    env_type = str(config.get('env', {}).get('type', 'ai2thor')).lower()
    context_config = config.get('context_management', {}) or {}
    enable_summary = bool(context_config.get('enable_long_term_summary', False))
    task_cfg = config.get('task', {}) or {}
    actions_cfg = config.get('actions', {}) or {}
    executor_type = state.get('executor_type')
    input_modality = task_cfg.get('input_modality') or state.get('input_modality')
    navigation_mode = actions_cfg.get('navigation_mode', 'discrete')
    vh_objects = None
    env = state.get('env')
    if env_type == 'virtualhome' and env is not None and hasattr(env, 'get_scene_interactable_object_types'):
        try:
            vh_objects = env.get_scene_interactable_object_types()
        except Exception:
            vh_objects = None
    success_criteria_block = state.get(
        'success_criteria_block',
        'Complete the task according to the instruction. Use EndTask(DONE) only after confirming success.',
    )
    prompt = get_system_prompt(
        env_type,
        enable_summary=enable_summary,
        executor_type=executor_type,
        input_modality=input_modality,
        navigation_mode=navigation_mode,
        virtualhome_interactable_object_types=vh_objects,
    ).format(
        task_prompt=state.get('task_prompt', 'Complete the task.'),
        success_criteria_block=success_criteria_block,
    )
    messages = [SystemMessage(content=prompt)]
    long_term_summary = state.get('long_term_summary', '')
    history = (state.get('short_term_history', []) or [])[-MODEL_HISTORY_TURNS:]
    for idx, entry in enumerate(history):
        content = []
        if idx == 0 and enable_summary and long_term_summary.strip():
            content.append({'type': 'text', 'text': f'**Previous Exploration Summary (Long-term Memory):**\n{long_term_summary}\n\n---\n'})
        content.append({'type': 'text', 'text': f"Step {entry.get('step', '?')}"})
        hist_image = entry.get('image_path')
        if hist_image and os.path.exists(hist_image):
            try:
                content.append({'type': 'image_url', 'image_url': {'url': _read_image_as_data_url(hist_image, 1, 0)}})
            except Exception:
                content.append({'type': 'text', 'text': '[Image unavailable]'})
        messages.append(HumanMessage(content=content))
        messages.append(AIMessage(content=entry.get('raw_response', '')))
    current_content = []
    if not history and enable_summary and long_term_summary.strip():
        current_content.append({'type': 'text', 'text': f'**Previous Exploration Summary (Long-term Memory):**\n{long_term_summary}\n\n---\n'})
    goal_image_path = state.get('goal_image_path')
    if goal_image_path and state.get('step_count', 0) == 0:
        try:
            current_content.append({'type': 'text', 'text': '**Goal Image (your target destination):**'})
            current_content.append({'type': 'image_url', 'image_url': {'url': _read_image_as_data_url(goal_image_path, 1, 0)}})
        except Exception:
            pass
    current_content.append({'type': 'text', 'text': f"Step {state.get('step_count', 0)}"})
    current_content.append({'type': 'image_url', 'image_url': {'url': image_url}})
    messages.append(HumanMessage(content=current_content))
    return messages


def think_node(state: AgentState) -> AgentState:
    observation = state['observation']
    vlm = state['vlm']
    state.setdefault('structured_trajectory', [])
    state.setdefault('conversation_history', [])
    state.setdefault('short_term_history', [])
    state.setdefault('long_term_summary', '')
    max_retries = LOCAL_RETRY_CONFIG['max_retries']
    api_max_retries = LOCAL_RETRY_CONFIG.get('api_max_retries', max_retries)
    retry_delay = LOCAL_RETRY_CONFIG['retry_delay']
    api_retry_delay = LOCAL_RETRY_CONFIG['api_retry_delay']
    config = state.get('config') or {}
    env_type = str(config.get('env', {}).get('type', 'ai2thor')).lower()
    enable_summary = bool((config.get('context_management') or {}).get('enable_long_term_summary', False))

    try:
        image_url = _read_image_as_data_url(observation.image_path, max_retries, retry_delay)
    except Exception as exc:
        state['failure_type'] = 'external_error'
        state['fail_reason'] = str(exc)
        state['should_continue'] = False
        state['success'] = None
        return state

    messages = _build_messages(state, image_url)
    last_error: Optional[BaseException] = None
    response_text: Optional[str] = None
    step_token_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'api_calls': 0}

    for api_attempt in range(api_max_retries):
        try:
            response = vlm.invoke(messages)
            response_text = getattr(response, 'content', str(response))
            usage = _extract_token_usage_from_response(response)
            _accumulate_token_usage(state, usage)
            for key in step_token_usage:
                step_token_usage[key] += usage.get(key, 0)
            break
        except Exception as exc:
            last_error = exc
            if api_attempt < api_max_retries - 1:
                time.sleep(api_retry_delay)
            else:
                state['failure_type'] = 'api_error'
                state['fail_reason'] = f'API error after {api_max_retries} attempts: {exc}'
                state['should_continue'] = False
                state['success'] = None
                state['structured_trajectory'].append({
                    'step': state.get('step_count', 0),
                    'thinking': '',
                    'action_string': '',
                    'action': {},
                    'raw_response': '',
                    'llm_token_usage': dict(step_token_usage),
                    'parse_error': state['fail_reason'],
                    'failure_type': 'api_error',
                    'reward': None,
                    'image_path': observation.image_path,
                })
                return state

    for parse_attempt in range(max_retries):
        try:
            parsed = parse_vlm_response(
                response_text or '',
                enable_summary=enable_summary,
                env_type=env_type,
                executor_type=state.get('executor_type'),
            )
            action = parsed['parsed_action']
            is_completion = action.get('action_type') == 'task_completion'
            state['next_action'] = action
            state['should_continue'] = not is_completion
            state['task_done_by_model'] = action.get('action_name') == 'DONE'
            state['task_fail_by_model'] = action.get('action_name') == 'FAIL'
            if enable_summary and parsed.get('updated_summary'):
                state['long_term_summary'] = parsed['updated_summary']
            trajectory_step = {
                'step': state.get('step_count', 0),
                'thinking': parsed['thinking_text'],
                'action_string': parsed['action_string'],
                'action': action,
                'updated_summary': parsed.get('updated_summary', ''),
                'raw_response': (response_text or '')[:2000],
                'llm_token_usage': dict(step_token_usage),
                'parse_error': None,
                'retry_count': parse_attempt,
                'reward': None,
                'observation_summary': None,
                'image_path': observation.image_path,
            }
            state['structured_trajectory'].append(trajectory_step)
            state['conversation_history'].append({
                'step': state.get('step_count', 0),
                'user_message': f"Step {state.get('step_count', 0)}",
                'assistant_response': response_text or '',
                'llm_token_usage': dict(step_token_usage),
                'action_executed': '',
                'reward': None,
                'error_message': None,
            })
            state['failure_type'] = None
            return state
        except Exception as exc:
            last_error = exc
            if parse_attempt < max_retries - 1:
                try:
                    response = vlm.invoke(messages)
                    response_text = getattr(response, 'content', str(response))
                    usage = _extract_token_usage_from_response(response)
                    _accumulate_token_usage(state, usage)
                    for key in step_token_usage:
                        step_token_usage[key] += usage.get(key, 0)
                except Exception as api_exc:
                    last_error = api_exc
                time.sleep(retry_delay)

    state['failure_type'] = 'parse_error'
    state['fail_reason'] = f"Step {state.get('step_count', 0)} parse failed after {max_retries} retries: {last_error}"
    state['should_continue'] = False
    state['success'] = False
    state['structured_trajectory'].append({
        'step': state.get('step_count', 0),
        'thinking': '',
        'action_string': '',
        'action': {},
        'raw_response': (response_text or '')[:2000],
        'llm_token_usage': dict(step_token_usage),
        'parse_error': state['fail_reason'],
        'failure_type': 'parse_error',
        'reward': None,
        'image_path': observation.image_path,
    })
    return state


def act_node(state: AgentState) -> AgentState:
    if state.get('think_failed') or state.get('failure_type') in {'api_error', 'parse_error', 'external_error'}:
        return state
    action = state.get('next_action')
    if not action:
        state['failure_type'] = 'action_error'
        state['fail_reason'] = 'No action available from think_node'
        state['should_continue'] = False
        state['success'] = False
        return state
    action_type = action.get('action_type')
    action_name = action.get('action_name')
    observation = state.get('observation')
    error_message = None
    if action_type == 'task_completion':
        state['step_count'] = int(state.get('step_count', 0) or 0) + 1
    else:
        try:
            observation, error_message = state['env'].step_with_action_dict(action)
            if observation is None:
                state['failure_type'] = 'action_error'
                state['fail_reason'] = f'Model output invalid action: {error_message}'
                state['should_continue'] = False
                state['success'] = False
                return state
            state['observation'] = observation
            state['step_count'] = int(state.get('step_count', 0) or 0) + 1
        except Exception as exc:
            state['failure_type'] = 'env_error'
            state['fail_reason'] = f'Environment exception: {exc}'
            state['should_continue'] = False
            state['success'] = None
            return state

    if state.get('structured_trajectory'):
        last_step = state['structured_trajectory'][-1]
        last_step['reward'] = 0 if action_type == 'task_completion' else getattr(observation, 'reward', 0)
        last_step['observation_summary'] = f'Task completion: {action_name}' if action_type == 'task_completion' else getattr(observation, 'text_state', '')
        last_step['error_message'] = error_message
    if state.get('conversation_history'):
        last_conv = state['conversation_history'][-1]
        last_conv['action_executed'] = action_name
        last_conv['reward'] = 0 if action_type == 'task_completion' else getattr(observation, 'reward', 0)
        last_conv['error_message'] = error_message

    context = (state.get('config') or {}).get('context_management') or {}
    configured_history = int(context.get('short_term_history_window_size', MODEL_HISTORY_TURNS) or MODEL_HISTORY_TURNS)
    max_history = min(MODEL_HISTORY_TURNS, max(0, configured_history))
    action_string = action_name
    if action.get('object_type'):
        action_string = f"{action_name}({action.get('object_type')})"
    state.setdefault('short_term_history', []).append({
        'step': int(state.get('step_count', 1) or 1) - 1,
        'action_string': action_string,
        'reward': 0 if action_type == 'task_completion' else getattr(observation, 'reward', 0),
        'image_path': getattr(observation, 'image_path', None),
        'raw_response': state.get('structured_trajectory', [{}])[-1].get('raw_response', ''),
        'error_message': error_message,
    })
    if len(state['short_term_history']) > max_history:
        state['short_term_history'] = state['short_term_history'][-max_history:]
    return state


def _count_consecutive_failures(state: AgentState) -> int:
    count = 0
    for step in reversed(state.get('structured_trajectory', [])):
        reward = step.get('reward')
        if reward is None or reward < 0.05:
            count += 1
        else:
            break
    return count


def perform_final_evaluation(state: AgentState = None, *, env=None, task_config: dict = None, observation=None) -> tuple:
    try:
        from evaluation import create_evaluator_from_config
        if state is not None:
            config = state.get('config') or {}
            task_config = config.get('task') or {}
            env = state.get('env')
            observation = state.get('observation')
        if not task_config or observation is None or not getattr(observation, 'metadata', None):
            return False, 0.0
        evaluator = create_evaluator_from_config(task_config)
        score = evaluator.evaluate(env, observation.metadata)
        return score >= 1.0, score
    except Exception:
        return False, 0.0


def evaluate_node(state: AgentState) -> AgentState:
    if state.get('task_done_by_model'):
        success, _score = perform_final_evaluation(state)
        state['success'] = success
        state['fail_reason'] = None if success else 'Model claimed DONE but success conditions not met'
        state['should_continue'] = False
        return state
    if state.get('task_fail_by_model'):
        state['success'] = False
        state['fail_reason'] = 'Model determined task cannot be completed or refused to continue'
        state['should_continue'] = False
        return state
    if state.get('should_continue') is False:
        return state
    if int(state.get('step_count', 0) or 0) >= int(state.get('max_steps', 30) or 30):
        state['success'] = False
        state['fail_reason'] = f"Reached maximum step limit ({state.get('max_steps')} steps)"
        state['should_continue'] = False
        return state
    state['should_continue'] = True
    return state


def final_node(state: AgentState) -> AgentState:
    output_dir = state.get('run_output_dir')
    if not output_dir:
        return state
    os.makedirs(output_dir, exist_ok=True)
    env = state.get('env')
    observation = state.get('observation')
    scene_name = getattr(env, 'scene', 'UnknownScene')
    metadata = getattr(observation, 'metadata', None) or {}
    if isinstance(metadata, dict):
        scene_name = metadata.get('sceneName', scene_name)
    task_name = ((state.get('config') or {}).get('task') or {}).get('name', 'task') or 'task'
    safe_scene = str(scene_name).replace(' ', '_').replace('/', '_')[:80]
    safe_task = str(task_name).replace(' ', '_').replace('/', '_')[:80]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(output_dir, f'episode_{safe_scene}_{safe_task}_{timestamp}.json')
    episode = {
        'task': state.get('task_prompt', ''),
        'scene': scene_name,
        'success': state.get('success', False),
        'fail_reason': state.get('fail_reason'),
        'failure_type': state.get('failure_type'),
        'step_count': state.get('step_count', 0),
        'max_steps': state.get('max_steps', 0),
        'action_sequence': env.get_action_sequence() if hasattr(env, 'get_action_sequence') else '(no action records)',
        'trajectory': [
            {
                'step': item.get('step'),
                'thinking': item.get('thinking'),
                'action_string': item.get('action_string'),
                'llm_token_usage': item.get('llm_token_usage'),
                'reward': item.get('reward'),
                'error_message': item.get('error_message'),
            }
            for item in state.get('structured_trajectory', [])
        ],
        'timestamp': datetime.now().isoformat(),
        'metadata': {
            'total_reward': sum((item.get('reward') or 0) for item in state.get('structured_trajectory', [])),
            'parse_errors_count': sum(1 for item in state.get('structured_trajectory', []) if item.get('parse_error')),
            'token_usage': state.get('token_usage', {}),
        },
    }
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(episode, handle, ensure_ascii=False, indent=2)
    state['episode_log_path'] = path
    return state


class AgentRunner:
    def __init__(self, recursion_limit: int = 1000) -> None:
        self.recursion_limit = recursion_limit

    def stream(self, initial_state: AgentState, config: Optional[dict] = None) -> Iterable[Dict[str, AgentState]]:
        state = initial_state
        task_cfg = (state.get('config') or {}).get('task') or {}
        if task_cfg:
            state['max_steps'] = resolve_max_steps_from_task(task_cfg, int(state.get('max_steps', 30) or 30))
        limit = int((config or {}).get('recursion_limit', self.recursion_limit) or self.recursion_limit)
        iterations = 0
        while True:
            if iterations >= limit:
                raise GraphRecursionError(f'Recursion limit reached: {limit}')
            iterations += 1
            state = think_node(state)
            yield {'think': state}
            state = act_node(state)
            yield {'act': state}
            state = evaluate_node(state)
            yield {'evaluate': state}
            if not state.get('should_continue', True):
                state = final_node(state)
                yield {'final': state}
                break
        self.last_state = state

    def invoke(self, initial_state: AgentState, config: Optional[dict] = None) -> AgentState:
        final_state = initial_state
        for chunk in self.stream(initial_state, config=config):
            for update in chunk.values():
                final_state = update
        return final_state


def create_agent_graph() -> AgentRunner:
    return AgentRunner()


_parse_action_string = None
try:
    from actions.parser import parse_action_string as _parse_action_string
except Exception:
    pass

parse_action_string = _parse_action_string
_perform_final_evaluation = lambda state: perform_final_evaluation(state)[0]
execute_action = lambda env, action_dict: (*env.step_with_action_dict(action_dict), False) if action_dict.get('action_type') != 'task_completion' else (None, None, True)
