#!/usr/bin/env python3
"""
OpenAI API      
            API                

       API  ：
1. client.responses.create -   Gemini 3-Pro-Preview   
2. client.chat.completions.create -   Gemini-2.5-flash   
"""

import base64
import io
import time
import logging
from typing import List, Dict, Any, Optional, Union, Literal
from PIL import Image
from openai import OpenAI

_logger = logging.getLogger(__name__)


#    API       
#               
MODEL_API_MAPPING = {
    # responses.create      
    "Gemini 3-Pro-Preview": "responses_jd",
    "Gemini-3-Flash-Preview": "responses_jd",
    "Gemini-3.1-Pro-Preview":"responses_jd",
    "gpt-5": "chat",

    # chat.completions.create      
    "Gemini-2.5-flash": "chat",
    "Gemini-2.5-pro":"chat"

}

#       
DEFAULT_API_TYPE = "chat"
MODEL_HISTORY_TURNS = 29

# API      
ApiType = Literal["responses", "chat", "responses_jd"]


def get_api_type_for_model(model: str) -> ApiType:
    """
            API    

    Args:
        model:     

    Returns:
        API     ("responses"   "chat")
    """
    #       
    if model in MODEL_API_MAPPING:
        return MODEL_API_MAPPING[model]

    #       （      ）
    model_lower = model.lower()
    for key, api_type in MODEL_API_MAPPING.items():
        if key.lower() in model_lower or model_lower in key.lower():
            return api_type

    #     responses  
    return DEFAULT_API_TYPE


def convert_to_responses_jd_format(conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
            responses_jd  （     ）

    Args:
        conversation_history:     

    Returns:
        responses_jd   extra_body  
    """
    #       （     ）
    system_instruction = None
    contents = []

    for item in conversation_history:
        role = item["role"]
        content = item["content"]

        if role == "system":
            #     
            if isinstance(content, list):
                #       
                text_parts = []
                for content_item in content:
                    if isinstance(content_item, dict):
                        if content_item.get("type") == "input_text":
                            text_parts.append(content_item.get("text", ""))
                        elif content_item.get("type") == "output_text":
                            text_parts.append(content_item.get("text", ""))
                        else:
                            text_parts.append(str(content_item))
                    else:
                        text_parts.append(str(content_item))
                system_text = " ".join(text_parts)
            elif isinstance(content, dict):
                if content.get("type") == "input_text":
                    system_text = content.get("text", "")
                elif content.get("type") == "output_text":
                    system_text = content.get("text", "")
                else:
                    system_text = str(content)
            else:
                system_text = str(content)

            system_instruction = {
                "parts": [{"text": system_text}]
            }
        else:
            #        
            parts = []

            if isinstance(content, list):
                for content_item in content:
                    if isinstance(content_item, dict):
                        if content_item.get("type") == "input_text":
                            parts.append({"text": str(content_item.get("text", ""))})
                        elif content_item.get("type") == "output_text":
                            parts.append({"text": str(content_item.get("text", ""))})
                        elif content_item.get("type") == "input_image":
                            #      -      inlineData  
                            image_url = content_item.get("image_url", "")
                            #   base64  （  data:image/png;base64,  ）
                            if image_url.startswith("data:image/"):
                                #   MIME   base64  
                                url_parts = image_url.split(";base64,")
                                if len(url_parts) == 2:
                                    mime_type = url_parts[0].replace("data:", "")
                                    base64_data = url_parts[1]
                                    parts.append({
                                        "inlineData": {
                                            "mimeType": mime_type,
                                            "data": base64_data
                                        }
                                    })
                                else:
                                    #        ，       
                                    parts.append({"text": f"[Image: {image_url[:50]}...]"})
                            else:
                                #     base64  ，       
                                parts.append({"text": f"[Image: {image_url[:50]}...]"})
                        else:
                            parts.append({"text": str(content_item)})
                    else:
                        parts.append({"text": str(content_item)})
            elif isinstance(content, dict):
                if content.get("type") == "input_text":
                    parts.append({"text": str(content.get("text", ""))})
                elif content.get("type") == "output_text":
                    parts.append({"text": str(content.get("text", ""))})
                elif content.get("type") == "input_image":
                    #      -      inlineData  
                    image_url = content.get("image_url", "")
                    #   base64  （  data:image/png;base64,  ）
                    if image_url.startswith("data:image/"):
                        #   MIME   base64  
                        url_parts = image_url.split(";base64,")
                        if len(url_parts) == 2:
                            mime_type = url_parts[0].replace("data:", "")
                            base64_data = url_parts[1]
                            parts.append({
                                "inlineData": {
                                    "mimeType": mime_type,
                                    "data": base64_data
                                }
                            })
                        else:
                            #        ，       
                            parts.append({"text": f"[Image: {image_url[:50]}...]"})
                    else:
                        #     base64  ，       
                        parts.append({"text": f"[Image: {image_url[:50]}...]"})
                else:
                    parts.append({"text": str(content)})
            else:
                parts.append({"text": str(content)})

            contents.append({
                "role": role,
                "parts": parts
            })

    #   extra_body
    extra_body = {
        "contents": contents
    }

    if system_instruction:
        extra_body["system_instruction"] = system_instruction

    #          
    extra_body["generation_config"] = {
        "response_modalities": ["TEXT"]
    }

    return extra_body


def convert_to_chat_format(conversation_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
     responses          chat  

    Args:
        conversation_history: responses       

    Returns:
        chat       
    """
    chat_history = []

    for item in conversation_history:
        role = item["role"]
        content = item["content"]

        #        
        if isinstance(content, list):
            #  responses     chat  
            chat_content = []
            for content_item in content:
                if isinstance(content_item, dict):
                    if content_item.get("type") == "input_text":
                        #     ：responses   -> chat  
                        chat_content.append({
                            "type": "text",
                            "text": str(content_item.get("text", ""))
                        })
                    elif content_item.get("type") == "input_image":
                        #     ：responses   -> chat  
                        # responses  : {"type": "input_image", "image_url": "data:image/png;base64,..."}
                        # chat  : {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
                        image_url = content_item.get("image_url", "")
                        chat_content.append({
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        })
                    elif content_item.get("type") == "output_text":
                        #     ：responses   -> chat  
                        chat_content.append({
                            "type": "text",
                            "text": str(content_item.get("text", ""))
                        })
                    else:
                        #          ，     
                        chat_content.append({
                            "type": "text",
                            "text": str(content_item)
                        })
                else:
                    #       ，     
                    chat_content.append({
                        "type": "text",
                        "text": str(content_item)
                    })
        elif isinstance(content, dict):
            #   content   ，       
            if content.get("type") == "input_text":
                chat_content = [{"type": "text", "text": str(content.get("text", ""))}]
            elif content.get("type") == "output_text":
                chat_content = [{"type": "text", "text": str(content.get("text", ""))}]
            elif content.get("type") == "input_image":
                #       
                image_url = content.get("image_url", "")
                chat_content = [{"type": "image_url", "image_url": {"url": image_url}}]
            else:
                #          ，     
                chat_content = [{"type": "text", "text": str(content)}]
        else:
            #        ，       
            chat_content = [{"type": "text", "text": str(content)}]

        chat_history.append({
            "role": role,
            "content": chat_content
        })

    return chat_history


def encode_image(image_input: Union[str, Image.Image]) -> str:
    """
          base64   

    Args:
        image_input:            PIL.Image  

    Returns:
        base64      

    Raises:
        ValueError:          
    """
    if isinstance(image_input, str):
        #        
        with open(image_input, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    elif isinstance(image_input, Image.Image):
        #    PIL.Image  
        buffer = io.BytesIO()
        image_input.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        raise ValueError("image_input must be a file path string or PIL.Image object")


def tools_to_text_description(tools: List[Dict[str, Any]]) -> str:
    """
                

    Args:
        tools:       

    Returns:
                  
    """
    if not tools:
        return ""

    descriptions = []
    for tool in tools:
        if tool.get("type") == "function":
            name = tool.get("name", "")
            description = tool.get("description", "")
            descriptions.append(f"- {name}: {description}")

    if descriptions:
        return "AVAILABLE ACTION." + "\n".join(descriptions) + "\n"
    return ""


def create_openai_response(
    client: OpenAI,
    model: str,
    conversation_history: List[Dict[str, Any]],
    api_type: Optional[ApiType] = None,
    retry_times: int = 3,
    retry_delay: float = 2.0
) -> Any:
    """
      OpenAI API   -    API    ，             
      ：         prompt ，    tools  

    Args:
        client: OpenAI     
        model:     
        conversation_history:     （       prompt）
        api_type:   API    ，   None         
        retry_times:          ，  3 （      = retry_times + 1）
        retry_delay:           ，  2.0 

    Returns:
        OpenAI API    

    Raises:
        ValueError:   API     
        Exception:                      
    """

    generation_config = {
        "response_modalities": ["TEXT"],
        "temperature": 1.0,           #     temperature   1
        "maxOutputTokens": 4096,      #       Token    4096
        "topP": 0.9                   #     top_p   0.9
    }

    #   API    
    if api_type is None:
        api_type = get_api_type_for_model(model)

    last_exc: Optional[Exception] = None
    total_attempts = retry_times + 1

    for attempt in range(total_attempts):
        try:
            if api_type == "responses":
                return client.responses.create(
                    model=model,
                    input=conversation_history,
                    temperature=1.0,
                    max_output_tokens=4096,
                    top_p=0.9     
                )
            elif api_type == "responses_jd":
                extra_body = convert_to_responses_jd_format(conversation_history)
                extra_body['generation_config'] = generation_config
                return client.responses.create(
                    model=model,
                    extra_body=extra_body
                )
            elif model.strip().lower() == "gpt-5":
                chat_history = convert_to_chat_format(conversation_history)
                return client.chat.completions.create(
                    model=model,
                    messages=chat_history,
                    temperature=1.0,
                    max_tokens=4096,
                    # top_p=0.9     #   OpenAI  API  top_p  
                )
            else:  # api_type == "chat"
                chat_history = convert_to_chat_format(conversation_history)
                return client.chat.completions.create(
                    model=model,
                    messages=chat_history,
                    temperature=1.0,
                    max_tokens=4096,
                    top_p=0.9     #   OpenAI  API  top_p  
                )
        except Exception as exc:
            last_exc = exc
            remaining = total_attempts - attempt - 1
            if remaining > 0:
                msg = (
                    f"API call failed (attempt {attempt + 1}/{total_attempts}): {exc}. "
                    f"Retrying in {retry_delay}s... ({remaining} attempt(s) left)"
                )
                _logger.warning(msg)
                print(f"[WARNING] {msg}", flush=True)
                time.sleep(retry_delay)
            else:
                msg = f"API call failed after {total_attempts} attempt(s): {exc}"
                _logger.error(msg)
                print(f"[ERROR] {msg}", flush=True)

    raise last_exc


def truncate_history(
    history: List[Dict[str, Any]],
    window_size: Optional[int]
) -> List[Dict[str, Any]]:
    """
               ，     N     user/assistant   

    Args:
        history:     user/assistant        
        window_size:            （   = 1 user + 1 assistant） 
                     None   0         29；   29        29 

    Returns:
                  
    """
    if not window_size:
        window_size = MODEL_HISTORY_TURNS
    else:
        window_size = min(MODEL_HISTORY_TURNS, max(0, int(window_size)))
    if window_size <= 0:
        return []

    #        ，        user   assistant   
    complete_message_count = len(history) - (len(history) % 2)
    history = history[:complete_message_count]

    if len(history) <= window_size * 2:
        return history

    return history[-(window_size * 2):]


def build_conversation_history(
    screenshot: str,
    prompt_text: str,
    step_number: int,
    is_first_call: bool = False,
    previous_history: Optional[List[Dict[str, Any]]] = None,
    history_window: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
          

    Args:
        screenshot: base64        
        prompt_text:         （     eval               ）
        step_number:     （  1   ）
        is_first_call:         
        previous_history:        
        history_window:         （      ） None   0         29；
                                29     + 1       

    Returns:
                 
    """
    conversation_history = []
    dialogue_history: List[Dict[str, Any]] = []

    if previous_history:
        conversation_history.append(previous_history[0])
        dialogue_history = previous_history[1:]
    else:
        conversation_history.append({
            "role": "system",
            "content": [
                {"type": "input_text", "text": prompt_text}
            ]
        })

    if dialogue_history:
        truncated = truncate_history(dialogue_history, history_window)
        conversation_history.extend(truncated)

    #         
    if is_first_call:
        user_content = [
            {"type": "input_text", "text": f"Current step: {step_number}."},
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{screenshot}"
            }
        ]
    else:
        user_content = [
            {
                "type": "input_text",
                "text": f"Current step: {step_number}. Continue based on previous conversation history."
            },
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{screenshot}"
            }
        ]

    conversation_history.append({
        "role": "user",
        "content": user_content
    })

    return conversation_history


def add_response_to_history(
    conversation_history: List[Dict[str, Any]],
    response_text: str
) -> List[Dict[str, Any]]:
    """
     AI         

    Args:
        conversation_history:        
        response_text: AI    

    Returns:
                
    """
    conversation_history.append({
        "role": "assistant",
        "content": [
            {"type": "output_text", "text": response_text}
        ]
    })

    return conversation_history




def extract_function_calls_from_responses_jd_text(text: str) -> List[Dict[str, Any]]:
    """
     responses_jd                

    Args:
        text:     

    Returns:
                
    """
    function_calls = []

    #   action_parser  JSON    
    from utils.action_parser import parse_json_from_text

    #         JSON
    json_data = parse_json_from_text(text)
    if json_data and "action" in json_data:
        #        action JSON，        
        action_name = json_data.get("action")
        function_calls.append({
            "function_name": action_name,
            "call_id": "N/A",
            "arguments": json_data,
            "api_type": "responses_jd"
        })

    return function_calls


def extract_function_calls(response: Any, api_type: Optional[ApiType] = None) -> List[Dict[str, Any]]:
    """
     OpenAI           ，    API    

    Args:
        response: OpenAI API    
        api_type: API    ，   None     

    Returns:
                
    """
    function_calls = []

    #     API  
    if api_type is None:
        if hasattr(response, 'candidates'):
            api_type = "responses_jd"
        elif hasattr(response, 'output'):
            api_type = "responses"
        elif hasattr(response, 'choices'):
            api_type = "chat"
        else:
            #     API  ，     
            return function_calls

    if api_type == "responses_jd":
        # responses_jd       （   ）
        #           
        response_text = get_response_text(response, api_type)
        if response_text:
            return extract_function_calls_from_responses_jd_text(response_text)
    elif api_type == "responses":
        # responses.create       
        if hasattr(response, 'output'):
            for item in response.output:
                if hasattr(item, 'type') and item.type == "function_call":
                    function_calls.append({
                        "function_name": item.name,
                        "call_id": getattr(item, 'call_id', 'N/A'),
                        "arguments": getattr(item, 'arguments', {}),
                        "api_type": "responses"
                    })
    elif api_type == "chat":
        # chat.completions.create       
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            # if hasattr(choice, 'message') and hasattr(choice.message, 'tool_calls'):
            if hasattr(choice, 'message') and choice.message.tool_calls:
                for tool_call in choice.message.tool_calls:
                    if hasattr(tool_call, 'function'):
                        function_calls.append({
                            "function_name": tool_call.function.name,
                            "call_id": getattr(tool_call, 'id', 'N/A'),
                            "arguments": getattr(tool_call.function, 'arguments', {}),
                            "api_type": "chat"
                        })

    return function_calls


def get_response_text(response: Any, api_type: Optional[ApiType] = None) -> str:
    """
      OpenAI       ，    API    

    Args:
        response: OpenAI API    
        api_type: API    ，   None     

    Returns:
            
    """
    #     API  
    if api_type is None:
        if hasattr(response, 'candidates'):
            api_type = "responses_jd"
        elif hasattr(response, 'output_text'):
            api_type = "responses"
        elif hasattr(response, 'choices'):
            api_type = "chat"
        else:
            return ""

    if api_type == "responses_jd":
        # responses_jd       （   ）
        #     response.text  （      None）
        if hasattr(response, 'text') and response.text is not None:
            return str(response.text)

        #     candidates  parts
        if hasattr(response, 'candidates') and response.candidates:
            #              
            candidate = response.candidates[0]

            #             candidate
            content = None
            if isinstance(candidate, dict):
                content = candidate.get('content')
            elif hasattr(candidate, 'content'):
                content = candidate.content

            if content:
                #             content
                parts = None
                if isinstance(content, dict):
                    parts = content.get('parts')
                elif hasattr(content, 'parts'):
                    parts = content.parts

                if parts:
                    text_parts = []
                    for part in parts:
                        text = None
                        if isinstance(part, dict):
                            text = part.get('text')
                        elif hasattr(part, 'text'):
                            text = part.text

                        if text:
                            text_parts.append(text)

                    if text_parts:
                        return "".join(text_parts)

                #   content    ，    
                if isinstance(content, str):
                    return content
                elif isinstance(content, dict) and 'text' in content:
                    return str(content['text'])

        #            
        if hasattr(response, 'output') and response.output:
            #   output    
            for item in response.output:
                if hasattr(item, 'text'):
                    return str(item.text)
                elif isinstance(item, dict) and 'text' in item:
                    return str(item['text'])

        return ""
    elif api_type == "responses":
        # responses.create       
        return response.output_text if hasattr(response, 'output_text') else ""
    else:  # api_type == "chat"
        # chat.completions.create       
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                return choice.message.content or ""
        return ""


#     
def add_model_mapping(model_name: str, api_type: ApiType):
    """
         API       

    Args:
        model_name:     
        api_type: API    
    """
    MODEL_API_MAPPING[model_name] = api_type


def get_all_mapped_models() -> Dict[str, ApiType]:
    """
              

    Returns:
             API         
    """
    return MODEL_API_MAPPING.copy()


#      ，           
def create_response(*args, **kwargs):
    """     ，  create_openai_response"""
    return create_openai_response(*args, **kwargs)
