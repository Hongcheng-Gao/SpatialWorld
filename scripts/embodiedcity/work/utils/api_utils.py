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
from typing import List, Dict, Any, Optional, Union, Literal, Tuple
from PIL import Image
from openai import OpenAI
import json
import re


#    API       
#               
MODEL_API_MAPPING = {
    # responses.create      
    "Gemini 3-Pro-Preview": "responses_jd",
    "Gemini-3.1-Pro-Preview": "responses_jd",
    "Gemini-3-Flash-Preview": "responses_jd",
    "gpt-5": "chat",
    "gpt-4": "responses",

    # chat.completions.create      
    "Gemini-2.5-flash": "chat",
    "Gemini-2.5-pro":"chat"

}

#       
DEFAULT_API_TYPE = "chat"
MODEL_INPUT_TURNS = 30

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
    temperature: float = 1.0,
    top_p: float = 0.9,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Any:
    """
      OpenAI API   -    API    ，        
      ：         prompt ，    tools  

    Args:
        client: OpenAI     
        model:     
        conversation_history:     （       prompt）
        api_type:   API    ，   None         

    Returns:
        OpenAI API    

    Raises:
        ValueError:   API     
    """
    generation_config = {
                "response_modalities": ["TEXT"],
                "temperature": temperature,           #     temperature   1
                "maxOutputTokens": 4096,      #       Token    4096
                "topP": top_p                   #     top_p   0.9
            }

    #   API    
    if api_type is None:
        api_type = get_api_type_for_model(model)

    if api_type == "responses":
        #   responses.create  
        return client.responses.create(
            model=model,
            input=conversation_history,
            temperature=temperature,
            max_output_tokens=4096,
            top_p=top_p     
        )
    elif api_type == "responses_jd":
        #   responses.create  ，        
        jd_extra_body = convert_to_responses_jd_format(conversation_history)
        jd_extra_body['generation_config'] = generation_config
        if extra_body:
            jd_extra_body.update(extra_body)
        return client.responses.create(
            model=model,
            extra_body=jd_extra_body,
        )
    else:  # api_type == "chat"
        #   chat.completions.create  
        #           chat  
        chat_history = convert_to_chat_format(conversation_history)

        kwargs = {
            "model": model,
            "messages": chat_history,
            "temperature": temperature,
            "max_tokens": 4096,
            "top_p": top_p,     #   OpenAI  API  top_p  
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        return client.chat.completions.create(**kwargs)


def build_conversation_history(
    screenshot: str,
    prompt_text: str,
    is_first_call: bool = False,
    previous_history: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
          

    Args:
        screenshot: base64        
        prompt_text:         （     eval               ）
        is_first_call:         
        previous_history:        

    Returns:
                 
    """
    conversation_history = []

    if previous_history:
        #          
        conversation_history.extend(previous_history)

    #         
    if is_first_call:
        user_content = [
            {"type": "input_text", "text": prompt_text},
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{screenshot}"
            }
        ]
    else:
        user_content = [
            {"type": "input_text", "text": "Continue based on previous conversation history."},
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

def parse_json_from_text(text: Any) -> Optional[Dict[str, Any]]:
    """
             JSON  

    Args:
        text:   JSON   ，     JSON     

    Returns:
            JSON  ，         None
    """
    #          ，    （               ）
    if isinstance(text, dict):
        #             （  action ）
        if "action" in text:
            return text
        #            JSON   
        else:
            #       ：text, content, message, response, data
            common_keys = ['text', 'content', 'message', 'response', 'data']
            for key in common_keys:
                if key in text and isinstance(text[key], str):
                    nested_result = parse_json_from_text(text[key])
                    if nested_result:
                        return nested_result

            #               JSON
            for value in text.values():
                if isinstance(value, str):
                    nested_result = parse_json_from_text(value)
                    if nested_result:
                        return nested_result

            #       ，     
            if not text:
                return text

            #       None
            return None

    #         ，    
    if isinstance(text, str):
        #           
        try:
            result = json.loads(text)
            #        ，            
            if isinstance(result, dict):
                if "action" in result:
                    return result
                #         ，        JSON   
                else:
                    #          JSON   
                    for value in result.values():
                        if isinstance(value, str):
                            nested_result = parse_json_from_text(value)
                            if nested_result:
                                return nested_result
                    return None
            #         ，      JSON  ，    
            elif isinstance(result, str):
                return parse_json_from_text(result)
            #      ，  None（       ）
            elif isinstance(result, list):
                return None
        except json.JSONDecodeError:
            pass

        #     JSON  （{...}）
        #              JSON  
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                result = json.loads(match)
                if isinstance(result, dict):
                    if "action" in result:
                        return result
                    #         ，        JSON   
                    else:
                        for value in result.values():
                            if isinstance(value, str):
                                nested_result = parse_json_from_text(value)
                                if nested_result:
                                    return nested_result
                elif isinstance(result, str):
                    nested_result = parse_json_from_text(result)
                    if nested_result:
                        return nested_result
                elif isinstance(result, list):
                    #      ，  
                    continue
            except json.JSONDecodeError:
                continue

        #     JSON  （[...]）
        array_pattern = r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'
        matches = re.findall(array_pattern, text, re.DOTALL)

        for match in matches:
            try:
                result = json.loads(match)
                #      ，  None（       ）
                if isinstance(result, list):
                    return None
                elif isinstance(result, dict):
                    if "action" in result:
                        return result
                elif isinstance(result, str):
                    nested_result = parse_json_from_text(result)
                    if nested_result:
                        return nested_result
            except json.JSONDecodeError:
                continue

    return None


def apply_history_window(history: List[Dict[str, Any]], window) -> List[Dict[str, Any]]:
    """
    Return a windowed view of conversation history.

    Keeps the first system message, plus enough recent user/assistant turns and
    the current user message so the total number of user-image messages sent to
    the model is at most N, including the current step.

    Args:
        history: full conversation history list
        window:  None/0 → use the unified default of 30 user-image turns total
                 int N  → keep at most N user-image turns total, capped at 30

    Returns:
        Windowed (possibly truncated) history list.
    """
    if len(history) <= 1:
        return history

    if not window:
        window = MODEL_INPUT_TURNS
    else:
        window = min(MODEL_INPUT_TURNS, max(1, int(window)))
    first_message = history[:1]
    remaining = history[1:]

    current_user = []
    completed_turns = remaining
    if remaining and remaining[-1].get("role") == "user":
        current_user = [remaining[-1]]
        completed_turns = remaining[:-1]

    if window <= 0:
        return first_message + current_user

    history_turns_to_keep = max(window - len(current_user), 0)
    return first_message + completed_turns[-2 * history_turns_to_keep:] + current_user


def extract_action_info_from_response(
    response: Any,
    api_type: Optional[str] = None
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
     OpenAI         ，    API    

    Args:
        response: OpenAI API    
        api_type: API    （"responses" "chat"），   None     

    Returns:
        (action_name, action_data)   ：
        - action_name:     ，        None
        - action_data:         ，      ：
            * function_name:    （         ）
            * json_data: JSON  （   JSON    ）
            * response_text:     
            * parsed_from_json:    JSON  
            *    JSON      （ face, direction, type, key ）
    """


    #       
    response_text = get_response_text(response, api_type)

    #             
    function_calls = extract_function_calls(response, api_type)
    if function_calls:
        #         
        function_call = function_calls[0]
        function_name = function_call.get("function_name")
        if function_name:
            #          
            action_data = {
                "function_name": function_name,
                "response_text": response_text,
                "arguments": function_call.get("arguments", {}),
                "call_id": function_call.get("call_id", "N/A"),
                "api_type": function_call.get("api_type", "unknown"),
                "parsed_from_function_call": True
            }
            return function_name, action_data

    #         ，        JSON
    if response_text:
        json_data = parse_json_from_text(response_text)
        if json_data:
            action_name = json_data.get("action")
            if action_name:
                #  JSON        -     ，    JSON  
                action_data = {
                    "json_data": json_data,
                    "response_text": response_text,
                    "parsed_from_json": True
                }
                #  JSON         action_data 
                for key, value in json_data.items():
                    if key not in action_data:  #         
                        action_data[key] = value
                return action_name, action_data

    #       ，  None
    return None, None

