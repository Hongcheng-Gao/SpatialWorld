#!/usr/bin/env python3
"""
      
   OpenAI         ，    API    （responses chat）
     ，        
"""

import json
import re
from typing import Dict, Any, Optional, Tuple


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
    from utils.openai_utils import get_response_text, extract_function_calls

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


def parse_action_from_json_rubic(
    json_data: Dict[str, Any],
    rubik_actions: Dict[str, Tuple[str, str]]
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
     JSON         

    Args:
        json_data: JSON    
        rubik_actions:         

    Returns:
        (action_name, face, direction)   
    """
    action_name = json_data.get("action")
    if not action_name:
        return None, None, None

    #             
    if action_name in rubik_actions:
        face, direction = rubik_actions[action_name]

        #         ，  JSON  face ，          
        #   JSON  face   "X" "Y" "Z"，     "RIGHT" "UP" "FRONT"
        if action_name.startswith("rotate_cube"):
            #         face direction
            return action_name, face, direction

        #        ，  JSON    face direction，  JSON   
        json_face = json_data.get("face")
        json_direction = json_data.get("direction")

        if json_face and json_direction:
            #   face  ：JSON     "F"/"B"/"L"/"R"/"U"/"D"            
            face_map = {
                "F": "FRONT",
                "B": "BACK",
                "L": "LEFT",
                "R": "RIGHT",
                "U": "UP",
                "D": "DOWN",
                "FRONT": "FRONT",  #          ，    
                "BACK": "BACK",
                "LEFT": "LEFT",
                "RIGHT": "RIGHT",
                "UP": "UP",
                "DOWN": "DOWN"
            }
            mapped_face = face_map.get(json_face.upper(), face)

            #       ：JSON  "CW"/"ACW"        "CLOCKWISE"/"ANTICLOCKWISE"
            direction_map = {
                "CW": "CLOCKWISE",
                "ACW": "ANTICLOCKWISE",
                "CLOCKWISE": "CLOCKWISE",  #          ，    
                "ANTICLOCKWISE": "ANTICLOCKWISE"
            }
            mapped_direction = direction_map.get(json_direction.upper(), direction)
            return action_name, mapped_face, mapped_direction
        else:
            return action_name, face, direction

    #           ，   JSON   
    face = json_data.get("face")
    direction = json_data.get("direction")

    #   face  
    if face:
        face_map = {
            "F": "FRONT",
            "B": "BACK",
            "L": "LEFT",
            "R": "RIGHT",
            "U": "UP",
            "D": "DOWN",
            "FRONT": "FRONT",
            "BACK": "BACK",
            "LEFT": "LEFT",
            "RIGHT": "RIGHT",
            "UP": "UP",
            "DOWN": "DOWN"
        }
        face = face_map.get(face.upper(), face)

    #       
    if direction:
        direction_map = {
            "CW": "CLOCKWISE",
            "ACW": "ANTICLOCKWISE",
            "CLOCKWISE": "CLOCKWISE",
            "ANTICLOCKWISE": "ANTICLOCKWISE"
        }
        direction = direction_map.get(direction.upper(), direction)

    return action_name, face, direction