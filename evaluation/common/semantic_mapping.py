"""
Semantic object mapping for evaluators
Maps original object types to their transformed variants for flexible success condition checking

This mapping is used in MultiConditionEvaluator to support checking success conditions
with transformed objects. For example, "Apple" task can succeed with either "Apple" or "AppleSliced".
"""

# Semantic object mapping: original -> possible variants after interactions
SEMANTIC_OBJECT_MAPPING = {
    # Sliceable objects
    "Apple": ["Apple", "AppleSliced"],
    "Bread": ["Bread", "BreadSliced", "BreadToasted"],
    "Tomato": ["Tomato", "TomatoSliced"],
    "Lettuce": ["Lettuce", "LettuceSliced"],
    "Potato": ["Potato", "PotatoSliced", "PotatoCooked"],
    "Egg": ["Egg", "EggSliced", "EggBroken", "EggCooked"],
    
    # Breakable objects
    "Bottle": ["Bottle", "BottleBroken"],
    "Cup": ["Cup", "CupBroken"],
    "Mug": ["Mug", "MugBroken"],
    "Plate": ["Plate", "PlateBroken"],
    "Vase": ["Vase", "VaseBroken"],
    "WineBottle": ["WineBottle", "WineBottleBroken"],
    "Window": ["Window", "WindowBroken"],
    "Statue": ["Statue", "StatueBroken"],
    
    # Consumable objects
    "PaperTowelRoll": ["PaperTowelRoll", "PaperTowel"],
}


def get_semantic_variants(object_type: str) -> list:
    """Get all possible variants of an object type
    
    Args:
        object_type: Original object type (e.g., "Apple")
        
    Returns:
        List of possible object types including original and variants
        If not in mapping, returns [object_type] for backward compatibility
    """
    return SEMANTIC_OBJECT_MAPPING.get(object_type, [object_type])
