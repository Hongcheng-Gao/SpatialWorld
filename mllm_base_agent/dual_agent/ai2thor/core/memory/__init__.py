"""
Dual Agent Memory Module
Extended memory buffer for dual-agent collaboration with shared memory
"""

from typing import List, Dict, Any


class DualAgentMemoryBuffer:
    """Dual Agent Memory Buffer

    Implements hybrid memory mechanism for dual-agent system:
    - Individual agent memory (sliding window + semantic summary)
    - Shared memory for inter-agent communication
    """

    def __init__(self, short_term_window_size: int = 3, shared_memory_size: int = 20):
        """Initialize dual agent memory buffer

        Args:
            short_term_window_size: Short-term memory window size per agent
            shared_memory_size: Maximum items in shared memory
        """
        self.short_term_window_size = short_term_window_size
        self.shared_memory_size = shared_memory_size

        # Individual agent memories
        self.agent_1_short_term: List[Dict[str, Any]] = []
        self.agent_1_long_term: str = ""

        self.agent_2_short_term: List[Dict[str, Any]] = []
        self.agent_2_long_term: str = ""

        # Shared memory components
        self.discovered_objects: Dict[str, List[str]] = {}
        self.explored_areas: List[str] = []
        self.key_findings: List[str] = []
        self.target_locations: Dict[str, str] = {}

        # Communication history
        self.communication_history: List[Dict[str, Any]] = []
        self.message_queue: List[Dict[str, Any]] = []

    def add_step(self, agent_id: str, step_data: Dict[str, Any]):
        """Add a step to agent's short-term memory

        Args:
            agent_id: "agent_1" or "agent_2"
            step_data: Dictionary containing step info
        """
        if agent_id == "agent_1":
            self.agent_1_short_term.append(step_data)
            if len(self.agent_1_short_term) > self.short_term_window_size:
                self.agent_1_short_term.pop(0)
        else:
            self.agent_2_short_term.append(step_data)
            if len(self.agent_2_short_term) > self.short_term_window_size:
                self.agent_2_short_term.pop(0)

    def update_long_term_summary(self, agent_id: str, summary: str):
        """Update agent's long-term semantic summary

        Args:
            agent_id: "agent_1" or "agent_2"
            summary: New long-term summary
        """
        if agent_id == "agent_1":
            self.agent_1_long_term = summary
        else:
            self.agent_2_long_term = summary

    def add_discovered_object(self, object_type: str, location: str):
        """Add a discovered object to shared memory

        Args:
            object_type: Type of object discovered
            location: Location description or coordinates
        """
        if object_type not in self.discovered_objects:
            self.discovered_objects[object_type] = []
        if location not in self.discovered_objects[object_type]:
            self.discovered_objects[object_type].append(location)

    def add_explored_area(self, area: str):
        """Add an explored area to shared memory

        Args:
            area: Description of explored area
        """
        if area not in self.explored_areas:
            self.explored_areas.append(area)
            # Maintain size limit
            if len(self.explored_areas) > self.shared_memory_size:
                self.explored_areas.pop(0)

    def add_key_finding(self, finding: str):
        """Add a key finding to shared memory

        Args:
            finding: Description of key finding
        """
        self.key_findings.append(finding)
        if len(self.key_findings) > self.shared_memory_size:
            self.key_findings.pop(0)

    def set_target_location(self, target: str, location: str):
        """Set a target's location in shared memory

        Args:
            target: Target name/type
            location: Location of target
        """
        self.target_locations[target] = location

    def add_communication(self, sender: str, receiver: str, message: str, step: int):
        """Add a communication event

        Args:
            sender: Sender agent ID
            receiver: Receiver agent ID
            message: Communication message
            step: Global step number
        """
        comm_entry = {
            "sender": sender,
            "receiver": receiver,
            "message": message,
            "step": step,
        }
        self.communication_history.append(comm_entry)
        self.message_queue.append(comm_entry)

    def get_pending_messages(self, receiver: str) -> List[Dict[str, Any]]:
        """Get pending messages for an agent

        Args:
            receiver: Receiver agent ID

        Returns:
            List of pending messages
        """
        messages = [m for m in self.message_queue if m["receiver"] == receiver]
        # Remove delivered messages
        self.message_queue = [
            m for m in self.message_queue if m["receiver"] != receiver
        ]
        return messages

    def get_short_term_history(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get agent's short-term history

        Args:
            agent_id: "agent_1" or "agent_2"

        Returns:
            Short-term history list
        """
        if agent_id == "agent_1":
            return self.agent_1_short_term.copy()
        else:
            return self.agent_2_short_term.copy()

    def get_long_term_summary(self, agent_id: str) -> str:
        """Get agent's long-term summary

        Args:
            agent_id: "agent_1" or "agent_2"

        Returns:
            Long-term summary string
        """
        if agent_id == "agent_1":
            return self.agent_1_long_term
        else:
            return self.agent_2_long_term

    def get_shared_context(self) -> Dict[str, Any]:
        """Get all shared memory as context

        Returns:
            Dictionary with all shared memory components
        """
        return {
            "discovered_objects": self.discovered_objects.copy(),
            "explored_areas": self.explored_areas.copy(),
            "key_findings": self.key_findings.copy(),
            "target_locations": self.target_locations.copy(),
            "recent_communications": self.communication_history[-5:],
        }

    def build_shared_context_string(self) -> str:
        """Build shared context as formatted string

        Returns:
            Formatted shared context string
        """
        parts = []

        if self.discovered_objects:
            parts.append("**Discovered Objects:**")
            for obj_type, locations in self.discovered_objects.items():
                parts.append(f"  - {obj_type}: {', '.join(locations)}")

        if self.explored_areas:
            parts.append("\n**Explored Areas:**")
            for area in self.explored_areas[-5:]:
                parts.append(f"  - {area}")

        if self.key_findings:
            parts.append("\n**Key Findings:**")
            for finding in self.key_findings[-5:]:
                parts.append(f"  - {finding}")

        if self.target_locations:
            parts.append("\n**Known Target Locations:**")
            for target, location in self.target_locations.items():
                parts.append(f"  - {target}: {location}")

        if self.communication_history:
            parts.append("\n**Recent Communications:**")
            for comm in self.communication_history[-3:]:
                parts.append(
                    f"  [{comm['sender']}→{comm['receiver']}]: {comm['message'][:100]}..."
                )

        return "\n".join(parts) if parts else "No shared context available."

    def clear(self):
        """Clear all memory"""
        self.agent_1_short_term = []
        self.agent_1_long_term = ""
        self.agent_2_short_term = []
        self.agent_2_long_term = ""
        self.discovered_objects = {}
        self.explored_areas = []
        self.key_findings = []
        self.target_locations = {}
        self.communication_history = []
        self.message_queue = []


class IndividualAgentMemory:
    """Individual Agent Memory

    Simplified memory class for single agent within dual-agent system
    """

    def __init__(self, short_term_window_size: int = 3):
        """Initialize individual agent memory

        Args:
            short_term_window_size: Short-term memory window size
        """
        self.short_term_window_size = short_term_window_size
        self.short_term_history: List[Dict[str, Any]] = []
        self.long_term_summary: str = ""

    def add_step(self, step_data: Dict[str, Any]):
        """Add a step to short-term memory

        Args:
            step_data: Step data dictionary
        """
        self.short_term_history.append(step_data)
        if len(self.short_term_history) > self.short_term_window_size:
            self.short_term_history.pop(0)

    def update_summary(self, summary: str):
        """Update long-term summary

        Args:
            summary: New summary
        """
        self.long_term_summary = summary

    def get_history(self) -> List[Dict[str, Any]]:
        """Get short-term history"""
        return self.short_term_history.copy()

    def get_summary(self) -> str:
        """Get long-term summary"""
        return self.long_term_summary

    def clear(self):
        """Clear memory"""
        self.short_term_history = []
        self.long_term_summary = ""
