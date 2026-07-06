import json
from typing import List, Dict

class UserMemory:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.history: List[Dict] = []
        self.max_history = 6

    def add_message(self, role: str, content: str):
        if len(content) > 500:
            content = content[:500] + "..."
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_history(self) -> List[Dict]:
        return self.history

    def clear(self):
        self.history = []


class MemoryManager:
    def __init__(self):
        self.users: Dict[str, UserMemory] = {}

    def get_user_memory(self, user_id: str) -> UserMemory:
        if user_id not in self.users:
            self.users[user_id] = UserMemory(user_id)
        return self.users[user_id]

    def add_message(self, user_id: str, role: str, content: str):
        self.get_user_memory(user_id).add_message(role, content)

    def get_history(self, user_id: str) -> List[Dict]:
        return self.get_user_memory(user_id).get_history()

    def clear_user(self, user_id: str):
        if user_id in self.users:
            self.users[user_id].clear()

    def save_to_file(self, filepath: str = "memory.json"):
        data = {uid: mem.history for uid, mem in self.users.items()}
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def load_from_file(self, filepath: str = "memory.json"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for uid, history in data.items():
                mem = self.get_user_memory(uid)
                mem.history = history[-mem.max_history:]
        except FileNotFoundError:
            pass