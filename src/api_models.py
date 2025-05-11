from pydantic import BaseModel
from typing import Literal
from datetime import datetime

class Stream(BaseModel):
    id: int
    name: str
    source: str
    status: Literal["active","inactive","paused"]

class Alert(BaseModel):
    id: int
    stream_id:int
    class_name: str
    alert_type: str
    timestamp: datetime

