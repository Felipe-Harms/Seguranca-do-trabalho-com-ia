from pydantic import BaseModel
from typing import Literal, Optional
from datetime import datetime

class Stream(BaseModel):
    id: Optional[int] = None
    name: str
    source: str
    status: Literal["active","inactive","paused"] = "paused"

class Alert(BaseModel):
    id: int = None
    stream_id:int
    class_name: str
    alert_type: str
    timestamp: datetime

