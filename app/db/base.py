from app.db.database import Base
from app.models.teacher import Teacher  # noqa
from app.models.student import Student  # noqa
from app.models.task import Task, TaskAssignment  # noqa
from app.models.submission import Submission  # noqa
from app.models.ai_mentor import (  # noqa
    AIMentorChatMessage,
    AIMentorChatSession,
    AIMentorDiagnosticAnswer,
    AIMentorDiagnosticQuestion,
    AIMentorDiagnosticSession,
    AIMentorPlan,
    AIMentorPlanItem,
    AIMentorPlanWeek,
)

from app.models.rag import RAGChunk, RAGDocument, RAGProcessingJob  # noqa
