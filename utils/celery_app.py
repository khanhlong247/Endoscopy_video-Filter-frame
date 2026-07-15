from celery import Celery
from config import CELERY_BROKER_URL

app = Celery(
    'endoscopy_tasks',
    broker=CELERY_BROKER_URL,
    include=['service.embedding_service']
)

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Ho_Chi_Minh',
    enable_utc=True,
)