from typing import Union

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, HttpUrl

app = FastAPI()


class controller(BaseModel):
    temperature: float 
    pulse: int 
    sys: int 
    dia: int
    rr: int 
    sats: int
    clientid: int



class controllerEvent(BaseModel):
    prediction: int
    probability: int


class controllerEventReceived(BaseModel):
    ok: bool


controllers_callback_router = APIRouter()


@controllers_callback_router.post(
    "{$callback_url}/controllers/{$request.body.id}", response_model=controllerEventReceived
)
def controller_notification(body: controllerEvent):
    pass


@app.post("/controllers/", callbacks=controllers_callback_router.routes)
def create_controller(controller: controller, callback_url: Union[HttpUrl, None] = None):
    """
    This is just another api
    """
    # Send the controller, collect the money, send the notification (the callback)
    return {"msg": "controller received"}
