import os
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi_socketio import SocketManager
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi import Depends
from fastapi import Query
from fastapi import Form
from fastapi import Body
from fastapi import APIRouter
from fastapi import HTTPException
from flask import Flask, render_template
from flask_socketio import SocketIO, emit, send
from flask_cors import CORS
import json
from fastapi import APIRouter, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Create a User class if it doesn't exist
class User:
    def __init__(self, username, meetingID, **kwargs):
        self.username = username
        self.meetingID = meetingID
        for key, value in kwargs.items():
            setattr(self, key, value)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
CORS(app)
socketio.init_app(app, cors_allowed_origins="*")
users = []
router = APIRouter()

# Remove the socket_manager initialization here
# socket_manager = SocketManager(app=None)  # This was causing the error

# Cấu hình templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# HTTP Routes
@router.get("/index", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.get("/meeting/{uid}", response_class=HTMLResponse)
async def get_meeting(request: Request, uid: str):
    return templates.TemplateResponse("meeting.html", {
        "request": request,
        "uid": uid
    })

# Create a function to set up socket handlers
def setup_socket_handlers(socket_manager):
    @socket_manager.on("connect")
    async def handle_connect(sid, environ):
        print(f"Client connected: {sid}")

    @socket_manager.on("disconnect")
    async def handle_disconnect(sid):
        print(f"Client disconnected: {sid}")

    @socket_manager.on('newUser')
    async def handle_new_user(sid, msg):
        try:
            data = json.loads(msg)
            new_user = User(**data)
            users.append(new_user)
            await socket_manager.emit('newUser', msg, skip_sid=sid)
            print(f"New user added: {new_user.username}")
        except Exception as e:
            print(f"Error: {str(e)}")

    @socket_manager.on('checkUser')
    async def handle_check_user(sid, msg):
        try:
            data = json.loads(msg)
            exists = any(
                user.username == data["username"] and 
                user.meetingID == data["meetingID"]
                for user in users
            )
            await socket_manager.emit('userExists' if exists else 'userOK', to=sid)
        except Exception as e:
            print(f"Check user error: {str(e)}")

    @socket_manager.on('userDisconnected')
    async def handle_user_disconnect(sid, msg):
        try:
            data = json.loads(msg)
            global users
            users = [
                user for user in users
                if not (user.username == data["username"] and user.meetingID == data["meetingID"])
            ]
            await socket_manager.emit('userDisconnected', msg, skip_sid=sid)
            print(f"User {data['username']} disconnected")
        except Exception as e:
            print(f"Disconnect error: {str(e)}")

    @socket_manager.on('message')
    async def handle_message(sid, msg):
        try:
            print(f"Received message: {msg}")
            await socket_manager.emit('message', msg, skip_sid=sid)
        except Exception as e:
            print(f"Message error: {str(e)}")

if __name__ == '__main__':
    socketio.run(app, host='localhost', port=5000)
