import asyncio
import base64
import os
import re
import time
import uuid
from collections import deque
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Body, File, HTTPException, Path, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from utils import detect_smoking, extract_hls_url_from_page

router = APIRouter(prefix="/stream", tags=["Streaming"])
websocket_router = APIRouter()


class FrameData(BaseModel):
    metric: float = Field(..., ge=0.0, le=1.0)
    cord: List[int] = Field(..., min_items=4, max_items=4)


class BroadcastRequest(BaseModel):
    frames: List[FrameData]


class StreamResponse(BaseModel):
    stream_token: str
    stream_id: str
    video_uuid: str
    video_id: str
    websocket_url: str
    status: str


class BroadcastResponse(BaseModel):
    message: str
    stream_id: str
    recipient_count: int
    data: dict


class CloseStreamResponse(BaseModel):
    message: str
    stream_id: str
    status: str
    closed_at: float


class StreamListResponse(BaseModel):
    active_streams: List[str]
    count: int


class SmokingDetectionResponse(BaseModel):
    verdict: str
    timestamp: float


class StreamUrlRequest(BaseModel):
    url: str
    detection_interval: Optional[int] = Field(5)


class StreamUrlResponse(BaseModel):
    stream_id: str
    url: str
    status: str
    websocket_url: str
    video_url: str
    message: str


stream_sessions: Dict[str, Dict] = {}
last_frames: Dict[str, bytes] = {}
detection_queues: Dict[str, Dict] = {}

ENABLE_VIDEO_DISPLAY = False


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, token: str):
        if token not in self.active_connections:
            self.active_connections[token] = []
        self.active_connections[token].append(websocket)

    def disconnect(self, websocket: WebSocket, token: str):
        if token in self.active_connections:
            if websocket in self.active_connections[token]:
                self.active_connections[token].remove(websocket)
            if not self.active_connections[token]:
                del self.active_connections[token]

    async def broadcast_json(self, data: dict, token: str):
        if token in self.active_connections:
            for connection in self.active_connections[token]:
                await connection.send_json(data)


manager = ConnectionManager()


def get_active_streams() -> List[str]:
    return list(stream_sessions.keys())


@router.get("/list", response_model=StreamListResponse)
async def list_active_streams():
    active_stream_ids = get_active_streams()
    return {
        "active_streams": active_stream_ids,
        "count": len(active_stream_ids)
    }


@router.get("/status/{stream_id}")
async def get_stream_status(stream_id: str = Path(...)):
    if stream_id not in stream_sessions:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")
    session = stream_sessions[stream_id]
    return {
        "stream_id": stream_id,
        "status": session.get("status", "unknown"),
        "live": session.get("live", False),
        "closing": session.get("closing", False),
        "error": session.get("error"),
        "url": session.get("url"),
        "created_at": session.get("created_at"),
        "type": session.get("type", "websocket")
    }


@router.post("/detect-smoking", response_model=SmokingDetectionResponse)
async def detect_smoking_photo(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format.")
        _, buffer = cv2.imencode('.png', img)
        b64_image = base64.b64encode(buffer).decode('utf-8')
        verdict_raw = detect_smoking(b64_image)
        if verdict_raw is None:
            raise HTTPException(status_code=500, detail="Failed to get response from AI model")
        timestamp = time.time()
        verdict_lower = verdict_raw.strip().lower()
        if "yes" in verdict_lower:
            verdict = "Yes"
        elif "no" in verdict_lower:
            verdict = "No"
        else:
            verdict = verdict_raw.strip()
        return {"verdict": verdict, "timestamp": timestamp}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


async def process_video_stream_from_url(stream_id: str, url: str, detection_interval: int = 5):
    cap = None
    video_writer = None
    video_path = None
    last_detection_time = 0
    frame_count = 0
    actual_url = url

    try:
        if url.startswith('blob:') or (url.startswith('http') and not url.endswith(('.m3u8', '.ts', '.mp4', '.avi', '.mov'))):
            page_url = url.replace('blob:', '') if url.startswith('blob:') else url
            extracted_url = await asyncio.to_thread(extract_hls_url_from_page, page_url)
            if extracted_url:
                actual_url = extracted_url

        cap = cv2.VideoCapture(actual_url)

        if not cap.isOpened():
            error_msg = f"Failed to open video stream: {actual_url}"
            stream_sessions[stream_id]["status"] = "error"
            stream_sessions[stream_id]["error"] = error_msg
            error_payload = {
                "type": "error",
                "message": "Failed to open video stream",
                "details": error_msg,
                "timestamp": time.time()
            }
            try:
                await manager.broadcast_json(error_payload, stream_id)
            except Exception:
                pass
            return

        stream_sessions[stream_id]["status"] = "streaming"

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        stream_folder = "stream"
        os.makedirs(stream_folder, exist_ok=True)
        video_path = os.path.join(stream_folder, f"{stream_id}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        stream_sessions[stream_id]["video_writer"] = video_writer
        stream_sessions[stream_id]["video_path"] = video_path

        detection_queues[stream_id] = {
            'results': {},
            'next_frame': 1,
            'lock': asyncio.Lock()
        }

        async def send_ordered_results():
            while stream_sessions.get(stream_id, {}).get("live", False):
                queue_data = detection_queues.get(stream_id)
                if not queue_data:
                    await asyncio.sleep(0.1)
                    continue
                async with queue_data['lock']:
                    next_frame_num = queue_data['next_frame']
                    if next_frame_num in queue_data['results']:
                        payload = queue_data['results'].pop(next_frame_num)
                        await manager.broadcast_json(payload, stream_id)
                        queue_data['next_frame'] += 1
                    else:
                        await asyncio.sleep(0.01)
                await asyncio.sleep(0.001)

        send_task = asyncio.create_task(send_ordered_results())

        while stream_sessions.get(stream_id, {}).get("live", False):
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = time.time()

            if video_writer is not None:
                video_writer.write(frame)

            try:
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                last_frames[stream_id] = buffer.tobytes()
            except Exception as e:
                print(f"[{stream_id}] Frame encode error: {e}")

            has_websocket_clients = stream_id in manager.active_connections and len(manager.active_connections[stream_id]) > 0

            if has_websocket_clients and (current_time - last_detection_time >= detection_interval):
                last_detection_time = current_time
                detection_frame_number = frame_count
                detection_timestamp = current_time
                frame_copy = frame.copy()

                async def run_detection(frame_num: int, timestamp: float, frame_data):
                    try:
                        def encode_frame():
                            _, buffer = cv2.imencode('.png', frame_data)
                            return base64.b64encode(buffer).decode('utf-8')
                        b64_image = await asyncio.to_thread(encode_frame)
                        verdict_raw = await asyncio.to_thread(detect_smoking, b64_image)
                        if verdict_raw is not None:
                            verdict_lower = verdict_raw.strip().lower()
                            if "yes" in verdict_lower:
                                verdict = "Yes"
                            elif "no" in verdict_lower:
                                verdict = "No"
                            else:
                                verdict = verdict_raw.strip()
                            payload = {
                                "type": "smoking_detection",
                                "timestamp": timestamp,
                                "verdict": verdict,
                                "frame_number": frame_num
                            }
                            await manager.broadcast_json(payload, stream_id)
                    except Exception as e:
                        print(f"[{stream_id}] Detection error: {e}")

                asyncio.create_task(run_detection(detection_frame_number, detection_timestamp, frame_copy))

            await asyncio.sleep(0.001)

            if stream_sessions.get(stream_id, {}).get("closing", False):
                break

    except Exception as e:
        print(f"[{stream_id}] Critical error: {e}")
        stream_sessions[stream_id]["status"] = "error"
        stream_sessions[stream_id]["error"] = str(e)

    finally:
        if cap is not None:
            cap.release()
        if video_writer is not None:
            video_writer.release()
        if stream_id in stream_sessions:
            stream_sessions[stream_id]["live"] = False
            stream_sessions[stream_id]["status"] = "stopped"


@router.post("/open-stream-url", response_model=StreamUrlResponse)
async def open_stream_from_url(request: StreamUrlRequest = Body(...)):
    try:
        stream_id = str(uuid.uuid4())
        stream_sessions[stream_id] = {
            "live": True,
            "closing": False,
            "created_at": time.time(),
            "url": request.url,
            "detection_interval": request.detection_interval,
            "status": "initializing",
            "type": "url_stream"
        }
        asyncio.create_task(process_video_stream_from_url(
            stream_id, request.url, request.detection_interval
        ))
        return {
            "stream_id": stream_id,
            "url": request.url,
            "status": "initializing",
            "websocket_url": f"/ws/stream/{stream_id}",
            "video_url": f"/stream/video/{stream_id}",
            "message": "Stream is being initialized."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to open stream: {str(e)}")


@router.post("/request", response_model=StreamResponse)
async def request_stream_token():
    stream_id = str(uuid.uuid4())
    stream_sessions[stream_id] = {
        "live": True,
        "closing": False,
        "created_at": time.time()
    }
    return {
        "stream_token": stream_id,
        "stream_id": stream_id,
        "video_uuid": stream_id,
        "video_id": stream_id,
        "websocket_url": f"/ws/stream/{stream_id}",
        "status": "active"
    }


@router.post("/broadcast/{token}", response_model=BroadcastResponse)
async def broadcast_to_stream(
    token: str = Path(...),
    data: BroadcastRequest = Body(...)
):
    if token not in stream_sessions:
        raise HTTPException(status_code=404, detail=f"Stream ID {token} not found")
    if not stream_sessions[token].get("live", False):
        raise HTTPException(status_code=400, detail=f"Stream {token} is not active")
    try:
        frames_data = [{"metric": f.metric, "cord": f.cord} for f in data.frames]
        payload = {"time": time.time(), "frames": frames_data}
        await manager.broadcast_json(payload, token)
        recipient_count = len(manager.active_connections.get(token, []))
        return {
            "message": "Data broadcasted successfully",
            "stream_id": token,
            "recipient_count": recipient_count,
            "data": payload
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to broadcast data: {str(e)}")


@router.post("/close/{token}", response_model=CloseStreamResponse)
async def close_stream(token: str = Path(...)):
    if token not in stream_sessions:
        raise HTTPException(status_code=404, detail=f"Stream ID {token} not found")
    session = stream_sessions[token]
    if session.get("closing", False):
        return {"message": "Stream is already closing", "stream_id": token, "status": "closing", "closed_at": time.time()}
    if not session.get("live", False):
        return {"message": "Stream is already closed", "stream_id": token, "status": "closed", "closed_at": time.time()}
    session["closing"] = True
    await asyncio.sleep(2)
    if "video_writer" in session and session["video_writer"] is not None:
        try:
            session["video_writer"].release()
        except Exception as e:
            print(f"VideoWriter release error: {e}")
    session["live"] = False
    session["closing"] = False
    session["closed_at"] = time.time()
    return {
        "message": "Stream closed successfully",
        "stream_id": token,
        "status": "closed",
        "closed_at": session["closed_at"]
    }


@router.post("/stream/start/{token}")
async def start_stream(token: str):
    if token not in stream_sessions:
        raise HTTPException(status_code=404, detail="Stream token not found")
    if stream_sessions[token].get("live", False):
        return {"message": "Stream already active", "stream_id": token, "video_uuid": token, "video_id": token}
    stream_sessions[token]["live"] = True
    return {"message": "Stream started successfully", "stream_id": token, "video_uuid": token, "video_id": token}


@router.get("/video/{token}")
async def get_video_stream(token: str = Path(...)):
    if token not in stream_sessions:
        raise HTTPException(status_code=404, detail=f"Stream ID {token} not found")

    async def generate_frames():
        while True:
            if token not in stream_sessions:
                break
            if token in last_frames:
                frame_bytes = last_frames[token]
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                await asyncio.sleep(0.1)
                continue
            await asyncio.sleep(0.1)

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@websocket_router.websocket("/ws/stream/{token}")
async def websocket_endpoint(websocket: WebSocket, token: str):
    client_host = websocket.client.host if websocket.client else "unknown"

    try:
        await websocket.accept()
    except Exception as e:
        print(f"WebSocket accept error: {e}")
        return

    if token not in stream_sessions or not stream_sessions[token].get("live", False):
        await websocket.close(code=4000, reason=f"Stream {token} not found or closed.")
        return

    await manager.connect(websocket, token)

    video_writer = None
    video_path = None
    display_window_name = f"Stream: {token[:8]}..." if ENABLE_VIDEO_DISPLAY else None

    stream_sessions[token]["video_writer"] = None
    stream_sessions[token]["display_window_name"] = display_window_name

    try:
        frame_count = 0
        last_checked = time.time()
        while True:
            try:
                data = await websocket.receive_text()
                frame_count += 1

                if "," not in data:
                    continue

                header, encoded = data.split(",", 1)
                img_data = base64.b64decode(encoded)
                np_arr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                current_time = time.time()
                if current_time - last_checked >= 5:
                    last_checked = current_time
                    _, buffer = cv2.imencode('.png', frame)
                    b64_image = base64.b64encode(buffer).decode('utf-8')
                    verdict_raw = await asyncio.to_thread(detect_smoking, b64_image)
                    if verdict_raw is not None:
                        verdict_lower = verdict_raw.strip().lower()
                        if "yes" in verdict_lower:
                            verdict = "Yes"
                        elif "no" in verdict_lower:
                            verdict = "No"
                        else:
                            verdict = verdict_raw.strip()
                        payload = {
                            "type": "smoking_detection",
                            "timestamp": current_time,
                            "verdict": verdict
                        }
                        await manager.broadcast_json(payload, token)

                if stream_sessions.get(token, {}).get("closing", False):
                    break

                if video_writer is None:
                    try:
                        stream_folder = "stream"
                        os.makedirs(stream_folder, exist_ok=True)
                        video_id = token
                        video_path = os.path.join(stream_folder, f"{video_id}.mp4")
                        height, width, _ = frame.shape
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        fps = 30.0
                        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                        if token in stream_sessions:
                            stream_sessions[token]["video_writer"] = video_writer
                            stream_sessions[token]["video_path"] = video_path
                    except Exception as e:
                        print(f"VideoWriter init error: {e}")
                        continue

                if ENABLE_VIDEO_DISPLAY:
                    try:
                        cv2.imshow(display_window_name, frame)
                        cv2.waitKey(1)
                    except Exception as e:
                        print(f"Display error: {e}")

                try:
                    if video_writer is not None:
                        video_writer.write(frame)
                except Exception as e:
                    print(f"Write error: {e}")

                try:
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    last_frames[token] = buffer.tobytes()
                except Exception as e:
                    print(f"MJPEG frame error: {e}")

            except WebSocketDisconnect:
                raise
            except Exception as e:
                print(f"Frame processing error for {token}: {e}")
                continue

    except WebSocketDisconnect:
        manager.disconnect(websocket, token)
        if not manager.active_connections.get(token):
            if video_writer is not None:
                try:
                    video_writer.release()
                except Exception as e:
                    print(f"VideoWriter release error: {e}")
            if ENABLE_VIDEO_DISPLAY and display_window_name:
                try:
                    cv2.destroyWindow(display_window_name)
                except Exception as e:
                    print(f"Window close error: {e}")
            if token in stream_sessions:
                del stream_sessions[token]
            if token in last_frames:
                del last_frames[token]
