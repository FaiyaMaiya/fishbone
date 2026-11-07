from __future__ import annotations

import json
import ipaddress
import mimetypes
import os
import re
import secrets
import shutil
import threading
from functools import partial
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from starlette.concurrency import run_in_threadpool

load_dotenv(Path(__file__).resolve().parent / ".env")

app = FastAPI(title="FishBone API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://fishbone-mwdb.onrender.com",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "FishBone Backend is LIVE",
        "message": "Creators build Universes, Users just click",
    }


@app.get("/health")
def health_check() -> dict[str, bool]:
    return {"ok": True}


DATA_DIR = Path(os.environ.get("FISHBONE_DATA_DIR", Path(__file__).resolve().parent / "data"))
PUBLISHED_DIR = DATA_DIR / "published"
SHARES_FILE = DATA_DIR / "shares.json"
CHARACTERS_FILE = Path(__file__).resolve().parent / "characters.json"
MAX_SOURCE_BYTES = 500 * 1024 * 1024
MAX_MOVIE_BYTES = 100 * 1024 * 1024
MAX_BUNDLE_BYTES = 1024 * 1024 * 1024
MAX_FACE_IMAGE_BYTES = 5 * 1024 * 1024
CHARACTER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
COLOR_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")
HF_MODEL = os.environ.get("FISHBONE_HF_MODEL", "Wan-AI/Wan2.1-I2V-14B-720P")
HF_PROVIDER = os.environ.get("FISHBONE_HF_PROVIDER", "fal-ai")
SHARE_LOCK = threading.RLock()


def load_shares() -> dict[str, dict[str, Any]]:
    if not SHARES_FILE.exists():
        return {}
    with SHARES_FILE.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise RuntimeError(f"Share registry must be a JSON object: {SHARES_FILE}")
    return data


SHARES = load_shares()
with CHARACTERS_FILE.open(encoding="utf-8") as file:
    CHARACTERS = json.load(file)


async def store_upload(upload: UploadFile, destination: Path, max_bytes: int) -> int:
    if not upload.content_type or not upload.content_type.startswith("video/"):
        raise HTTPException(status_code=415, detail="Only video files can be published.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        with destination.open("wb") as file:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail="The uploaded video exceeds the size limit.")
                file.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    if total == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded video is empty.")
    return total


def public_share(share_id: str, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": share_id,
        "created_at": record["created_at"],
        "duration": record["duration"],
        "source_url": f"/api/shares/{share_id}/source",
        "characters": [
            {
                **character,
                "price": int(character.get("price", 0)),
                "movie_url": f"/api/shares/{share_id}/movie/{character['id']}",
            }
            for character in record["characters"]
        ],
    }


def save_shares() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary_file = SHARES_FILE.with_suffix(".tmp")
    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(SHARES, file, ensure_ascii=False)
    temporary_file.replace(SHARES_FILE)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def is_loopback_request(request: Request) -> bool:
    client = request.client
    if client is None:
        return False
    try:
        return ipaddress.ip_address(client.host).is_loopback
    except ValueError:
        return False


def hf_generation_allowed(request: Request) -> bool:
    if is_loopback_request(request):
        return True
    return os.environ.get("FISHBONE_HF_ALLOW_REMOTE", "").lower() == "true"


@app.get("/api/generation/config")
def generation_config(request: Request) -> dict[str, bool]:
    return {
        "enabled": bool(os.environ.get("HF_TOKEN", "").strip()) and hf_generation_allowed(request)
    }


@app.post("/api/generate-bone-ai")
async def generate_bone_ai(
    request: Request,
    face_image: Annotated[UploadFile, File()],
    char_id: Annotated[str, Form()],
    name: Annotated[str, Form()],
    role: Annotated[str, Form()] = "",
    story: Annotated[str, Form()] = "",
) -> Response:
    if not hf_generation_allowed(request):
        raise HTTPException(
            status_code=403,
            detail="Remote AI generation is disabled unless protected authentication is configured.",
        )
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise HTTPException(status_code=503, detail="Hugging Face generation is not configured.")
    if not CHARACTER_ID_PATTERN.fullmatch(char_id):
        raise HTTPException(status_code=400, detail="Character identifier is invalid.")
    name = name.strip()
    role = role.strip()
    story = story.strip()
    if not name or len(name) > 100 or len(role) > 200 or len(story) > 2000:
        raise HTTPException(status_code=400, detail="Character details are invalid.")
    if face_image.content_type != "image/jpeg":
        raise HTTPException(status_code=415, detail="The face crop must be a JPEG image.")

    try:
        image_bytes = await face_image.read(MAX_FACE_IMAGE_BYTES + 1)
    finally:
        await face_image.close()
    if not image_bytes or len(image_bytes) > MAX_FACE_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="The face crop must be under 5 MB.")
    if not image_bytes.startswith(b"\xff\xd8\xff"):
        raise HTTPException(status_code=400, detail="The face crop is not a valid JPEG image.")

    prompt = (
        f"Create a short cinematic, natural-looking video based on this person in the reference image. "
        f"Preserve their recognizable face and clothing. Character: {name}. "
        f"Role: {role or 'person'}. Story: {story or 'A short personal story.'}"
    )
    try:
        client = InferenceClient(
            model=HF_MODEL,
            provider=HF_PROVIDER,
            token=token,
            timeout=180,
        )
        video_bytes = await run_in_threadpool(
            partial(
                client.image_to_video,
                image=image_bytes,
                prompt=prompt,
                num_frames=49,
                guidance_scale=5.0,
            )
        )
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail="Hugging Face video generation failed or the provider/model is unavailable.",
        ) from error

    if not isinstance(video_bytes, bytes) or len(video_bytes) > MAX_MOVIE_BYTES:
        raise HTTPException(status_code=502, detail="The generated video is empty or exceeds 100 MB.")
    if len(video_bytes) < 12:
        raise HTTPException(status_code=502, detail="The provider returned an invalid video.")
    if b"ftyp" in video_bytes[4:12]:
        media_type = "video/mp4"
    elif video_bytes.startswith(b"\x1a\x45\xdf\xa3"):
        media_type = "video/webm"
    else:
        raise HTTPException(status_code=502, detail="The provider returned an unsupported video format.")
    return Response(
        content=video_bytes,
        media_type=media_type,
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/characters")
def get_all_characters() -> dict[str, Any]:
    return CHARACTERS


@app.get("/api/characters/{character_id}")
def get_character(character_id: str) -> dict[str, Any]:
    character = CHARACTERS.get(character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found.")
    return character


@app.post("/api/bend")
def bend_to_character(data: dict[str, Any]) -> dict[str, Any]:
    character_id = data.get("char_id")
    main_time = data.get("main_time")
    character = CHARACTERS.get(character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found.")
    if not isinstance(main_time, (int, float)) or main_time < 0:
        raise HTTPException(status_code=400, detail="Main video time is invalid.")
    return {
        "message": f"Bending to {character['name']}'s Bone",
        "saved_main_time": main_time,
        "bone": character,
        "blend_back_to": main_time,
    }


@app.post("/api/shares", status_code=201)
async def publish_share(
    characters_json: Annotated[str, Form(alias="characters")],
    duration: Annotated[float, Form()],
    source_video: Annotated[UploadFile, File()],
    bone_movies: Annotated[list[UploadFile], File()],
) -> dict[str, Any]:
    if source_video is None:
        raise HTTPException(status_code=400, detail="Upload the original video before publishing.")
    if not 0 < duration <= 24 * 60 * 60:
        raise HTTPException(status_code=400, detail="Video duration is invalid.")
    try:
        characters_data = json.loads(characters_json)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=400, detail="Character information is invalid.") from error
    if not isinstance(characters_data, list) or not 1 <= len(characters_data) <= 40:
        raise HTTPException(status_code=400, detail="Publish between 1 and 40 characters.")

    movies_by_id: dict[str, UploadFile] = {}
    movie_formats: dict[str, tuple[str, str]] = {}
    for movie in bone_movies:
        filename = Path(movie.filename or "").name
        movie_path = Path(filename)
        character_id = movie_path.stem
        extension = movie_path.suffix.lower()
        media_type = (movie.content_type or "").split(";", 1)[0].lower()
        expected_media_type = {".mp4": "video/mp4", ".webm": "video/webm"}.get(extension)
        if not CHARACTER_ID_PATTERN.fullmatch(character_id) or character_id in movies_by_id:
            raise HTTPException(status_code=400, detail="A character movie has an invalid filename.")
        if expected_media_type is None or media_type != expected_media_type:
            raise HTTPException(
                status_code=415,
                detail="Bone Movies must be uploaded as matching MP4 or WebM files.",
            )
        movies_by_id[character_id] = movie
        movie_formats[character_id] = (extension, media_type)

    characters: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in characters_data:
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail="Character information is invalid.")
        character_id = str(item.get("id", ""))
        name = str(item.get("name", "")).strip()
        if not CHARACTER_ID_PATTERN.fullmatch(character_id) or character_id in seen_ids:
            raise HTTPException(status_code=400, detail="A character identifier is invalid or duplicated.")
        if not name or len(name) > 100:
            raise HTTPException(status_code=400, detail="Every published character needs a name.")
        if character_id not in movies_by_id:
            raise HTTPException(status_code=400, detail=f"Generate a Bone Movie for {name} before publishing.")
        price = item.get("price", 0)
        if type(price) is not int or price not in (0, 2, 5):
            raise HTTPException(status_code=400, detail=f"The unlock price for {name} must be FREE, K2, or K5.")
        raw_moments = item.get("moments")
        if not isinstance(raw_moments, list) or not raw_moments:
            raise HTTPException(status_code=400, detail=f"No detected moments are available for {name}.")
        moments = []
        for moment in raw_moments[:1800]:
            try:
                point = {
                    "time": float(moment["time"]),
                    "x": float(moment["x"]),
                    "y": float(moment["y"]),
                    "w": float(moment["w"]),
                    "h": float(moment["h"]),
                }
            except (KeyError, TypeError, ValueError) as error:
                raise HTTPException(status_code=400, detail=f"Detected moments for {name} are invalid.") from error
            if (
                0 <= point["time"] <= duration
                and -20 <= point["x"] <= 120
                and -20 <= point["y"] <= 120
                and 0 < point["w"] <= 100
                and 0 < point["h"] <= 100
            ):
                moments.append(point)
        if not moments:
            raise HTTPException(status_code=400, detail=f"Detected moments for {name} are invalid.")
        characters.append(
            {
                "id": character_id,
                "name": name,
                "role": str(item.get("role", ""))[:120],
                "story": str(item.get("story", ""))[:4000],
                "price": price,
                "movie_extension": movie_formats[character_id][0],
                "movie_content_type": movie_formats[character_id][1],
                "color": (
                    str(item.get("color"))
                    if COLOR_PATTERN.fullmatch(str(item.get("color", "")))
                    else "#00e5c7"
                ),
                "moments": moments,
            }
        )
        seen_ids.add(character_id)

    share_id = secrets.token_urlsafe(24)
    share_dir = PUBLISHED_DIR / share_id
    try:
        total_bytes = await store_upload(
            source_video,
            share_dir / "source-video",
            min(MAX_SOURCE_BYTES, MAX_BUNDLE_BYTES),
        )
        for character in characters:
            movie = movies_by_id[character["id"]]
            remaining_bytes = MAX_BUNDLE_BYTES - total_bytes
            if remaining_bytes <= 0:
                raise HTTPException(status_code=413, detail="The published video bundle exceeds the size limit.")
            total_bytes += await store_upload(
                movie,
                share_dir / f"{character['id']}{character['movie_extension']}",
                min(MAX_MOVIE_BYTES, remaining_bytes),
            )
        record = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "duration": duration,
            "source_content_type": source_video.content_type or "application/octet-stream",
            "characters": characters,
        }
        with SHARE_LOCK:
            SHARES[share_id] = record
            try:
                save_shares()
            except Exception:
                del SHARES[share_id]
                raise
    except Exception:
        shutil.rmtree(share_dir, ignore_errors=True)
        raise
    return public_share(share_id, record)


def get_share(share_id: str) -> dict[str, Any]:
    record = SHARES.get(share_id)
    if record is None:
        raise HTTPException(status_code=404, detail="This FishBone link was not found.")
    return record


@app.get("/api/shares/{share_id}")
def get_public_share(share_id: str) -> dict[str, Any]:
    return public_share(share_id, get_share(share_id))


class DemoUnlockRequest(BaseModel):
    visitor_id: str


@app.post("/api/shares/{share_id}/demo-unlocks/{character_id}")
def record_demo_unlock(
    share_id: str,
    character_id: str,
    data: DemoUnlockRequest,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,100}", data.visitor_id):
        raise HTTPException(status_code=400, detail="Viewer identifier is invalid.")
    with SHARE_LOCK:
        record = get_share(share_id)
        character = next((item for item in record["characters"] if item["id"] == character_id), None)
        if character is None:
            raise HTTPException(status_code=404, detail="This character was not found.")
        price = int(character.get("price", 0))
        if price == 0:
            raise HTTPException(status_code=400, detail="This character is free and does not need unlocking.")
        unlocks = record.setdefault("demo_unlocks", {})
        visitors = unlocks.setdefault(character_id, [])
        already_unlocked = data.visitor_id in visitors
        if not already_unlocked:
            visitors.append(data.visitor_id)
            save_shares()
        return {
            "unlocked": True,
            "already_unlocked": already_unlocked,
            "price": price,
            "simulation_only": True,
            "message": "Demo unlock only. No payment was processed and no money was earned.",
        }


@app.get("/api/shares/{share_id}/demo-unlocks")
def get_demo_unlock_summary(
    share_id: str,
) -> dict[str, Any]:
    record = get_share(share_id)
    unlocks = record.get("demo_unlocks", {})
    characters = []
    for character in record["characters"]:
        visitor_count = len(set(unlocks.get(character["id"], [])))
        price = int(character.get("price", 0))
        characters.append(
            {
                "id": character["id"],
                "name": character["name"],
                "demo_unlocks": visitor_count,
                "simulated_value": visitor_count * price,
            }
        )
    return {
        "simulation_only": True,
        "total_demo_unlocks": sum(item["demo_unlocks"] for item in characters),
        "simulated_value": sum(item["simulated_value"] for item in characters),
        "characters": characters,
        "notice": "These are simulated unlock values, not payments or earnings.",
    }


@app.get("/api/shares/{share_id}/source")
def get_share_source(share_id: str) -> FileResponse:
    record = get_share(share_id)
    path = PUBLISHED_DIR / share_id / "source-video"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="The shared video is no longer available.")
    media_type = record.get("source_content_type") or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type)


@app.get("/api/shares/{share_id}/movie/{character_id}")
def get_bone_movie(share_id: str, character_id: str) -> FileResponse:
    record = get_share(share_id)
    character = next((item for item in record["characters"] if item["id"] == character_id), None)
    if character is None:
        raise HTTPException(status_code=404, detail="This character movie was not found.")
    extension = character.get("movie_extension", ".webm")
    media_type = character.get("movie_content_type", "video/webm")
    if extension not in {".mp4", ".webm"} or media_type not in {"video/mp4", "video/webm"}:
        raise HTTPException(status_code=500, detail="The published character movie format is invalid.")
    path = PUBLISHED_DIR / share_id / f"{character_id}{extension}"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="This character movie is no longer available.")
    return FileResponse(path, media_type=media_type)
