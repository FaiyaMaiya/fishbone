# FishBone backend

The API stores published videos and share records on the server. Keep that storage on a persistent disk when deploying; the public share links stop working if the `data` directory is lost.

## Run locally

From this directory, install `requirements.txt` and start the API:

```powershell
python -m pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000
```

## Optional Hugging Face video generation

The default Bone Movie remains a local highlight made from detected moments. To enable the optional image-to-video generation locally, add a Hugging Face token with Inference Providers permission to `backend/.env`:

```env
HF_TOKEN=your_hugging_face_token
```

Keep the token private and out of source control. The configured Wan image-to-video model uses Hugging Face's `fal-ai` provider. Hugging Face Inference Providers are not free for free-tier accounts; check current provider pricing before enabling generation. Every attempted generation asks for confirmation because it sends a cropped face image and that character's name, role, and story to Hugging Face and its inference provider, and may incur charges. The original source video is not sent. If the provider is unavailable or generation fails, FishBone falls back to the existing local highlight.

The billable generation endpoint only accepts loopback requests by default. Do not enable remote access with `FISHBONE_HF_ALLOW_REMOTE` on a public backend unless you have added authentication, rate limiting, and spending controls.

For production, use HTTPS and configure a persistent volume. Do not commit `.env` or the published `data` directory.

When the frontend and API are on different hosts, define `window.FISHBONE_API_BASE` before the app script loads (for example, in the hosting platform's frontend configuration) to the API's HTTPS origin. The share links use the current frontend URL and a public share ID.

Creator tools and publishing do not require a login. Anyone who can reach the API can publish videos; configure network access accordingly. Published source videos and Bone Movies are publicly accessible to anyone with the share link.

Bone Movies are recorded as MP4 when the browser supports MP4 recording, with WebM as a fallback. The API preserves the uploaded MP4 or WebM format and serves it with the matching content type. Older share records without format metadata continue to use WebM.
