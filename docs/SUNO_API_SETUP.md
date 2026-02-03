# SUNO API Setup Guide

This guide explains how to set up and use the unofficial SUNO API to generate songs for the APT attack evaluation.

## Prerequisites

- Node.js 18+
- A Suno.ai account (free tier works)
- 2Captcha account for CAPTCHA solving (~$0.50-$1 per solution)
- Chrome or Firefox browser

## Step 1: Set Up suno-api

### Option A: Local Installation (Recommended)

```bash
# Clone the unofficial SUNO API
git clone https://github.com/gcui-art/suno-api.git
cd suno-api

# Install dependencies
npm install
```

### Option B: Docker Installation

```bash
docker pull gcuiart/suno-api
```

## Step 2: Get Your SUNO Cookie

1. Go to [app.suno.ai](https://app.suno.ai) and log in
2. Open Developer Tools (F12 or right-click -> Inspect)
3. Go to the **Network** tab
4. Refresh the page
5. Look for a request containing `client?_clerk_js_version`
6. Click on it and find **Headers** -> **Request Headers** -> **Cookie**
7. Copy the entire Cookie value

**Example cookie format:**
```
__client=eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...; __client_uat=1234567890; ...
```

## Step 3: Get 2Captcha API Key

1. Go to [2captcha.com](https://2captcha.com) and create an account
2. Add funds (~$3 minimum recommended)
3. Go to **API Settings** and copy your API key

## Step 4: Configure Environment Variables

Create a `.env` file in the `suno-api` directory:

```bash
# Required
SUNO_COOKIE=your_cookie_value_here
TWOCAPTCHA_KEY=your_2captcha_api_key

# Optional (defaults shown)
BROWSER=chromium
BROWSER_HEADLESS=true
BROWSER_LOCALE=en
BROWSER_GHOST_CURSOR=false
```

## Step 5: Start the API Server

```bash
# In the suno-api directory
npm run dev
```

The API will start on `http://localhost:3000`

### Verify it's working:

```bash
curl http://localhost:3000/api/get_limit
```

Expected response:
```json
{"credits_left": 50, "period": "day", "monthly_limit": 50}
```

## Step 6: Generate Songs

### Using the provided script:

```bash
cd CMU_phoneme

# Check what will be generated (dry run)
python suno_generator.py --dry-run

# Generate 3 samples for all songs
python suno_generator.py --k 3

# Generate samples for specific songs only
python suno_generator.py --songs dna lose_yourself humble --k 3

# Use a different API URL
python suno_generator.py --api-url http://your-server:3000 --k 3
```

### Available songs:

| Key | Song | Artist | Genre |
|-----|------|--------|-------|
| `dna` | DNA | Kendrick Lamar | Rap |
| `lose_yourself` | Lose Yourself | Eminem | Rap |
| `humble` | HUMBLE | Kendrick Lamar | Rap |
| `espresso` | Espresso | Sabrina Carpenter | Pop |
| `let_it_be` | Let It Be | The Beatles | Pop |
| `we_will_rock_you` | We Will Rock You | Queen | Pop |
| `apt` | APT | ROSÉ & Bruno Mars | Korean |
| `gangnam_style` | Gangnam Style | PSY | Korean |
| `gentleman` | GENTLEMAN | PSY | Korean |
| `jingle_bell` | Jingle Bells | Traditional | Christmas |
| `jingle_bell_rock` | Jingle Bell Rock | Bobby Helms | Christmas |

## Step 7: Download Generated Audio

The generated songs will have URLs in the JSON output files. Download them:

```bash
# The script saves results to CMU_phoneme/generated_songs/
# Each file contains audio_url fields

# Example using curl
curl -o song.mp3 "https://cdn.suno.ai/audio/..."
```

### Batch download script:

```python
import json
import requests
from pathlib import Path

def download_songs(results_dir):
    results_dir = Path(results_dir)
    for json_file in results_dir.glob("*.json"):
        if "summary" in json_file.name:
            continue

        with open(json_file) as f:
            data = json.load(f)

        api_response = data.get("api_response", [])
        for i, clip in enumerate(api_response):
            audio_url = clip.get("audio_url")
            if audio_url:
                filename = f"{json_file.stem}_clip{i}.mp3"
                print(f"Downloading: {filename}")
                resp = requests.get(audio_url)
                with open(results_dir / filename, 'wb') as f:
                    f.write(resp.content)

download_songs("CMU_phoneme/generated_songs")
```

## API Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/get_limit` | GET | Check remaining credits |
| `/api/custom_generate` | POST | Generate with custom lyrics |
| `/api/get?ids=` | GET | Get song info by IDs |
| `/api/generate` | POST | Generate from text prompt |
| `/api/generate_lyrics` | POST | Generate lyrics |
| `/api/extend_audio` | POST | Extend audio duration |

## Troubleshooting

### "Cookie expired" error
- Re-extract your cookie from Suno.ai (they expire periodically)

### "CAPTCHA required"
- Ensure your 2Captcha key is set and has funds
- Check 2Captcha dashboard for errors

### "Rate limited"
- Suno has daily limits (~50 credits for free tier)
- Wait for daily reset or upgrade your Suno account

### Generation timeout
- Increase timeout in script (default 5 minutes)
- Check if Suno servers are busy

### Connection refused
- Ensure suno-api is running: `npm run dev`
- Check firewall settings

## Cost Estimation

| Component | Cost |
|-----------|------|
| Suno Free Tier | 50 credits/day (~5-10 songs) |
| Suno Pro | $8/month, 500 credits |
| 2Captcha | ~$0.50-$1 per CAPTCHA |

For generating k=3 samples for all 11 songs (33 total):
- Approximately 330 credits needed
- Suno Pro subscription recommended
- ~$3-5 in 2Captcha costs

## Output Structure

```
CMU_phoneme/generated_songs/
├── dna_sample1.json
├── dna_sample2.json
├── dna_sample3.json
├── lose_yourself_sample1.json
├── ...
└── generation_summary_20250203_120000.json
```

Each JSON file contains:
```json
{
  "song_key": "dna",
  "song_name": "DNA",
  "artist": "Kendrick Lamar",
  "genre": "Rap",
  "sample_num": 1,
  "timestamp": "2025-02-03T12:00:00",
  "api_response": [
    {
      "id": "clip_abc123",
      "audio_url": "https://cdn.suno.ai/...",
      "status": "complete",
      ...
    }
  ]
}
```

## Next Steps

After generating songs:

1. Move audio files to `AudioJudge/audio_mp3/[song]/generated/`
2. Run AudioJudge evaluation: `cd AudioJudge && python batch_audiojudge.py`
3. Generate heatmaps: `python generate_heatmap_final.py`
