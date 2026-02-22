import os
from decart import DecartClient, models # Added models
from dotenv import load_dotenv

load_dotenv()

decart_client = DecartClient(api_key=os.getenv("DECART_API_KEY"))

# --- CONFIGURATION ---
# We use Lucy 2 Realtime. We set it to 720p for the best balance 
# between quality and speed for your thesis presentation.
RT_MODEL = models.realtime("lucy_2_rt")

async def get_realtime_token():
    try:
        # We can pass specific constraints to the token if needed, 
        # but for now, we just need the session auth.
        token = await decart_client.tokens.create()
        return {
            "apiKey": token.api_key, 
            "expiresAt": token.expires_at,
            "config": {
                "width": RT_MODEL.width,
                "height": RT_MODEL.height,
                "fps": RT_MODEL.fps
            }
        }
    except Exception as e:
        print(f"❌ Decart Token Error: {e}")
        return None