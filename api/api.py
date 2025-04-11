import os
import shutil
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
from typing import Optional

# Create a directory to store uploaded audio files
AUDIO_DIR = "audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)

app = FastAPI()

@app.get("/health")
async def health_check():
    """Returns a simple health check response."""
    return JSONResponse(content={"status": "ok"})

@app.post("/clone_voice")
async def clone_voice(
    input1_type: str = Form(...),
    voice_ref_type: str = Form(...),
    input1_text: Optional[str] = Form(None),
    input1_audio: Optional[UploadFile] = File(None),
    voice_ref_audio: Optional[UploadFile] = File(None)
):
    """
    Receives voice cloning parameters, saves the reference audio,
    and returns a predefined audio file as response.
    """
    saved_ref_path = None
    try:
        # Save the uploaded voice reference audio file (optional, depending on type)
        if voice_ref_audio:
            saved_ref_path = os.path.join(AUDIO_DIR, f"reference_{voice_ref_audio.filename}")
            with open(saved_ref_path, "wb") as buffer:
                shutil.copyfileobj(voice_ref_audio.file, buffer)
            print(f"Saved reference audio to: {saved_ref_path}") # Added print for debugging
        else:
            # If voice ref type requires audio but none was provided
            raise HTTPException(status_code=400, detail="Error during saving")

        # --- Placeholder for actual voice cloning logic ---
        # This is where the cloning would happen, using input1 and the voice reference.
        # The result of the cloning should be the path to the output audio file.
        # For now, we hardcode the response file path.
        output_audio_path = saved_ref_path
        # ---------------------------------------------------

        # Check if the output file exists before returning
        if not os.path.exists(output_audio_path):
            # You might want to create a dummy file for testing if it doesn't exist
            # with open(output_audio_path, 'w') as f:
            #     f.write("dummy content") # Or create a silent audio file
            # print(f"Warning: Output file {output_audio_path} not found. A dummy file might be needed.")
            raise HTTPException(status_code=404, detail=f"Output audio file not found at {output_audio_path}")

        # Return the generated audio file
        return FileResponse(path=output_audio_path, media_type='audio/mpeg', filename="cloned_voice.mp3")

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly
        raise http_exc
    except Exception as e:
        # Log the error for debugging
        print(f"Error in /clone_voice: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
    finally:
        # Ensure uploaded files are closed
        if voice_ref_audio:
            await voice_ref_audio.close()
        if input1_audio:
            await input1_audio.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
