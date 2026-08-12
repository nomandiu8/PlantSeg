from http.server import BaseHTTPRequestHandler
import json
import os
import tempfile
import base64
from gradio_client import Client, handle_file

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            body = json.loads(post_data.decode('utf-8'))
            
            image_b64 = body.get('image')
            if not image_b64:
                self.send_error(400, "No image provided")
                return

            # Decode base64 and save to temp file
            import uuid
            img_data = base64.b64decode(image_b64.split(',')[1])
            temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.jpg")
            with open(temp_path, "wb") as f:
                f.write(img_data)
                
            # Connect to Hugging Face API securely using Vercel Environment Variable
            hf_token = os.environ.get("HF_TOKEN")
            client = Client("nomandiu9/diseases_prediction", token=hf_token)
            
            # Make the prediction
            result = client.predict(
                image=handle_file(temp_path),
                api_name="/diagnose"
            )
            
            # Helper to convert output images back to base64 for the frontend
            def encode_file(filepath):
                if not filepath or not os.path.exists(filepath): return None
                with open(filepath, "rb") as f:
                    return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

            # The API returns a tuple of 4 elements. result[1] and result[2] are dicts containing the 'path'
            lesion_path = result[1].get('path') if isinstance(result[1], dict) else result[1]
            cam_path = result[2].get('path') if isinstance(result[2], dict) else result[2]
            
            response = {
                "summary": result[3],
                "lesion_overlay": encode_file(lesion_path),
                "cam_overlay": encode_file(cam_path)
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
