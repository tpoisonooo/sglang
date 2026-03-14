"""
Simple HTTP server to receive and log user queries from SGLang server.
Logs are saved in JSONL format similar to submit/math.jsonl
"""

import json
import os
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading


# Configuration
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, "received_queries.jsonl")
PORT = int(os.getenv("LOG_SERVER_PORT", "10005"))

# Global counter for index
query_counter = 0
counter_lock = threading.Lock()


class LogHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress default logging
        pass

    def _send_response(self, status_code=200, message="OK"):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": message}).encode())

    def do_POST(self):
        global query_counter

        if self.path != "/log":
            self._send_response(404, "Not Found")
            return

        try:
            # Read request body
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode("utf-8"))

            # Generate index
            with counter_lock:
                idx = query_counter
                query_counter += 1

            # Extract question from messages
            messages = data.get("messages", [])
            question = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    if isinstance(content, str):
                        question += content + "\n"
                    elif isinstance(content, list):
                        # Handle multi-modal content
                        for item in content:
                            if item.get("type") == "text":
                                question += item.get("text", "") + "\n"

            # Create log entry similar to math.jsonl format
            log_entry = {
                "index": idx,
                "question": question.strip(),
                "source": "user_query",
                "model": data.get("model", "unknown"),
                "timestamp": data.get("timestamp", time.time()),
                "received_at": datetime.now().isoformat(),
            }

            # Append to JSONL file
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            print(f"[INFO] Logged query #{idx} to {LOG_FILE}")
            self._send_response(200, "Logged successfully")

        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON: {e}")
            self._send_response(400, "Invalid JSON")
        except Exception as e:
            print(f"[ERROR] Failed to process request: {e}")
            self._send_response(500, f"Internal error: {e}")

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "status": "running",
            "log_file": LOG_FILE,
            "total_queries": query_counter
        }).encode())


def run_server(port=PORT):
    server = HTTPServer(("0.0.0.0", port), LogHandler)
    print(f"[INFO] Log server started on http://0.0.0.0:{port}")
    print(f"[INFO] Logging to: {LOG_FILE}")
    print(f"[INFO] Health check: curl http://localhost:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down server...")
        server.shutdown()


if __name__ == "__main__":
    run_server()


#   curl -X POST http://localhost:30000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{"model":"test","messages":[{"role":"user","content":"Hello, what is 2+2?"}]}'