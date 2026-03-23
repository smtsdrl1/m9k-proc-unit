#!/usr/bin/env python3
import http.server
import cgi
import os
import urllib.parse

UPLOAD_DIR = "/home/user/m9k-proc-unit/input_models"
os.makedirs(UPLOAD_DIR, exist_ok=True)

HTML = """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<title>3MF Dosya Yükle</title>
<style>
  body { font-family: Arial; max-width: 600px; margin: 60px auto; background: #1a1a2e; color: #eee; }
  h2 { color: #00d4ff; }
  .box { border: 2px dashed #00d4ff; border-radius: 12px; padding: 40px; text-align: center; }
  input[type=file] { margin: 20px 0; font-size: 16px; color: #eee; }
  button { background: #00d4ff; color: #000; border: none; padding: 12px 30px; font-size: 16px; border-radius: 8px; cursor: pointer; }
  button:hover { background: #00b8d9; }
  .files { margin-top: 30px; background: #16213e; border-radius: 8px; padding: 20px; }
  .files ul { list-style: none; padding: 0; }
  .files li { padding: 6px 0; border-bottom: 1px solid #333; }
  .ok { color: #00ff88; font-weight: bold; }
</style>
</head>
<body>
<h2>3MF Dosya Yükleyici</h2>
<div class="box">
  <form method="POST" enctype="multipart/form-data">
    <p>Birden fazla dosya seçebilirsin</p>
    <input type="file" name="files" accept=".3mf" multiple><br>
    <button type="submit">Yükle</button>
  </form>
</div>
<div class="files">
  <h3>input_models klasöründeki dosyalar:</h3>
  <ul>FILES_LIST</ul>
</div>
</body>
</html>"""

class UploadHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        files = os.listdir(UPLOAD_DIR)
        items = "".join(f"<li>📦 {f}</li>" for f in sorted(files)) or "<li>Henüz dosya yok</li>"
        page = HTML.replace("FILES_LIST", items)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(page.encode("utf-8"))

    def do_POST(self):
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers,
            environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers["Content-Type"]})
        uploaded = []
        files_field = form["files"]
        if not isinstance(files_field, list):
            files_field = [files_field]
        for f in files_field:
            if f.filename:
                dest = os.path.join(UPLOAD_DIR, os.path.basename(f.filename))
                with open(dest, "wb") as out:
                    out.write(f.file.read())
                uploaded.append(f.filename)
        files = os.listdir(UPLOAD_DIR)
        items = "".join(f"<li>📦 {f}</li>" for f in sorted(files))
        msg = f'<p class="ok">✅ Yüklendi: {", ".join(uploaded)}</p>' if uploaded else ""
        page = HTML.replace("FILES_LIST", items).replace('<div class="box">', msg + '<div class="box">')
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(page.encode("utf-8"))

    def log_message(self, format, *args):
        pass

if __name__ == "__main__":
    server = http.server.HTTPServer(("0.0.0.0", 8080), UploadHandler)
    print("Sunucu başladı → http://localhost:8080")
    server.serve_forever()
