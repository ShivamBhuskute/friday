import socket
import wave
import os

# Configuration
PORT = 5000
SAMPLE_RATE = 16000
SAVE_DIR = "deteced_recordings"

os.makedirs(SAVE_DIR, exist_ok=True)


def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', PORT))
    server_socket.listen(1)

    print(f"[*] Listening on port {PORT}...")

    file_counter = 1

    try:
        while True:
            # The ESP32 opens one connection per wake-word-triggered utterance,
            # streams audio, then closes the socket once it detects 500 ms of
            # silence. So: accept a connection, buffer everything it sends,
            # and only write the file once that connection closes.
            print("[*] Waiting for the messsage...")
            conn, addr = server_socket.accept()
            print(f"[+] Message recieved at {addr}")

            audio_buffer = bytearray()
            try:
                while True:
                    data = conn.recv(4096)
                    if not data:
                        break  # ESP32 closed the socket -> utterance is over
                    audio_buffer.extend(data)
            finally:
                conn.close()

            if len(audio_buffer) == 0:
                print("[!] Connection closed with no audio received, skipping.")
                continue

            filename = os.path.join(SAVE_DIR, f"sample_{file_counter:03d}.wav")
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)       # Mono
                wf.setsampwidth(2)       # 16-bit
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(bytes(audio_buffer))

            duration_s = len(audio_buffer) / 2 / SAMPLE_RATE
            print(f"[Saved] {filename} ({duration_s:.2f}s) - Say the next word!")
            file_counter += 1

    except KeyboardInterrupt:
        print("\n[*] Stopping server...")
    finally:
        server_socket.close()


if __name__ == "__main__":
    start_server()
