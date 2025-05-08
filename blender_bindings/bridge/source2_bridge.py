#!/usr/bin/env python3
import asyncio
import struct
import sys

HOST = '127.0.0.1'
PORT = 55555  # You can change this if needed

async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    addr = writer.get_extra_info('peername')
    print(f"[+] Connected from {addr}")
    try:
        # Read 4-byte little-endian length header
        header = await reader.readexactly(4)
        message_size = struct.unpack('<I', header)[0]
        # Read the message body
        data = await reader.readexactly(message_size)
        message = data.decode('utf-8')
        print(f"[>] Received: {message!r}")
    except asyncio.IncompleteReadError:
        # Connection closed before header or body -> liveness check
        print("[>] Received liveness check")
    except Exception as e:
        print(f"[!] Error: {e}")
    finally:
        writer.close()
        await writer.wait_closed()
        print("[*] Connection closed")

async def main():
    server = await asyncio.start_server(handle_client, HOST, PORT)
    addr = server.sockets[0].getsockname()
    print(f"[*] Serving on {addr}")
    async with server:
        await server.serve_forever()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[*] Server stopped by user")
        sys.exit(0)