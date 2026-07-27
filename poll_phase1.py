#!/usr/bin/env python3
"""Background polling script for Jules session status - runs until completion."""
import json
import subprocess
import time
import sys
from datetime import datetime

SESSION_ID = "3994777587211401140"
JULES_SERVER = "/home/rahul-reddy/.hermes/scripts/jules_mcp_server.py"
POLL_INTERVAL = 60  # 1 minute

def send_mcp_request(proc, request):
    """Send a JSON-RPC request to the Jules MCP server."""
    msg_bytes = json.dumps(request).encode()
    header = f'Content-Length: {len(msg_bytes)}\r\n\r\n'.encode()
    proc.stdin.write(header + msg_bytes)
    proc.stdin.flush()
    
    header_data = b''
    while True:
        line = proc.stdout.readline()
        if not line or line == b'\r\n':
            break
        header_data += line
    
    length = 0
    for line in header_data.decode().split('\r\n'):
        if line.lower().startswith('content-length:'):
            length = int(line.split(':')[1].strip())
    
    if length > 0:
        body = proc.stdout.read(length)
        return json.loads(body.decode())
    return None

def get_session_status(proc):
    """Get the current session state."""
    request = {
        'jsonrpc': '2.0',
        'id': int(time.time() * 1000),
        'method': 'tools/call',
        'params': {
            'name': 'jules_get_session',
            'arguments': {'session_id': SESSION_ID}
        }
    }
    response = send_mcp_request(proc, request)
    if response and 'result' in response:
        content = response['result']['content'][0]['text']
        return json.loads(content)
    return None

def format_status(session_data):
    """Format session data for display."""
    if not session_data:
        return "Unknown"
    
    state = session_data.get('state', 'UNKNOWN')
    update_time = session_data.get('updateTime', 'N/A')
    url = session_data.get('url', 'N/A')
    
    return f"State: {state}\nLast Update: {update_time}\nURL: {url}"

def main():
    print(f"Starting Jules session poller for session {SESSION_ID}")
    print(f"Polling every {POLL_INTERVAL} seconds")
    print("Press Ctrl+C to stop\n")
    
    proc = subprocess.Popen(
        ['python3', '/home/rahul-reddy/.hermes/scripts/jules_mcp_server.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    try:
        # Initialize connection
        init_msg = {
            'jsonrpc': '2.0',
            'id': 1,
            'method': 'initialize',
            'params': {
                'protocolVersion': '2024-11-05',
                'capabilities': {},
                'clientInfo': {'name': 'poller', 'version': '1.0'}
            }
        }
        send_mcp_request(proc, init_msg)
        
        init_notif = {
            'jsonrpc': '2.0',
            'method': 'notifications/initialized',
            'params': {}
        }
        send_mcp_request(proc, init_notif)
        
        poll_count = 0
        while True:
            poll_count += 1
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n[{timestamp}] Poll #{poll_count}")
            
            session_data = get_session_status(proc)
            if session_data:
                print(format_status(session_data))
                
                state = session_data.get('state', 'UNKNOWN')
                if state in ('COMPLETED', 'FAILED'):
                    print(f"\n{'='*50}")
                    print(f"SESSION {state}!")
                    print(f"{'='*50}")
                    print(f"Final Status: {state}")
                    print(f"Update Time: {session_data.get('updateTime', 'N/A')}")
                    print(f"URL: {session_data.get('url', 'N/A')}")
                    
                    if state == 'COMPLETED':
                        outputs = session_data.get('outputs', [])
                        if outputs:
                            pr_url = outputs[0].get('pullRequest', {}).get('url', 'N/A')
                            print(f"PR URL: {pr_url}")
                    break
            else:
                print("Failed to get session status")
            
            print(f"Next poll in {POLL_INTERVAL} seconds...")
            time.sleep(POLL_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\nPolling stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        proc.terminate()
        proc.wait(timeout=5)

if __name__ == '__main__':
    main()