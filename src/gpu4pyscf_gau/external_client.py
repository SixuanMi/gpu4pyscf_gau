"""Profiled callback; import/startup and round-trip timings saved separately."""
import time
SCRIPT_START = time.perf_counter()
import json
import os
from pathlib import Path
import socket
import sys
import traceback


def request(payload, socket_path=None):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(float(os.environ.get('GPU_GAU_REQUEST_TIMEOUT', '43200')))
        sock.connect(socket_path or os.environ['GPU_GAU_SOCKET'])
        sock.sendall(json.dumps(payload).encode() + b'\n')
        response = json.loads(sock.makefile('rb').readline())
    if not response.get('ok'):
        raise RuntimeError(response.get('error', 'GPU worker failed'))
    return response


def main():
    imports_done = time.perf_counter()
    # Gaussian 16 supplies layer, input, output, message, fchk, matel.
    if len(sys.argv) < 5:
        raise ValueError('Expected layer InputFile OutputFile MsgFile [FChkFile MatElFile]')
    layer, inp, out, msg = sys.argv[1:5]
    Path(out).unlink(missing_ok=True)
    try:
        request_start = time.perf_counter()
        response = request(dict(operation='external', layer=layer,
                                input=str(Path(inp).resolve()), output=str(Path(out).resolve())))
        request_end = time.perf_counter()
        deriv = response['result']['derivative_order']
        Path(msg).write_text(f'GPU4PySCF: SCF converged; requested derivative order {deriv} supplied.\n'
                             + json.dumps(response['summary']) + '\n')
        if os.environ.get('GPU_GAU_CLIENT_TRACE'):
            with open(os.environ['GPU_GAU_CLIENT_TRACE'],'a') as f:
                f.write(json.dumps(dict(evaluation=response['summary']['evaluation'],derivative_order=deriv,
                    imports_seconds=imports_done-SCRIPT_START,round_trip_seconds=request_end-request_start,
                    script_seconds=time.perf_counter()-SCRIPT_START,started_monotonic=SCRIPT_START))+'\n')
    except Exception:
        Path(msg).write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
